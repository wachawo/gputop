#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only localhost probing of llama.cpp server, Ollama, LM Studio and vLLM, with a cache."""

import hashlib
import http.client
import json
import logging
import os
import re
import time
import traceback

# Local imports
from gputop import constants
from gputop import formatting

# Transport
PROBE_HOST = "127.0.0.1"
PROBE_TIMEOUT = 0.5
MAX_BODY_BYTES = 4 * constants.MIB
READ_CHUNK_BYTES = 64 * 1024
STATUS_OK = 200
STATUS_UNAUTHORIZED = 401
STATUS_UNAVAILABLE = 503

# Cache
POSITIVE_TTL = 3.0
NEGATIVE_TTL = 30.0
LOADING_TTL = 3.0
CACHE_PRUNE_SIZE = 256

# Response fields
MODALITY_TYPES = {
    "text": "llm",
    "decisions": "decision",
    "embeddings": "embedding",
    "embedding": "embedding",
}
GEN_RATE_PATTERN = re.compile(r"^llamacpp:predicted_tokens_seconds\s+(\S+)", re.MULTILINE)
PROMPT_RATE_PATTERN = re.compile(r"^llamacpp:prompt_tokens_seconds\s+(\S+)", re.MULTILINE)

logger = logging.getLogger(__name__)


def create_server(kind: str, port: int) -> dict:
    """Create a server record.

    Keys: kind (a SERVER_* value), url (str), n_ctx (int or None), slots_total (int or None),
    slots_busy (int or None), gen_tok_s (float or None), prompt_tok_s (float or None),
    model_path (str or None), models (list of server model dicts).
    """
    return {
        "kind": kind,
        "url": f"http://{PROBE_HOST}:{port}",
        "n_ctx": None,
        "slots_total": None,
        "slots_busy": None,
        "gen_tok_s": None,
        "prompt_tok_s": None,
        "model_path": None,
        "models": [],
    }


def create_server_model(name: str) -> dict:
    """Create a normalised model entry of a server: name, type, quant, ctx, size, size_vram, path.

    Everything but the name starts as None (unknown).
    """
    return {
        "name": name,
        "type": None,
        "quant": None,
        "ctx": None,
        "size": None,
        "size_vram": None,
        "path": None,
    }


def create_reply() -> dict:
    """Create a reply: status (int or None), data (JSON value or None), text (str or None)."""
    return {"status": None, "data": None, "text": None}


def create_outcome(server: dict | None, unauthorized: bool) -> dict:
    """Create a cache entry: time (monotonic seconds), server (or None), unauthorized (bool).

    `loading` is set by the probe when the server answered 503: it is still loading its
    model and is asked again soon instead of being written off for NEGATIVE_TTL.
    """
    return {
        "time": time.monotonic(),
        "server": server,
        "unauthorized": unauthorized,
        "loading": False,
    }


def open_prober() -> dict:
    """Create the prober state: cache maps (port, expect_vllm, origin, key digest) to an outcome.

    A port is "locked" while the outcome stored for it is unauthorized; a later probe that
    succeeds or fails differently replaces that outcome.
    """
    return {"cache": {}}


def limit_socket_wait(stream, deadline: float) -> None:
    """Cap the next wait on the socket at the time left for the request.

    Raises TimeoutError when no time is left, so the caller's transport handling ends it.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("probe deadline passed")
    if stream is not None:
        stream.settimeout(remaining)


def read_body(stream, response, deadline: float) -> bytes:
    """Read at most MAX_BODY_BYTES of the body, in pieces, within the request deadline."""
    chunks = []
    size = 0
    while size < MAX_BODY_BYTES and not response.isclosed():
        limit_socket_wait(stream, deadline)
        chunk = response.read1(min(READ_CHUNK_BYTES, MAX_BODY_BYTES - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def exchange_request(connection, path: str, headers: dict, reply: dict) -> None:
    """Send the GET and fill status and text; PROBE_TIMEOUT bounds the whole exchange."""
    deadline = time.monotonic() + PROBE_TIMEOUT
    connection.request("GET", path, headers=headers)
    stream = connection.sock
    limit_socket_wait(stream, deadline)
    response = connection.getresponse()
    reply["status"] = response.status
    if response.status != STATUS_OK:
        return
    body = read_body(stream, response, deadline)
    reply["text"] = body.decode("utf-8", errors="replace")


def fetch_reply(port: int, path: str, api_key: str) -> dict:
    """One GET on the local port. Never raises; a failure leaves the reply empty.

    A header value the HTTP client refuses (a key with a line break or a character outside
    Latin-1) is a transport failure like any other; its message repeats the key, so it is
    dropped unseen.
    """
    reply = create_reply()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    connection = http.client.HTTPConnection(PROBE_HOST, port, timeout=PROBE_TIMEOUT)
    try:
        exchange_request(connection, path, headers, reply)
    except (OSError, http.client.HTTPException, ValueError):
        reply["status"] = None
        reply["text"] = None
    finally:
        connection.close()
    if reply["text"] is not None:
        try:
            reply["data"] = json.loads(reply["text"])
        except (ValueError, RecursionError):
            reply["data"] = None
    return reply


def convert_to_text(value) -> str | None:
    """Non-empty string or None."""
    if isinstance(value, str) and value:
        return value
    return None


def get_field(container, key: str):
    """container[key] when container is a JSON object, else None."""
    if isinstance(container, dict):
        return container.get(key)
    return None


def parse_rate(pattern, text: str) -> float | None:
    """Value of one Prometheus sample line, None when absent or not a number."""
    rate_match = pattern.search(text)
    if rate_match is None:
        return None
    try:
        return float(rate_match.group(1))
    except ValueError:
        return None


def build_llama_model(item, model_path: str | None) -> dict:
    """Model entry from one item of the /v1/models list of a llama.cpp server."""
    identifier = convert_to_text(get_field(item, "id")) or model_path
    name = os.path.basename(identifier) if identifier else ""
    entry = create_server_model(name or "?")
    modalities = get_field(get_field(item, "architecture"), "output_modalities")
    if isinstance(modalities, list) and modalities and isinstance(modalities[0], str):
        entry["type"] = MODALITY_TYPES.get(modalities[0], modalities[0])
    meta = get_field(item, "meta")
    ftype = get_field(meta, "ftype")
    entry["quant"] = None if ftype is None else str(ftype)
    entry["ctx"] = formatting.convert_to_int(get_field(meta, "n_ctx"))
    entry["size"] = formatting.convert_to_int(get_field(meta, "size"))
    entry["path"] = model_path
    return entry


def list_llama_models(port: int, api_key: str, server: dict, previous: dict | None) -> list[dict]:
    """Model identity of a llama.cpp server; refetched only when the model path changed.

    An empty previous list is never reused: it may be a request that timed out.
    """
    if previous is not None and previous["kind"] == constants.SERVER_LLAMA:
        if previous["models"] and previous["model_path"] == server["model_path"]:
            return previous["models"]
    reply = fetch_reply(port, "/v1/models", api_key)
    items = get_field(reply["data"], "data")
    if not isinstance(items, list):
        return []
    return [build_llama_model(item, server["model_path"]) for item in items]


def read_llama_activity(port: int, api_key: str, server: dict) -> None:
    """Fill slot counts and throughput from /slots and /metrics when they are enabled."""
    slots_reply = fetch_reply(port, "/slots", api_key)
    if slots_reply["status"] is None:
        return
    slots = slots_reply["data"]
    if isinstance(slots, list):
        busy = [slot for slot in slots if get_field(slot, "is_processing") is True]
        server["slots_total"] = len(slots)
        server["slots_busy"] = len(busy)
    metrics = fetch_reply(port, "/metrics", api_key)
    if metrics["text"] is not None and metrics["data"] is None:
        server["gen_tok_s"] = parse_rate(GEN_RATE_PATTERN, metrics["text"])
        server["prompt_tok_s"] = parse_rate(PROMPT_RATE_PATTERN, metrics["text"])


def build_llama_server(port: int, api_key: str, props: dict, previous: dict | None) -> dict:
    """Server record of a llama.cpp server from its /props answer and three more requests."""
    server = create_server(constants.SERVER_LLAMA, port)
    server["model_path"] = convert_to_text(props.get("model_path"))
    settings = props.get("default_generation_settings")
    n_ctx = formatting.convert_to_int(get_field(settings, "n_ctx"))
    if n_ctx is None:
        n_ctx = formatting.convert_to_int(props.get("n_ctx"))
    server["n_ctx"] = n_ctx
    server["slots_total"] = formatting.convert_to_int(props.get("total_slots"))
    server["models"] = list_llama_models(port, api_key, server, previous)
    read_llama_activity(port, api_key, server)
    return server


def build_ollama_server(port: int, items: list) -> dict:
    """Server record of an Ollama server from the models list of /api/ps."""
    server = create_server(constants.SERVER_OLLAMA, port)
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = create_server_model(convert_to_text(item.get("name")) or "?")
        entry["type"] = "llm"
        entry["quant"] = convert_to_text(get_field(item.get("details"), "quantization_level"))
        entry["ctx"] = formatting.convert_to_int(item.get("context_length"))
        entry["size"] = formatting.convert_to_int(item.get("size"))
        entry["size_vram"] = formatting.convert_to_int(item.get("size_vram"))
        server["models"].append(entry)
    return server


def build_lmstudio_instance(model: dict, instance) -> dict:
    """Model entry of one loaded instance from the current LM Studio API."""
    name = convert_to_text(get_field(instance, "id")) or convert_to_text(model.get("key"))
    entry = create_server_model(name or "?")
    model_type = convert_to_text(model.get("type"))
    decision_type = convert_to_text(model.get("decision_type"))
    if model_type == "decision" and decision_type:
        model_type = f"decision ({decision_type})"
    entry["type"] = model_type
    entry["quant"] = convert_to_text(get_field(model.get("quantization"), "name"))
    entry["ctx"] = formatting.convert_to_int(get_field(get_field(instance, "config"), "context_length"))
    entry["size"] = formatting.convert_to_int(model.get("size_bytes"))
    return entry


def build_lmstudio_server(port: int, items: list) -> dict:
    """Server record from the models list of /api/v1/models; unloaded models are skipped."""
    server = create_server(constants.SERVER_LMSTUDIO, port)
    for model in items:
        instances = get_field(model, "loaded_instances")
        if not isinstance(instances, list):
            continue
        for instance in instances:
            server["models"].append(build_lmstudio_instance(model, instance))
    return server


def build_lmstudio_legacy_server(port: int, items: list) -> dict:
    """Server record from the data list of /api/v0/models; only loaded items count."""
    server = create_server(constants.SERVER_LMSTUDIO, port)
    for item in items:
        if get_field(item, "state") != "loaded":
            continue
        entry = create_server_model(convert_to_text(item.get("id")) or "?")
        entry["type"] = convert_to_text(item.get("type"))
        entry["quant"] = convert_to_text(item.get("quantization"))
        ctx = formatting.convert_to_int(item.get("loaded_context_length"))
        if ctx is None:
            ctx = formatting.convert_to_int(item.get("max_context_length"))
        entry["ctx"] = ctx
        server["models"].append(entry)
    return server


def probe_lmstudio(port: int) -> dict | None:
    """LM Studio by response shape: the current endpoint first, then the older one."""
    current = fetch_reply(port, "/api/v1/models", "")
    if current["status"] is None:
        return None
    items = get_field(current["data"], "models")
    if isinstance(items, list):
        return build_lmstudio_server(port, items)
    legacy = fetch_reply(port, "/api/v0/models", "")
    items = get_field(legacy["data"], "data")
    if isinstance(items, list):
        return build_lmstudio_legacy_server(port, items)
    return None


def probe_vllm(port: int, api_key: str) -> dict:
    """Single request for a process with a vLLM signature: the list of served model ids."""
    reply = fetch_reply(port, "/v1/models", api_key)
    if reply["status"] == STATUS_UNAUTHORIZED:
        return create_outcome(None, True)
    items = get_field(reply["data"], "data")
    if not isinstance(items, list):
        return create_outcome(None, False)
    server = create_server(constants.SERVER_VLLM, port)
    for item in items:
        identifier = get_field(item, "id")
        if not isinstance(identifier, str) or not identifier.strip():
            return create_outcome(None, False)
        server["models"].append(create_server_model(identifier))
    return create_outcome(server, False)


def is_llama_props(props_body) -> bool:
    """Tell whether a /props answer has the shape of a llama.cpp server."""
    if not isinstance(props_body, dict):
        return False
    return "default_generation_settings" in props_body or "model_path" in props_body


def probe_generic(port: int, api_key: str, previous: dict | None) -> dict:
    """Detection order on one port: llama.cpp server, Ollama, LM Studio.

    A port that gives no HTTP answer at all ends the probe at once: the remaining endpoints
    would only wait out their timeouts on the sampling thread.
    """
    props = fetch_reply(port, "/props", api_key)
    if props["status"] is None:
        return create_outcome(None, False)
    if props["status"] == STATUS_UNAUTHORIZED:
        return create_outcome(None, True)
    if props["status"] == STATUS_UNAVAILABLE:
        outcome = create_outcome(None, False)
        outcome["loading"] = True
        return outcome
    props_body = props["data"]
    if is_llama_props(props_body):
        return create_outcome(build_llama_server(port, api_key, props_body, previous), False)
    running = fetch_reply(port, "/api/ps", "")
    if running["status"] is None:
        return create_outcome(None, False)
    items = get_field(running["data"], "models")
    if isinstance(items, list):
        return create_outcome(build_ollama_server(port, items), False)
    return create_outcome(probe_lmstudio(port), False)


def get_key_digest(api_key: str) -> str | None:
    """SHA-256 of the key for the cache key, None without a key; the key itself is never kept."""
    if not api_key:
        return None
    encoded = api_key.encode("utf-8", errors="replace")
    hasher = hashlib.sha256(encoded)
    return hasher.hexdigest()


def prune_cache(cache: dict) -> None:
    """Drop expired outcomes once the cache has grown large."""
    if len(cache) <= CACHE_PRUNE_SIZE:
        return
    now = time.monotonic()
    for cache_key in list(cache):
        if now - cache[cache_key]["time"] > NEGATIVE_TTL:
            del cache[cache_key]


def probe_port(prober: dict, cache_key: tuple, api_key: str) -> dict:
    """Outcome for one port: the cached one while fresh, else a new probe."""
    cache = prober["cache"]
    cached = cache.get(cache_key)
    if cached is not None:
        lifetime = NEGATIVE_TTL
        if cached["server"] is not None:
            lifetime = POSITIVE_TTL
        elif cached["loading"]:
            lifetime = LOADING_TTL
        if time.monotonic() - cached["time"] < lifetime:
            return cached
    port = cache_key[0]
    if cache_key[1]:
        outcome = probe_vllm(port, api_key)
    else:
        previous = cached["server"] if cached is not None else None
        outcome = probe_generic(port, api_key, previous)
    prune_cache(cache)
    cache[cache_key] = outcome
    return outcome


def copy_server(server: dict) -> dict:
    """Independent copy of a cached server record for the caller to keep."""
    copied = dict(server)
    copied["models"] = [dict(entry) for entry in server["models"]]
    return copied


def probe_ports(
    prober: dict, ports: list[int], origin: tuple, api_key: str, expect_vllm: bool
) -> dict:
    """Probe the ports in order; the first one that yields a server wins.

    Returns {"server": server or None, "locked_port": int or None}. locked_port is the first port
    that answered 401 when no port yielded a server. origin identifies whose ports these are, so
    a recycled pid or port never reuses an old answer. Never raises.
    """
    result = {"server": None, "locked_port": None}
    digest = get_key_digest(api_key)
    for port in ports:
        cache_key = (port, bool(expect_vllm), origin, digest)
        try:
            outcome = probe_port(prober, cache_key, api_key)
        except Exception as exc:
            logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
            continue
        if outcome["server"] is not None:
            result["server"] = copy_server(outcome["server"])
            result["locked_port"] = None
            return result
        if outcome["unauthorized"] and result["locked_port"] is None:
            result["locked_port"] = port
    return result


def main():
    pass


if __name__ == "__main__":
    main()
