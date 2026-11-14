#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the LLM view of one process: models, memory segments, notes and table summary."""

import logging
import os
import traceback

# Local imports
from gputop import classify
from gputop import cmdline
from gputop import constants
from gputop import formatting
from gputop import gguf
from gputop import llm_flags
from gputop import llm_log
from gputop import llm_probe
from gputop import modelfiles
from gputop import records

# KV cache element types: bytes per element
DEFAULT_KV_TYPE = "f16"
DEFAULT_KV_ELEMENT_BYTES = 2.0
KV_ELEMENT_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
    "q4_1": 20 / 32,
    "q5_0": 22 / 32,
    "q5_1": 24 / 32,
}

# Thresholds
WEIGHTS_OVERSHOOT = 1.03
CPU_WEIGHTS_FACTOR = 1.5
NEARLY_EMPTY_BYTES = 64 * constants.MIB
LOG_STALE_SECONDS = 5.0
LMSTUDIO_NOTE_MODELS = 4

# Strings taken from a model header or from a server answer are shown only in this form
TEXT_LIMIT = 128
TEXT_PLACEHOLDER = "?"
FIRST_PRINTABLE = 0x20
CONTROL_RANGE = (0x7F, 0x9F)
SERVER_TEXT_KEYS = ("name", "type", "quant")
PROC_DIR = "/proc"

# Segment labels
LOG_SEGMENT_LABELS = (
    (constants.SEGMENT_WEIGHTS, "weights"),
    (constants.SEGMENT_KV, "KV cache"),
    (constants.SEGMENT_COMPUTE, "compute buffers"),
)
LABEL_LOG_OVERHEAD = "runtime and other overhead"
LABEL_RUNTIME = "compute buffers + runtime"
LABEL_REMAINDER = "KV cache + compute + runtime"

# Notes
NOTE_UNAVAILABLE = "Breakdown unavailable for this process."
NOTE_VLLM_SIGNATURE = (
    "vLLM process signature only; no unambiguous local API association was established."
)
NOTE_TOTAL_UNKNOWN = (
    "This process's VRAM total is unknown (the driver hides per-process memory); "
    "segments are estimates from the model file and launch flags."
)
NOTE_NO_GGUF = "No GGUF mapped. Model may be loaded with --no-mmap or via another format."
NOTE_NEARLY_EMPTY = (
    "Model is mapped but almost nothing is in VRAM: CPU inference, or another GPU."
)
NOTE_PARENT_LOCKED = (
    "Parent-process API locked (401). Parent probes never send the worker's API key; "
    "parent server details are skipped."
)
REASON_KEY_REJECTED = "rejected the key from the launch arguments"
REASON_KEY_MISSING = (
    "wants an API key, and none was found in the launch arguments "
    "(keys from LLAMA_API_KEY or --api-key-file are not read)"
)
VLLM_VIA = {
    constants.SERVER_SOURCE_PROCESS: "process's own listener",
    constants.SERVER_SOURCE_PARENT: "parent process",
}
SUMMARY_PREFIXES = {
    constants.SERVER_SOURCE_PROCESS: "reported (API): ",
    constants.SERVER_SOURCE_PARENT: "reported (parent API): ",
}

logger = logging.getLogger(__name__)


def create_view() -> dict:
    """Create an empty LLM view.

    Keys: models (list of model entries), other_files (list of model file records), flags (dict
    of parsed launch flags), server (server record or None), host (server record or None),
    server_source ("process", "parent" or None), log_source (str or None), log (dict device to
    {"weights", "kv", "compute"} bytes, or None), segments (list of segments), notes (list of
    str), kv_per_token (int or None, bytes), kv_type (str), n_ctx (int or None), summary (str).
    """
    return {
        "models": [],
        "other_files": [],
        "flags": {},
        "server": None,
        "host": None,
        "server_source": None,
        "log_source": None,
        "log": None,
        "segments": [],
        "notes": [],
        "kv_per_token": None,
        "kv_type": DEFAULT_KV_TYPE,
        "n_ctx": None,
        "summary": "",
    }


def create_segment(label: str, size: int, kind: str, exact: bool) -> dict:
    """Create a segment: label (str), bytes (int), kind (a SEGMENT_* value), exact (bool)."""
    return {"label": label, "bytes": int(size), "kind": kind, "exact": exact}


def create_probe_result() -> dict:
    """Create the result of server probing for one process.

    Keys: server, host (server record or None), server_source (str or None), role (vLLM role or
    ""), own_locked_port (int or None: own port that answered 401 while no own server was found),
    parent_locked (bool: a parent port answered 401 and no parent server was found), key_sent
    (bool: an API key from the launch arguments went to the own port).
    """
    return {
        "server": None,
        "host": None,
        "server_source": None,
        "role": "",
        "own_locked_port": None,
        "parent_locked": False,
        "key_sent": False,
    }


def create_draft(view: dict, total: int | None, spill: float) -> dict:
    """Create the working record of one view build.

    Keys: view (the LLM view being filled), infos (list of GGUF header infos, same order as
    view["models"]), main (header info of the main causal LLM or None), total (int bytes or None
    when unknown), spill (float bytes in shared memory), matched (list of server model entries
    tied to this process), no_kv (bool: only non-causal primaries).
    """
    return {
        "view": view,
        "infos": [],
        "main": None,
        "total": total,
        "spill": spill,
        "matched": [],
        "no_kv": False,
    }


def open_llm() -> dict:
    """Create the LLM state: {"prober": prober state}. Nothing to close."""
    return {"prober": llm_probe.open_prober()}


def get_ollama_log_path() -> str:
    """Path of the Ollama server log."""
    return llm_log.get_ollama_log_path()


def list_parent_ports(parents: list[dict]) -> list[int]:
    """Exclusive ports of the parent targets, nearest parent first, without repeats."""
    ports = []
    for parent in parents:
        for port in parent["exclusive_ports"]:
            if port not in ports:
                ports.append(port)
    return ports


def is_probe_blocked(
    detail: dict, role: str, own_ports: list[int], parent_ports: list[int]
) -> bool:
    """vLLM association guard: True when no port can be tied to this process for sure."""
    if not role:
        return False
    if classify.is_vllm_ambiguous(detail["cmdline"]):
        return True
    if len(own_ports) > 1:
        return True
    shared_ports = set(detail["ports"]) - set(own_ports)
    if shared_ports:
        return True
    return bool(own_ports) and bool(parent_ports)


def clean_text(value: str) -> str:
    """Return a header or server string fit to show: no control characters, capped length."""
    kept = []
    for character in value[:TEXT_LIMIT]:
        code = ord(character)
        if code < FIRST_PRINTABLE or CONTROL_RANGE[0] <= code <= CONTROL_RANGE[1]:
            kept.append(TEXT_PLACEHOLDER)
        else:
            kept.append(character)
    return "".join(kept)


def clean_server(server: dict | None) -> None:
    """Clean the model names, types and quantisation labels a server reported, in place.

    The record is the caller's own copy of the probe result.
    """
    if server is None:
        return
    for entry in server["models"]:
        for key in SERVER_TEXT_KEYS:
            if isinstance(entry[key], str):
                entry[key] = clean_text(entry[key])


def probe_own_ports(state: dict, detail: dict, ports: list[int], api_key: str, role: str) -> dict:
    """Probe the process's own exclusive ports, with its API key."""
    if not ports:
        return {"server": None, "locked_port": None}
    origin = (constants.SERVER_SOURCE_PROCESS, detail["pid"], detail["create_time"])
    return llm_probe.probe_ports(state["prober"], ports, origin, api_key, bool(role))


def probe_parent_ports(state: dict, parents: list[dict], ports: list[int], role: str) -> dict:
    """Probe the exclusive ports of the parents; the child's API key is never sent."""
    if not ports:
        return {"server": None, "locked_port": None}
    identities = tuple((parent["pid"], parent["create_time"]) for parent in parents)
    origin = (constants.SERVER_SOURCE_PARENT, identities)
    return llm_probe.probe_ports(state["prober"], ports, origin, "", bool(role))


def probe_servers(state: dict, detail: dict, parents: list[dict]) -> dict:
    """Find the server of the process (own port first, else a parent's) and its front-end."""
    result = create_probe_result()
    role = classify.get_vllm_role(detail)
    result["role"] = role
    own_ports = list(detail["exclusive_ports"])
    parent_ports = list_parent_ports(parents)
    if is_probe_blocked(detail, role, own_ports, parent_ports):
        return result
    api_key = cmdline.find_api_key(detail["cmdline"])
    own = probe_own_ports(state, detail, own_ports, api_key, role)
    parent = probe_parent_ports(state, parents, parent_ports, role)
    clean_server(own["server"])
    clean_server(parent["server"])
    result["key_sent"] = bool(api_key)
    if own["server"] is not None:
        result["server"] = own["server"]
        result["server_source"] = constants.SERVER_SOURCE_PROCESS
        result["host"] = parent["server"]
    else:
        result["own_locked_port"] = own["locked_port"]
        if parent["server"] is not None:
            result["server"] = parent["server"]
            result["server_source"] = constants.SERVER_SOURCE_PARENT
    if parent["server"] is None and parent["locked_port"] is not None:
        result["parent_locked"] = True
    return result


def add_api_source(model: dict) -> None:
    """Mark a model file copy as reported by a runtime API."""
    if constants.EVIDENCE_API not in model["sources"]:
        model["sources"].insert(0, constants.EVIDENCE_API)


def find_file_by_path(files: list[dict], path: str) -> dict | None:
    """Record that refers to the path a server reported, None when there is none."""
    if not os.path.isabs(path):
        relative = os.path.normpath(path)
        if relative.startswith(".."):
            return None
        suffix = "/" + relative
        for model in files:
            if model["path"].endswith(suffix):
                return model
        return None
    identity = modelfiles.get_file_identity(path)
    for model in files:
        if modelfiles.get_file_identity(model["path"]) == identity:
            return model
    return None


def merge_llama_evidence(files: list[dict], server: dict | None) -> None:
    """Tie the model path of a llama.cpp server to a record, or add a record for it."""
    if server is None or server["kind"] != constants.SERVER_LLAMA:
        return
    path = server["model_path"]
    if not path:
        return
    found = find_file_by_path(files, path)
    if found is not None:
        add_api_source(found)
        return
    if not os.path.isabs(path) or not os.path.isfile(path):
        return
    created = modelfiles.create_model_from_path(path, constants.EVIDENCE_API)
    if created is not None:
        files.insert(0, created)


def is_listed(entries: list[dict], entry: dict) -> bool:
    """Identity test: the very same entry dict is already in the list."""
    for item in entries:
        if item is entry:
            return True
    return False


def merge_ollama_server(files: list[dict], server: dict, matched: list[dict]) -> None:
    """Mark records whose store name equals a model the Ollama server has loaded."""
    for model in files:
        if not modelfiles.has_model_evidence(model) or not model["store_name"]:
            continue
        names = model["store_name"].split(", ")
        for entry in server["models"]:
            if entry["name"] not in names:
                continue
            add_api_source(model)
            if not is_listed(matched, entry):
                matched.append(entry)


def list_servers_of_kind(view: dict, kind: str, host_first: bool) -> list[dict]:
    """Server and host records of the given kind."""
    order = ("host", "server") if host_first else ("server", "host")
    found = []
    for key in order:
        server = view[key]
        if server is not None and server["kind"] == kind:
            found.append(server)
    return found


def merge_lmstudio_evidence(files: list[dict], view: dict, matched: list[dict]) -> None:
    """Pair the single loaded LM Studio model with the single primary record, if both exist."""
    studios = list_servers_of_kind(view, constants.SERVER_LMSTUDIO, True)
    if not studios or len(studios[0]["models"]) != 1:
        return
    primaries = []
    for model in files:
        is_side = model["kind"] in constants.GGUF_SIDE_KINDS
        if modelfiles.has_model_evidence(model) and not is_side:
            primaries.append(model)
    if len(primaries) != 1:
        return
    add_api_source(primaries[0])
    matched.append(studios[0]["models"][0])


def merge_api_evidence(files: list[dict], draft: dict) -> None:
    """Add API evidence from the probed servers to the model file copies."""
    view = draft["view"]
    merge_llama_evidence(files, view["server"])
    for server in list_servers_of_kind(view, constants.SERVER_OLLAMA, False):
        merge_ollama_server(files, server, draft["matched"])
    merge_lmstudio_evidence(files, view, draft["matched"])


def get_decision_type(info: dict) -> str | None:
    """Decision type of a model header in its shown form, None when the header has none."""
    decision_type = gguf.get_gguf_decision_type(info)
    if decision_type is None:
        return None
    return clean_text(decision_type)


def create_model_entry(model: dict, info: dict) -> dict:
    """Create a model entry of the view from a model file record and its header info.

    Keys: path, file_name, name, label, arch, kind, kind_label (str), decision_type (str or
    None), n_params (int), params_label, quant (str), n_layers (int, 0 unknown), ctx_train (int
    or None), gpu_weight_bytes (int, full-offload estimate), sources (list of str), store_name
    (str or None), evidence (str).
    """
    name = clean_text(gguf.get_gguf_name(info))
    base_name = info["metadata"].get("general.basename")
    if not isinstance(base_name, str) or not base_name:
        base_name = name
    kind = gguf.get_gguf_kind(info)
    decision_type = get_decision_type(info)
    kind_label = f"{kind} ({decision_type})" if decision_type else kind
    return {
        "path": model["path"],
        "file_name": os.path.basename(model["path"]),
        "name": name,
        "label": clean_text(model["store_name"] or base_name),
        "arch": clean_text(gguf.get_gguf_arch(info)),
        "kind": kind,
        "kind_label": kind_label,
        "decision_type": decision_type,
        "n_params": info["n_params"],
        "params_label": gguf.format_params(info["n_params"]),
        "quant": gguf.get_gguf_quant(info),
        "n_layers": gguf.get_gguf_layers(info),
        "ctx_train": gguf.get_gguf_ctx_train(info),
        "gpu_weight_bytes": gguf.estimate_gpu_weight_bytes(info, None),
        "sources": list(model["sources"]),
        "store_name": model["store_name"],
        "evidence": modelfiles.describe_evidence(model["sources"]),
    }


def split_model_files(files: list[dict], draft: dict) -> None:
    """Sort records into parsed models and other files."""
    view = draft["view"]
    for model in files:
        info = None
        if modelfiles.has_model_evidence(model) and model["gguf"]:
            info = gguf.read_gguf(model["path"])
        if info is None or info["error"]:
            view["other_files"].append(model)
            continue
        view["models"].append(create_model_entry(model, info))
        draft["infos"].append(info)


def resolve_flag_path(pid: int, value: str) -> str:
    """Return a launch-flag path as the target process sees it; "" when that is unknown.

    A relative value belongs to the working directory of the target, not to this one.
    """
    if os.path.isabs(value):
        return value
    try:
        working_directory = os.readlink(f"{PROC_DIR}/{pid}/cwd")
    except OSError:
        return ""
    if not os.path.isabs(working_directory):
        return ""
    return os.path.join(working_directory, value)


def read_load_log(draft: dict, detail: dict, log_path: str | None) -> None:
    """Parse the load log; the explicit path wins over the --log-file flag."""
    view = draft["view"]
    path = log_path
    flag_path = view["flags"].get("log_file")
    if not path and flag_path:
        path = resolve_flag_path(detail["pid"], str(flag_path))
    if not path:
        return
    parsed = llm_log.parse_load_log(path, detail["create_time"] - LOG_STALE_SECONDS)
    if parsed:
        view["log"] = parsed
        view["log_source"] = path


def is_primary_info(info: dict) -> bool:
    """A model that is neither a vision projector nor a LoRA adapter."""
    return gguf.get_gguf_kind(info) not in constants.GGUF_SIDE_KINDS


def find_main_model(draft: dict) -> None:
    """Pick the main causal LLM and decide whether a KV cache can exist at all."""
    primaries = []
    for info in draft["infos"]:
        is_llm = gguf.get_gguf_kind(info) == constants.GGUF_KIND_LLM
        if draft["main"] is None and is_llm and gguf.get_gguf_layers(info) > 0:
            draft["main"] = info
        if is_primary_info(info):
            primaries.append(info)
    causal = [info for info in primaries if gguf.is_gguf_causal(info)]
    draft["no_kv"] = bool(primaries) and not causal


def compute_kv_cost(draft: dict) -> None:
    """Set kv_type and kv_per_token from the cache-type flags and the main model."""
    view = draft["view"]
    key_flag = str(view["flags"].get("ctk", DEFAULT_KV_TYPE))
    value_flag = str(view["flags"].get("ctv", DEFAULT_KV_TYPE))
    key_type = key_flag.lower()
    value_type = value_flag.lower()
    view["kv_type"] = key_type if key_type == value_type else f"K {key_type}/V {value_type}"
    if draft["main"] is None:
        return
    elements = gguf.estimate_kv_elements_per_token(draft["main"])
    if elements is None:
        return
    key_bytes = KV_ELEMENT_BYTES.get(key_type, DEFAULT_KV_ELEMENT_BYTES)
    value_bytes = KV_ELEMENT_BYTES.get(value_type, DEFAULT_KV_ELEMENT_BYTES)
    view["kv_per_token"] = int(elements * (key_bytes + value_bytes) / 2)


def find_first_ctx(entries: list[dict]) -> int | None:
    """Context length of the first server model entry that reports one."""
    for entry in entries:
        if entry["ctx"]:
            return entry["ctx"]
    return None


def resolve_context(draft: dict) -> None:
    """Context length: launch flag, llama.cpp server, then an API entry of the main model."""
    view = draft["view"]
    server = view["server"]
    n_ctx = view["flags"].get("ctx")
    if n_ctx is not None and n_ctx <= 0:
        n_ctx = None
    if n_ctx is None and server is not None and server["kind"] == constants.SERVER_LLAMA:
        n_ctx = server["n_ctx"]
    if n_ctx is None and draft["main"] is not None:
        if draft["matched"]:
            n_ctx = find_first_ctx(draft["matched"])
        elif server is not None and server["kind"] == constants.SERVER_OLLAMA:
            n_ctx = find_first_ctx(server["models"])
    view["n_ctx"] = n_ctx


def sum_log_category(log: dict, category: str, on_gpu: bool) -> tuple[int, list[str]]:
    """Bytes of one category over the GPU (or the non-GPU) devices, and the devices behind it."""
    total = 0
    names = []
    for device in sorted(log):
        size = log[device].get(category, 0)
        if llm_log.is_gpu_device(device) != on_gpu or not size:
            continue
        total += size
        names.append(device)
    return total, names


def build_log_segments(draft: dict) -> None:
    """Measured segments from the load log, plus the overhead the log does not explain."""
    view = draft["view"]
    measured = 0
    for category, label in LOG_SEGMENT_LABELS:
        size, names = sum_log_category(view["log"], category, True)
        if not size:
            continue
        joined = ",".join(names)
        view["segments"].append(create_segment(f"{label} ({joined})", size, category, True))
        measured += size
    total = draft["total"]
    if total is not None and total > measured:
        overhead = create_segment(
            LABEL_LOG_OVERHEAD, total - measured, constants.SEGMENT_RUNTIME, False
        )
        view["segments"].append(overhead)
    cpu_weights, cpu_names = sum_log_category(view["log"], constants.SEGMENT_WEIGHTS, False)
    main = draft["main"]
    if main is not None and cpu_weights > CPU_WEIGHTS_FACTOR * main["token_embd_bytes"]:
        size_text = formatting.format_size(cpu_weights)
        view["notes"].append(
            f"{size_text} of weights are on the CPU side (partial offload or CPU-mapped)."
        )


def estimate_weights(draft: dict) -> int:
    """Weights on the GPU by the model files; partial offload only for a single model."""
    infos = draft["infos"]
    ngl = draft["view"]["flags"].get("ngl") if len(infos) == 1 else None
    weights = 0
    for info in infos:
        if gguf.get_gguf_kind(info) == constants.GGUF_KIND_LORA:
            weights += info["tensor_bytes"]
        else:
            weights += gguf.estimate_gpu_weight_bytes(info, ngl)
    return weights


def add_weights_segment(draft: dict) -> int:
    """Add the estimated weights segment, clamped to the measured total; returns its bytes."""
    view = draft["view"]
    total = draft["total"]
    weights = estimate_weights(draft)
    if total is not None and total > 0 and weights > total * WEIGHTS_OVERSHOOT:
        need_text = formatting.format_size(weights)
        held_text = formatting.format_size(total)
        view["notes"].append(
            f"Weights need ~{need_text} but the process holds {held_text} of VRAM: "
            "partial offload, paging, or placement on another GPU may explain the gap."
        )
        weights = total
    if weights > 0:
        count = len(draft["infos"])
        label = "weights" if count == 1 else f"weights ({count} files)"
        view["segments"].append(create_segment(label, weights, constants.SEGMENT_WEIGHTS, False))
    return weights


def create_kv_segment(view: dict, kv_bytes: int, suffix: str) -> dict:
    """Estimated KV cache segment for the resolved context length."""
    n_ctx = view["n_ctx"]
    kv_type = view["kv_type"]
    label = f"KV cache ({n_ctx:,} ctx, {kv_type}) est.{suffix}"
    return create_segment(label, kv_bytes, constants.SEGMENT_KV, False)


def add_kv_and_runtime(draft: dict, kv_bytes: int, rest: int) -> None:
    """KV estimate fits into the remainder: KV segment, then what is left as runtime."""
    view = draft["view"]
    window = gguf.get_gguf_sliding_window(draft["main"]) if draft["main"] is not None else None
    suffix = " max" if window else ""
    view["segments"].append(create_kv_segment(view, kv_bytes, suffix))
    if window:
        view["notes"].append(
            f"Sliding-window attention ({window:,} tokens): most layers keep a short KV window, "
            "so the real KV cache is smaller than this estimate."
        )
    left = rest - kv_bytes
    if left > 0:
        view["segments"].append(
            create_segment(LABEL_RUNTIME, left, constants.SEGMENT_RUNTIME, False)
        )


def build_estimate_segments(draft: dict) -> None:
    """Estimated segments from the model files, launch flags and the measured total."""
    view = draft["view"]
    total = draft["total"]
    weights = add_weights_segment(draft)
    kv_bytes = 0
    if view["kv_per_token"] and view["n_ctx"] and view["n_ctx"] > 0:
        kv_bytes = view["kv_per_token"] * view["n_ctx"]
    if total is None:
        if kv_bytes > 0:
            view["segments"].append(create_kv_segment(view, kv_bytes, ""))
        view["notes"].append(NOTE_TOTAL_UNKNOWN)
        return
    rest = total - weights
    if kv_bytes > 0 and kv_bytes <= rest:
        add_kv_and_runtime(draft, kv_bytes, rest)
    elif rest > 0 and draft["no_kv"]:
        view["segments"].append(
            create_segment(LABEL_RUNTIME, rest, constants.SEGMENT_RUNTIME, False)
        )
    elif rest > 0:
        view["segments"].append(
            create_segment(LABEL_REMAINDER, rest, constants.SEGMENT_UNKNOWN, False)
        )


def add_vllm_notes(view: dict, role: str) -> None:
    """Notes 1 and 2: what a vLLM API report does and does not prove."""
    server = view["server"]
    if server is not None and server["kind"] == constants.SERVER_VLLM:
        via = VLLM_VIA.get(view["server_source"], "parent process")
        view["notes"].append(
            f"vLLM API reported via {via}; model IDs do not prove this process's model "
            "residency, weights, KV cache or GPU placement."
        )
    elif role:
        view["notes"].append(NOTE_VLLM_SIGNATURE)


def describe_primary_kinds(draft: dict) -> str:
    """Sorted unique kinds of the primary models, each with its decision type in front."""
    kinds = set()
    for info in draft["infos"]:
        if not is_primary_info(info):
            continue
        kind = gguf.get_gguf_kind(info)
        decision_type = get_decision_type(info)
        kinds.add(f"{decision_type} {kind}" if decision_type else kind)
    return ", ".join(sorted(kinds))


def add_model_notes(draft: dict) -> None:
    """Notes 7 to 11: missing model, encoder, KV cost, shared memory, nearly empty VRAM."""
    view = draft["view"]
    total = draft["total"]
    notes = view["notes"]
    if not view["models"] and view["log"] is None and total is not None and total > 0:
        notes.append(NOTE_NO_GGUF)
    if draft["no_kv"]:
        kinds = describe_primary_kinds(draft)
        notes.append(
            f"Non-causal encoder ({kinds}): no KV cache grows with use; "
            "VRAM is weights + compute."
        )
    per_token = view["kv_per_token"]
    if draft["main"] is not None and per_token:
        small = formatting.format_size(per_token * 4096)
        large = formatting.format_size(per_token * 32768)
        notes.append(
            f"KV cost {per_token / 1024:.0f} KB/token ({view['kv_type']}): "
            f"4k ctx = {small}, 32k ctx = {large}."
        )
    if draft["spill"] > constants.SPILL_WARN_BYTES:
        spill_text = formatting.format_size(draft["spill"])
        notes.append(
            f"! {spill_text} is in SHARED GPU memory (system RAM over PCIe). May include "
            "pinned host buffers; under VRAM pressure, fallback can slow inference."
        )
    if total is not None and total < NEARLY_EMPTY_BYTES and view["models"]:
        notes.append(NOTE_NEARLY_EMPTY)


def add_ollama_notes(draft: dict) -> None:
    """Note 12: what each Ollama server reports about its loaded models."""
    view = draft["view"]
    for server in list_servers_of_kind(view, constants.SERVER_OLLAMA, False):
        entries = [entry for entry in server["models"] if is_listed(draft["matched"], entry)]
        prefix = "Ollama:"
        if not entries:
            entries = server["models"]
            prefix = "Ollama (not matched to this runner):"
        for entry in entries:
            if not entry["size"]:
                continue
            size_text = formatting.format_size(entry["size"])
            percent = round((entry["size_vram"] or 0) / entry["size"] * 100)
            ctx_text = entry["ctx"] if entry["ctx"] else "?"
            view["notes"].append(
                f"{prefix} {entry['name']} {size_text} total, {percent}% on GPU, ctx {ctx_text}."
            )


def add_lmstudio_notes(view: dict) -> None:
    """Note 13: models the managing LM Studio front-end has loaded."""
    host = view["host"]
    if host is None or host["kind"] != constants.SERVER_LMSTUDIO:
        return
    for entry in host["models"][:LMSTUDIO_NOTE_MODELS]:
        model_type = entry["type"] or ""
        quant = entry["quant"] or ""
        ctx_text = entry["ctx"] if entry["ctx"] else "?"
        view["notes"].append(f"LM Studio: {entry['name']} {model_type} {quant} ctx {ctx_text}.")


def add_locked_notes(view: dict, probe: dict) -> None:
    """Notes 14 and 15: an API answered 401 and gave no server details."""
    port = probe["own_locked_port"]
    if port is not None and view["server_source"] != constants.SERVER_SOURCE_PROCESS:
        reason = REASON_KEY_REJECTED if probe["key_sent"] else REASON_KEY_MISSING
        view["notes"].append(f"Local API on port {port} {reason}. Server details are skipped.")
    if probe["parent_locked"]:
        view["notes"].append(NOTE_PARENT_LOCKED)


def build_summary(view: dict) -> str:
    """One-line summary for the process table; empty when there is nothing to say."""
    server = view["server"]
    if server is not None and server["kind"] == constants.SERVER_VLLM and server["models"]:
        names = [entry["name"] for entry in server["models"]]
        prefix = SUMMARY_PREFIXES.get(view["server_source"], "")
        return prefix + ", ".join(names)
    parts = []
    for entry in view["models"]:
        if entry["kind"] in constants.GGUF_SIDE_KINDS:
            parts.append(f"+{entry['kind']}")
        else:
            parts.append(f"{entry['label']} {entry['params_label']} {entry['quant']}")
    return ", ".join(parts)


def assemble_view(
    state: dict, detail: dict, parents: list[dict], total: int | None, spill: float,
    log_path: str | None,
) -> dict:
    """Run every step of the breakdown; may raise, the caller turns that into a fallback view."""
    view = create_view()
    draft = create_draft(view, total, spill)
    view["flags"] = llm_flags.parse_llm_flags(detail["cmdline"])
    probe = probe_servers(state, detail, parents)
    view["server"] = probe["server"]
    view["host"] = probe["host"]
    view["server_source"] = probe["server_source"]
    add_vllm_notes(view, probe["role"])
    files = [records.copy_model_file(model) for model in detail["models"]]
    merge_api_evidence(files, draft)
    split_model_files(files, draft)
    read_load_log(draft, detail, log_path)
    find_main_model(draft)
    compute_kv_cost(draft)
    resolve_context(draft)
    if view["log"] is not None:
        build_log_segments(draft)
    else:
        build_estimate_segments(draft)
    add_model_notes(draft)
    add_ollama_notes(draft)
    add_lmstudio_notes(view)
    add_locked_notes(view, probe)
    view["summary"] = build_summary(view)
    return view


def build_llm_view(
    state: dict, detail: dict, parents: list[dict], held: float, spill: float,
    attributed: bool, log_path: str | None,
) -> dict:
    """LLM view of one process: models, servers, segments of its GPU memory, notes, summary.

    held and spill are bytes as the counters deliver them; the total is int(held), or unknown
    when attributed is false. detail is never modified. Never raises: an internal failure is
    logged and yields a view without models and segments that carries one explanatory note.
    """
    try:
        total = int(held) if attributed else None
        return assemble_view(state, detail, parents, total, spill or 0.0, log_path)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        view = create_view()
        view["notes"].append(NOTE_UNAVAILABLE)
        return view


def main():
    pass


if __name__ == "__main__":
    main()
