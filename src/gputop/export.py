#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn a snapshot and a system sample into JSON-ready dicts and JSON text."""

import json

# Local imports
from gputop import constants
from gputop import records
from gputop import system

# Fixed label of the system section
SYSTEM_SOURCE = "OS-reported (psutil)"

# Key lists of the exported objects, in output order
ADAPTER_KEYS = ("key", "name", "vendor_id", "dedicated_vram", "shared_system")
DEVICE_KEYS = (
    "index", "name", "pci_bus", "driver_model", "mem_total", "mem_used", "mem_reserved",
    "util_gpu", "util_mem", "enc", "dec", "video", "temp", "temp_kind", "power_w",
    "power_kind", "power_limit_w", "fan", "pstate", "sm_clock", "sm_clock_max",
    "sm_clock_requested", "gfx_clock", "mem_clock", "video_clock", "pcie_rx_kbs",
    "pcie_tx_kbs", "pcie_gen", "pcie_width", "throttle", "compute_cap", "source",
)
MODEL_FILE_KEYS = ("path", "size", "resident", "gguf", "sources", "store_name", "kind")
SEGMENT_KEYS = ("label", "bytes", "kind", "exact")
SERVER_KEYS = (
    "kind", "url", "n_ctx", "slots_total", "slots_busy", "gen_tok_s", "prompt_tok_s",
    "model_path",
)
SERVER_MODEL_KEYS = ("name", "type", "quant", "ctx", "size", "size_vram", "path")
SYSTEM_PROC_KEYS = ("pid", "name", "rss")

# Exported key and the model entry key it is read from
LLM_MODEL_KEYS = (
    ("path", "path"), ("name", "name"), ("label", "label"), ("arch", "arch"),
    ("kind", "kind"), ("kind_label", "kind_label"), ("decision_type", "decision_type"),
    ("params", "n_params"), ("quant", "quant"), ("layers", "n_layers"),
    ("ctx_train", "ctx_train"), ("gpu_weight_bytes", "gpu_weight_bytes"),
    ("sources", "sources"), ("store_name", "store_name"),
)


def copy_value(value):
    """Return a list copy for lists so the document shares nothing with the snapshot."""
    if isinstance(value, list):
        return list(value)
    return value


def pick_keys(source: dict, keys: tuple) -> dict:
    """Copy the named keys of a record, in the given order; a missing key reads as None."""
    return {key: copy_value(source.get(key)) for key in keys}


def round_or_none(value, digits: int):
    """Round a number, keeping None as None."""
    if value is None:
        return None
    return round(value, digits)


def convert_int_or_none(value: float | None) -> int | None:
    """Convert a byte count to an integer, keeping None as None."""
    if value is None:
        return None
    return int(value)


def build_system_proc(proc: dict) -> dict:
    """One process object of the system section."""
    entry = pick_keys(proc, SYSTEM_PROC_KEYS)
    entry["cpu_pct"] = round_or_none(proc.get("cpu_pct"), 1)
    return entry


def build_system_dict(sample: dict) -> dict:
    """The `system` object of the JSON document for one system sample."""
    top_ram = system.get_top_procs(sample, "rss", constants.SYSTEM_TOP_ROWS)
    top_cpu = system.get_top_procs(sample, "cpu_pct", constants.SYSTEM_TOP_ROWS)
    return {
        "source": SYSTEM_SOURCE,
        "cpu": {
            "pct": round(sample["cpu_pct"], 1),
            "per_core": list(sample["per_core"]),
            "logical": sample["logical"],
            "physical": sample["physical"],
        },
        "ram": {
            "total": sample["ram_total"],
            "used": sample["ram_used"],
            "available": sample["ram_available"],
            "swap_total": sample["swap_total"],
            "swap_used": sample["swap_used"],
        },
        "top_ram": [build_system_proc(proc) for proc in top_ram],
        "top_cpu": [build_system_proc(proc) for proc in top_cpu],
        "denied": sample["denied"],
        "cost_ms": round(sample["cost_ms"], 1),
    }


def build_server_dict(server: dict | None) -> dict | None:
    """The `server` object of an exported LLM view."""
    if server is None:
        return None
    entry = pick_keys(server, SERVER_KEYS)
    reported_models = server.get("models") or []
    entry["models"] = [pick_keys(model, SERVER_MODEL_KEYS) for model in reported_models]
    return entry


def build_llm_model_dict(model: dict) -> dict:
    """One parsed model of an exported LLM view."""
    return {target: copy_value(model.get(source)) for target, source in LLM_MODEL_KEYS}


def build_llm_dict(view: dict) -> dict:
    """The `llm` object of one process."""
    return {
        "models": [build_llm_model_dict(model) for model in view["models"]],
        "segments": [pick_keys(segment, SEGMENT_KEYS) for segment in view["segments"]],
        "kv_bytes_per_token": view["kv_per_token"],
        "n_ctx": view["n_ctx"],
        "flags": dict(view["flags"]),
        "server": build_server_dict(view["server"]),
        "server_source": view["server_source"],
        "log_source": view["log_source"],
        "notes": list(view["notes"]),
    }


def build_engine_dict(gpu: dict) -> dict | None:
    """Engine name to busy share, zero-valued engines left out; None when not measured."""
    if constants.COUNTER_ENGINE not in gpu["available"]:
        return None
    rounded = {name: round(value, 1) for name, value in gpu["engines"].items()}
    return {name: value for name, value in rounded.items() if value}


def build_gpu_fields(gpu: dict, local_drm: bool) -> dict:
    """The memory and load keys of one process entry."""
    available = gpu["available"]
    has_engine = constants.COUNTER_ENGINE in available
    has_shared = local_drm and constants.COUNTER_SHARED in available
    return {
        "vram_attributed": bool(gpu["attributed"]),
        "vram_inferred_by": gpu["inferred_by"] or None,
        "vram_bytes": convert_int_or_none(records.get_shown_vram(gpu)),
        "vram_resident_bytes": convert_int_or_none(gpu["resident_vram"]) if local_drm else None,
        "gtt_resident_bytes": convert_int_or_none(gpu["resident_gtt"]) if local_drm else None,
        "vram_unattributed_reason": (gpu["drm_reason"] or None) if local_drm else None,
        "shared_bytes": int(gpu["shared"]) if has_shared else None,
        "gpu_pct": round(records.get_gpu_pct(gpu), 1) if has_engine else None,
        "engines": build_engine_dict(gpu),
    }


def build_process_dict(row: dict, is_drm: bool) -> dict:
    """One entry of `processes`."""
    detail = row["detail"]
    created = detail["create_time"]
    entry = {
        "pid": row["pid"],
        "host": constants.HOST_OUTSIDE if row["outside"] else None,
        "name": row["name"],
        "kind": row["kind"],
        "runtimes": list(detail["runtimes"]),
        "started_at": round(created, 3) if created else None,
    }
    entry.update(build_gpu_fields(row["gpu"], is_drm and not row["outside"]))
    entry["ports"] = list(detail["ports"])
    entry["inside"] = row["inside"]
    entry["model_files"] = [pick_keys(model, MODEL_FILE_KEYS) for model in detail["models"]]
    entry["installed_models"] = [
        pick_keys(model, MODEL_FILE_KEYS) for model in detail["candidates"]
    ]
    if row["llm"] is not None:
        entry["llm"] = build_llm_dict(row["llm"])
    return entry


def build_gpu_document(snapshot: dict) -> dict:
    """The GPU part of the JSON document (everything except `system`)."""
    adapter = snapshot["adapter"]
    device = snapshot["device"]
    is_drm = snapshot["backend"] == constants.BACKEND_DRM
    return {
        "taken": snapshot["taken"],
        "gpu": adapter["name"] if adapter is not None else None,
        "driver": snapshot["driver"],
        "cuda": snapshot["cuda"],
        "adapter": pick_keys(adapter, ADAPTER_KEYS) if adapter is not None else None,
        "vram_total": snapshot["vram_total"] or None,
        "vram_used": snapshot["vram_used"],
        "shared_used": snapshot["shared_used"],
        "attributed": records.get_snapshot_attributed(snapshot),
        "backend": snapshot["backend"],
        "shared_is_spill": bool(snapshot["shared_is_spill"]),
        "process_status": snapshot["process_status"] or None,
        "memory_note": records.get_memory_note(snapshot) or None,
        "counter_status": snapshot["counter_status"] or None,
        "device": pick_keys(device, DEVICE_KEYS) if device is not None else None,
        "processes": [build_process_dict(row, is_drm) for row in snapshot["procs"]],
    }


def build_snapshot_dict(
    snapshot: dict,
    system_sample: dict | None = None,
    system_error: str = "",
    include_system: bool = True,
) -> dict:
    """The complete JSON document for one snapshot.

    With `include_system` false the document has no `system` key. Otherwise it carries
    the system object of the sample, or `{"error": system_error}` when there is no sample.
    """
    document = build_gpu_document(snapshot)
    if not include_system:
        return document
    if system_sample is None:
        document["system"] = {"error": system_error}
    else:
        document["system"] = build_system_dict(system_sample)
    return document


def dump_json(document: dict, compact: bool = False) -> str:
    """Serialise a document: indented by two spaces, or on one line when `compact`."""
    if compact:
        return json.dumps(document, default=str, separators=(",", ":"))
    return json.dumps(document, default=str, indent=2)


def main():
    pass


if __name__ == "__main__":
    main()
