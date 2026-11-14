#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Factories of the shared gputop records and the derived-value helpers on them."""

# Local imports
from gputop import constants
from gputop import formatting

HEX_DIGITS = "0123456789abcdef"
RUNTIME_LABEL_LIMIT = 3
RUNTIME_LABEL_SEPARATOR = "+"
AMD_SOURCE_PREFIX = "AMD "
DEVICE_LABEL_AMD = "AMD gfx"
DEVICE_LABEL_INTEL = "Intel GT"
DEVICE_LABEL_DEFAULT = "GPU"
CLOCK_LABEL_AMD = "gfx"
CLOCK_LABEL_INTEL = "GT"
CLOCK_LABEL_DEFAULT = "SM"
OUTSIDE_KEY_PREFIX = "o"
OUTSIDE_PID_PREFIX = "H:"


def create_adapter(
    key: int,
    name: str,
    vendor_id: int,
    dedicated_vram: int = 0,
    shared_system: int = 0,
) -> dict:
    """Create an adapter record: one physical GPU, immutable after enumeration.

    key: int, identity derived from the PCI address (compute_adapter_key).
    name: str, marketing name.
    vendor_id: int, PCI vendor id.
    dedicated_vram: int, total device memory in bytes; 0 means unknown.
    shared_system: int, total GPU-accessible system RAM in bytes (AMD GTT); 0 otherwise.
    """
    return {
        "key": key,
        "name": name,
        "vendor_id": vendor_id,
        "dedicated_vram": dedicated_vram,
        "shared_system": shared_system,
    }


def create_device(index: int, source: str) -> dict:
    """Create a device telemetry record; every optional reading starts as None (no reading).

    index: int, NVML device index or the --gpu index of a DRM card.
    name: str, device name, "?" by default.
    pci_bus: str, PCI bus id as the provider spells it.
    driver_model: str, kernel driver name on DRM (amdgpu, i915, xe), "" on NVML.
    mem_total, mem_used, mem_reserved: int or None, bytes.
    util_gpu, util_mem, enc, dec, video: float or None, percent.
    temp: float or None, degrees Celsius; temp_kind: str, label of a fallback sensor.
    power_w, power_limit_w: float or None, watts; power_kind: str, what is measured.
    fan: int or None, percent. pstate: int or None.
    sm_clock, sm_clock_max, sm_clock_requested, gfx_clock, mem_clock, video_clock:
        int or None, MHz.
    pcie_rx_kbs, pcie_tx_kbs: int or None, KB/s. pcie_gen, pcie_width: int or None.
    throttle: list[str] or None; None is unreadable, [] is readable with none active.
    compute_cap: str, "major.minor" or "".
    source: str, one of the SOURCE_* constants.
    """
    return {
        "index": index,
        "name": constants.UNKNOWN_NAME,
        "pci_bus": "",
        "driver_model": "",
        "mem_total": None,
        "mem_used": None,
        "mem_reserved": None,
        "util_gpu": None,
        "util_mem": None,
        "enc": None,
        "dec": None,
        "video": None,
        "temp": None,
        "temp_kind": "",
        "power_w": None,
        "power_limit_w": None,
        "power_kind": "",
        "fan": None,
        "pstate": None,
        "sm_clock": None,
        "sm_clock_max": None,
        "sm_clock_requested": None,
        "gfx_clock": None,
        "mem_clock": None,
        "video_clock": None,
        "pcie_rx_kbs": None,
        "pcie_tx_kbs": None,
        "pcie_gen": None,
        "pcie_width": None,
        "throttle": None,
        "compute_cap": "",
        "source": source,
    }


def create_proc_gpu(available: set[str]) -> dict:
    """Create the per-process counters of one process on one adapter.

    dedicated: float, device memory charged to the process, bytes; meaningful only when
        "dedicated" is in available and attributed is true.
    shared: float, GPU memory of the process in system RAM, bytes; meaningful only when
        "shared" is in available.
    engines: dict[str, float], busy percent per ENGINE_* name, a missing key is 0;
        meaningful only when "engine" is in available.
    available: set[str], subset of the COUNTER_* names (a copy of the argument).
    attributed: bool, False when the per-process memory is not known.
    inferred_by: str, "" or one of the INFERRED_* values.
    resident_vram, resident_gtt: float or None, DRM resident totals in bytes, never summed.
    drm_reason: str, "" or one of the DRM_REASON_* codes.
    kfd: bool, the process holds /dev/kfd (AMD).
    """
    return {
        "dedicated": 0.0,
        "shared": 0.0,
        "engines": {},
        "available": set(available),
        "attributed": True,
        "inferred_by": "",
        "resident_vram": None,
        "resident_gtt": None,
        "drm_reason": "",
        "kfd": False,
    }


def create_backend_sample() -> dict:
    """Create an empty backend sample, the result of one NVML or DRM read of one adapter.

    procs: dict[int, proc_gpu], local PID to counters, unfiltered.
    outside: dict[int, proc_gpu], foreign-namespace PID to counters, in display order.
    vram_used: int or None, device memory in use, adapter-wide, bytes.
    shared_used: int or None, GPU memory in system RAM in use, adapter-wide, bytes.
    device: device or None, telemetry of the selected adapter.
    driver: str, driver version or "kernel <release>".
    cuda: str, "major.minor", NVML only.
    process_status: str, DRM scan status.
    counter_status: str, explanation of missing Intel metrics.
    shared_is_spill: bool, False only for an integrated DRM card.
    """
    return {
        "procs": {},
        "outside": {},
        "vram_used": None,
        "shared_used": None,
        "device": None,
        "driver": "",
        "cuda": "",
        "process_status": "",
        "counter_status": "",
        "shared_is_spill": True,
    }


def create_model_file(
    path: str,
    size: int,
    resident: int | None = 0,
    gguf: bool = False,
    sources: list[str] | None = None,
    store_name: str | None = None,
    kind: str = "",
) -> dict:
    """Create a model file record.

    path: str, absolute file path, or a folder path for a grouped installed model.
    size: int, bytes.
    resident: int or None, bytes of the file in the process RSS; None for a path hint.
    gguf: bool, the file has a verified GGUF magic.
    sources: list[str], ordered EVIDENCE_* values without duplicates (a new list).
    store_name: str or None, friendly store name.
    kind: str, a GGUF_KIND_* value or "".
    """
    return {
        "path": path,
        "size": size,
        "resident": resident,
        "gguf": gguf,
        "sources": list(sources) if sources else [],
        "store_name": store_name,
        "kind": kind,
    }


def create_detail(pid: int, name: str = constants.UNKNOWN_NAME) -> dict:
    """Create a process detail record.

    pid: int. name: str, short process name. exe: str, executable path.
    cmdline: list[str], raw argument vector (redact before display).
    create_time: float, Unix seconds, 0.0 means unknown.
    runtimes: list[str], runtime labels in table order.
    models: list[model file], evidence first, then size descending.
    candidates: list[model file], installed and unproven.
    ports: list[int], listening TCP ports, ascending.
    exclusive_ports: list[int], ports no other PID listens on.
    access_denied: bool.
    maps_time: float, Unix seconds of the last mapped-file scan, 0.0 means never.
    scan_mem: float, GPU bytes of the process at that scan.
    parents: list[int] or None, ancestor PIDs nearest first; None means not looked up yet.
    """
    return {
        "pid": pid,
        "name": name,
        "exe": "",
        "cmdline": [],
        "create_time": 0.0,
        "runtimes": [],
        "models": [],
        "candidates": [],
        "ports": [],
        "exclusive_ports": [],
        "access_denied": False,
        "maps_time": 0.0,
        "scan_mem": 0.0,
        "parents": None,
    }


def copy_model_file(model: dict) -> dict:
    """Return a shallow copy of a model file record with its own sources list."""
    duplicate = dict(model)
    duplicate["sources"] = list(model["sources"])
    return duplicate


def copy_detail(detail: dict) -> dict:
    """Return a copy of a detail whose lists are new and whose model files are copies."""
    duplicate = {}
    for key, value in detail.items():
        if key in ("models", "candidates"):
            duplicate[key] = [copy_model_file(model) for model in value]
        elif isinstance(value, list):
            duplicate[key] = list(value)
        else:
            duplicate[key] = value
    return duplicate


def create_row(
    pid: int,
    name: str,
    kind: str,
    gpu: dict,
    detail: dict,
    outside: bool = False,
) -> dict:
    """Create one line of the process table.

    pid: int, local PID, or the foreign PID when outside.
    name: str, process name, or OUTSIDE_NAME.
    kind: str, a KIND_* value.
    gpu: proc_gpu. detail: detail (a copy owned by the row).
    llm: LLM view or None.
    outside: bool, the PID lives in another PID namespace.
    inside: str, one-line summary filled by the collector.
    gpu_index: int or None, set only by the TUI when several GPUs are shown.
    ram: int or None, RSS bytes, set only by the TUI.
    cpu: float or None, percent of the whole machine, set only by the TUI.
    """
    return {
        "pid": pid,
        "name": name,
        "kind": kind,
        "gpu": gpu,
        "detail": detail,
        "llm": None,
        "outside": outside,
        "inside": "",
        "gpu_index": None,
        "ram": None,
        "cpu": None,
    }


def create_snapshot(adapter: dict | None, ordered_adapters: list[dict], backend: str) -> dict:
    """Create an empty snapshot: one sample of one adapter.

    taken: float, Unix seconds. adapter: adapter or None. adapters: list[adapter], ordered.
    device: device or None. driver, cuda: str.
    vram_total: int, bytes, 0 means unknown.
    vram_used, shared_used: int or None, bytes.
    procs: list[row], local rows by shown VRAM descending, then outside rows.
    backend: str, BACKEND_NVML or BACKEND_DRM.
    process_status, counter_status: str. shared_is_spill: bool.
    """
    return {
        "taken": 0.0,
        "adapter": adapter,
        "adapters": ordered_adapters,
        "device": None,
        "driver": "",
        "cuda": "",
        "vram_total": 0,
        "vram_used": None,
        "shared_used": None,
        "procs": [],
        "backend": backend,
        "process_status": "",
        "counter_status": "",
        "shared_is_spill": True,
    }


def compute_adapter_key(bus_id: str) -> int:
    """Derive the adapter key from a PCI bus id: its hex digits read as one base-16 number."""
    lowered = bus_id.lower()
    digits = "".join(character for character in lowered if character in HEX_DIGITS)
    if not digits:
        return 0
    return int(digits, 16)


def get_vendor_name(vendor_id: int) -> str:
    """Return the vendor name, or the lower-case hexadecimal literal of an unknown id."""
    known = constants.VENDOR_NAMES.get(vendor_id)
    if known is not None:
        return known
    return f"0x{vendor_id:04x}"


def get_shown_vram(gpu: dict) -> float | None:
    """Return the device memory to display for a process, or None when it is not measured."""
    if gpu["attributed"] and constants.COUNTER_DEDICATED in gpu["available"]:
        return gpu["dedicated"]
    return None


def is_spill(row: dict, snapshot: dict) -> bool:
    """True when the system-RAM share of a model process means it spilled out of VRAM."""
    return (
        row["gpu"]["shared"] >= constants.SPILL_WARN_BYTES
        and row["kind"] in constants.SPILL_KINDS
        and snapshot["shared_is_spill"]
    )


def get_gpu_pct(gpu: dict) -> float:
    """Return the load of the busiest engine of a process, 0.0 without engines."""
    engines = gpu["engines"]
    if not engines:
        return 0.0
    return max(engines.values())


def get_engine_columns(gpu: dict) -> dict:
    """Return the four load columns of a process; all None when engines are not measured."""
    if constants.COUNTER_ENGINE not in gpu["available"]:
        return {"gpu": None, "3d": None, "copy": None, "video": None}
    engines = gpu["engines"]
    render = engines.get(constants.ENGINE_3D, 0.0) + engines.get(constants.ENGINE_COMPUTE, 0.0)
    video = (
        engines.get(constants.ENGINE_DECODE, 0.0)
        + engines.get(constants.ENGINE_ENCODE, 0.0)
        + engines.get(constants.ENGINE_VIDEO, 0.0)
    )
    return {
        "gpu": get_gpu_pct(gpu),
        "3d": render,
        "copy": engines.get(constants.ENGINE_COPY, 0.0),
        "video": video,
    }


def format_vram_label(gpu: dict) -> str:
    """Return the VRAM cell text of a process, with the inferred-owner mark when one applies."""
    shown = get_shown_vram(gpu)
    if shown is None:
        return formatting.MISSING_CELL
    mark = constants.INFERRED_MARKS.get(gpu["inferred_by"], "")
    return formatting.format_size(shown) + mark


def get_row_key(row: dict) -> str:
    """Return the unique id of a row across every shown GPU."""
    base = str(row["pid"])
    if row["outside"]:
        base = OUTSIDE_KEY_PREFIX + base
    if row["gpu_index"] is not None:
        return f"{row['gpu_index']}:{base}"
    return base


def get_pid_label(row: dict) -> str:
    """Return the PID cell text; a foreign-namespace PID is marked."""
    if row["outside"]:
        return f"{OUTSIDE_PID_PREFIX}{row['pid']}"
    return str(row["pid"])


def get_runtime_label(detail: dict) -> str:
    """Return the first runtime labels of a process joined into one cell text."""
    shown = detail["runtimes"][:RUNTIME_LABEL_LIMIT]
    return RUNTIME_LABEL_SEPARATOR.join(shown)


def get_device_name(snapshot: dict) -> str:
    """Adapter name, else the telemetry name, else a generic label."""
    adapter = snapshot["adapter"]
    if adapter is not None and adapter["name"]:
        return adapter["name"]
    device = snapshot["device"]
    if device is not None and device["name"] and device["name"] != constants.UNKNOWN_NAME:
        return device["name"]
    return "GPU"


def is_amd_device(device: dict) -> bool:
    """Tell whether a telemetry record comes from an AMD provider."""
    return device["source"].startswith(AMD_SOURCE_PREFIX)


def get_device_label(device: dict) -> str:
    """Return the label of the device load figure for the telemetry provider."""
    if is_amd_device(device):
        return DEVICE_LABEL_AMD
    if device["source"] == constants.SOURCE_INTEL:
        return DEVICE_LABEL_INTEL
    return DEVICE_LABEL_DEFAULT


def get_main_clock(device: dict) -> tuple[str, int | None]:
    """Return the name and the value (MHz) of the main clock of the device."""
    if is_amd_device(device):
        return (CLOCK_LABEL_AMD, device["gfx_clock"])
    if device["source"] == constants.SOURCE_INTEL:
        return (CLOCK_LABEL_INTEL, device["sm_clock"])
    return (CLOCK_LABEL_DEFAULT, device["sm_clock"])


def get_snapshot_attributed(snapshot: dict) -> int | None:
    """Return the device memory of all rows summed, or None when any row is not measured."""
    total = 0.0
    for row in snapshot["procs"]:
        shown_vram = get_shown_vram(row["gpu"])
        if shown_vram is None:
            return None
        total += shown_vram
    return int(total)


def get_memory_note(snapshot: dict) -> str:
    """Return the Intel memory note when it applies to the snapshot, else an empty string."""
    device = snapshot["device"]
    if snapshot["backend"] != constants.BACKEND_DRM or device is None:
        return ""
    if device["source"] != constants.SOURCE_INTEL:
        return ""
    if snapshot["vram_used"] is not None or snapshot["vram_total"]:
        return constants.INTEL_MEMORY_NOTE
    return ""


def sum_kind_bytes(rows: list[dict]) -> dict:
    """Sum the device memory of the attributed rows per kind."""
    sums = {}
    for row in rows:
        gpu = row["gpu"]
        if not gpu["attributed"]:
            continue
        sums[row["kind"]] = sums.get(row["kind"], 0.0) + gpu["dedicated"]
    return sums


def compute_free_bytes(snapshot: dict) -> int | None:
    """Return the free device memory, or None when it must not be shown."""
    vram_used = snapshot["vram_used"]
    vram_total = snapshot["vram_total"]
    if vram_used is None or not vram_total or get_memory_note(snapshot):
        return None
    return max(int(vram_total - vram_used), 0)


def compute_kind_totals(snapshot: dict) -> dict:
    """Return the memory bar figures of a snapshot.

    kinds: list[tuple[str, float]], raw bytes per kind, largest first.
    attributed_sum: float, sum of the kind totals.
    scale: float, factor that fits the kind totals into vram_used (1.0 when not needed).
    unattributed: float, bytes in use that no row accounts for; 0.0 when vram_used is unknown.
    free: int or None, free device memory; None when it must not be shown.
    """
    sums = sum_kind_bytes(snapshot["procs"])
    kinds = sorted(sums.items(), key=lambda item: item[1], reverse=True)
    attributed_sum = float(sum(sums.values()))
    vram_used = snapshot["vram_used"]
    scale = 1.0
    if attributed_sum > 0 and vram_used is not None:
        scale = min(1.0, vram_used / attributed_sum)
    unattributed = 0.0
    if vram_used is not None:
        unattributed = max(vram_used - attributed_sum * scale, 0.0)
    return {
        "kinds": kinds,
        "attributed_sum": attributed_sum,
        "scale": scale,
        "unattributed": unattributed,
        "free": compute_free_bytes(snapshot),
    }


def main():
    pass


if __name__ == "__main__":
    main()
