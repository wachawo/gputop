#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collector: assemble one snapshot of one adapter per call from its backend and inspectors."""

import time

# Local imports
from gputop import adapters
from gputop import classify
from gputop import cmdline
from gputop import constants
from gputop import drm
from gputop import llm
from gputop import nvidia
from gputop import procinfo
from gputop import records

# Which backend module owns the backend state of a collector
OWNER_NVIDIA = "nvidia"
OWNER_DRM = "drm"
OWNER_NONE = ""

# Counters an entry needs before a memory breakdown is requested
COUNTERS_DEVICE_ONLY = (constants.COUNTER_DEDICATED,)
COUNTERS_WITH_SPILL = (constants.COUNTER_DEDICATED, constants.COUNTER_SHARED)


def create_collector_state(
    gpu_index: int,
    adapter: dict | None,
    ordered_adapters: list[dict],
    budget_scale: float | None,
    nvidia_shared: dict | None,
) -> dict:
    """Create the collector state before anything is opened.

    gpu_index: int, index into the ordered adapter list.
    adapter: adapter or None, the selected adapter.
    adapters: list[adapter], the ordered adapter list.
    backend: str, BACKEND_NVML or BACKEND_DRM.
    owner: str, OWNER_NVIDIA, OWNER_DRM or OWNER_NONE (no backend state to sample or close).
    backend_state: dict or None, the state of the owning backend module.
    nvidia_shared: dict or None, the NVML reads shared with the other collectors.
    scan_budget: float or None, seconds one sample may spend on deep scans; None is unlimited.
    drm_budget: float or None, seconds one DRM process scan may take; None is unlimited.
    inspector: dict or None, the process inspector state.
    llm_state: dict or None, the LLM builder state.
    closed: bool, the backend state was released.
    """
    scan_budget = None
    drm_budget = None
    if budget_scale is not None:
        scan_budget = constants.DEEP_SCAN_BUDGET * budget_scale
        drm_budget = constants.DRM_SCAN_BUDGET * budget_scale
    return {
        "gpu_index": gpu_index,
        "adapter": adapter,
        "adapters": ordered_adapters,
        "backend": constants.BACKEND_NVML,
        "owner": OWNER_NONE,
        "backend_state": None,
        "nvidia_shared": nvidia_shared,
        "scan_budget": scan_budget,
        "drm_budget": drm_budget,
        "inspector": None,
        "llm_state": None,
        "closed": False,
    }


def open_backend(state: dict) -> None:
    """Open the backend that serves the selected adapter and record it in the state."""
    adapter = state["adapter"]
    if adapter is None or adapter["vendor_id"] == constants.VENDOR_NVIDIA:
        state["backend_state"] = nvidia.open_nvidia(
            adapter, state["gpu_index"], state["nvidia_shared"]
        )
        state["owner"] = OWNER_NVIDIA
        return
    drm_state = drm.open_drm(adapter, state["gpu_index"], state["drm_budget"])
    if drm_state is None:
        return
    state["backend_state"] = drm_state
    state["owner"] = OWNER_DRM
    state["backend"] = constants.BACKEND_DRM


def open_collector(
    gpu_index: int = 0,
    budget_scale: float | None = 1.0,
    ordered_adapters: list[dict] | None = None,
    nvidia_shared: dict | None = None,
) -> dict:
    """Open a collector for the adapter at gpu_index of the ordered adapter list.

    budget_scale None means unlimited scans; otherwise it scales both scan budgets.
    nvidia_shared is one nvidia.create_shared_reads record handed to every collector of
    a program that opens several, so machine-wide NVML reads happen once per tick.
    Raises ValueError for a bad index; any other failure closes what was opened and propagates.
    """
    if ordered_adapters is None:
        ordered_adapters = adapters.list_ordered_adapters()
    adapter = adapters.select_adapter(ordered_adapters, gpu_index)
    state = create_collector_state(
        gpu_index, adapter, ordered_adapters, budget_scale, nvidia_shared
    )
    opened = False
    try:
        open_backend(state)
        state["inspector"] = procinfo.open_inspector()
        state["llm_state"] = llm.open_llm()
        opened = True
    finally:
        if not opened:
            close_collector(state)
    return state


def close_collector(state: dict) -> None:
    """Release the backend state. Idempotent, never raises."""
    if state["closed"]:
        return
    state["closed"] = True
    backend_state = state["backend_state"]
    owner = state["owner"]
    state["backend_state"] = None
    state["owner"] = OWNER_NONE
    if owner == OWNER_NVIDIA:
        nvidia.close_nvidia(backend_state)
    elif owner == OWNER_DRM:
        drm.close_drm(backend_state)


def read_backend(state: dict) -> dict:
    """Take one backend sample; an absent backend yields an empty sample."""
    if state["owner"] == OWNER_NVIDIA:
        return nvidia.sample_nvidia(state["backend_state"])
    if state["owner"] == OWNER_DRM:
        return drm.sample_drm(state["backend_state"])
    return records.create_backend_sample()


def compute_vram_total(device: dict | None, adapter: dict | None) -> int:
    """Return the device memory total: telemetry first, then the adapter, else 0 (unknown)."""
    if device is not None and device["mem_total"]:
        return device["mem_total"]
    if adapter is not None and adapter["dedicated_vram"]:
        return adapter["dedicated_vram"]
    return 0


def build_snapshot(state: dict, sample: dict) -> dict:
    """Create the snapshot of this sample and copy the adapter-wide values into it."""
    snapshot = records.create_snapshot(state["adapter"], state["adapters"], state["backend"])
    snapshot["taken"] = time.time()
    snapshot["device"] = sample["device"]
    snapshot["driver"] = sample["driver"]
    snapshot["cuda"] = sample["cuda"]
    snapshot["vram_used"] = sample["vram_used"]
    snapshot["shared_used"] = sample["shared_used"]
    snapshot["process_status"] = sample["process_status"]
    snapshot["counter_status"] = sample["counter_status"]
    snapshot["shared_is_spill"] = sample["shared_is_spill"]
    snapshot["vram_total"] = compute_vram_total(sample["device"], state["adapter"])
    return snapshot


def is_live(gpu: dict) -> bool:
    """Tell whether the counters of a process are worth a row."""
    if not gpu["attributed"]:
        return True
    if gpu["dedicated"] >= constants.LIVE_MIN_DEDICATED:
        return True
    if records.get_gpu_pct(gpu) >= constants.LIVE_MIN_GPU_PCT:
        return True
    return gpu["shared"] >= constants.LIVE_MIN_SHARED


def get_sort_vram(item: tuple[int, dict]) -> float:
    """Return the ordering value of a (pid, proc_gpu) pair: shown VRAM, unknown as 0."""
    return records.get_shown_vram(item[1]) or 0.0


def list_live_procs(procs: dict) -> list[tuple[int, dict]]:
    """Return the live (pid, proc_gpu) pairs, largest shown VRAM first."""
    live = [(pid, gpu) for pid, gpu in procs.items() if is_live(gpu)]
    return sorted(live, key=get_sort_vram, reverse=True)


def create_entry(pid: int, gpu: dict, detail: dict, kind: str) -> dict:
    """Create the working record of one live process during a sample.

    pid: int, local PID. gpu: proc_gpu of this sample.
    detail: the cached detail of the inspector (never mutated here, copied into the row).
    kind: str, a KIND_* value, recomputed after the deep scan.
    view: LLM view or None.
    """
    return {
        "pid": pid,
        "gpu": gpu,
        "detail": detail,
        "kind": kind,
        "view": None,
    }


def build_entries(state: dict, live: list[tuple[int, dict]]) -> list[dict]:
    """Fetch the cheap detail of every live process and classify it a first time."""
    entries = []
    for pid, gpu in live:
        detail = procinfo.get_detail(state["inspector"], pid)
        kind = classify.classify_process(detail, gpu["engines"])
        entries.append(create_entry(pid, gpu, detail, kind))
    return entries


def get_scan_mem(gpu: dict) -> float:
    """Return the GPU bytes of a process that the rescan rule compares between scans."""
    return (records.get_shown_vram(gpu) or 0.0) + gpu["shared"]


def is_scan_candidate(position: int, gpu: dict) -> bool:
    """Tell whether an entry may be deep scanned: a top row or a busy process."""
    if position < constants.DEEP_SCAN_TOP_ROWS:
        return True
    return records.get_gpu_pct(gpu) >= constants.DEEP_SCAN_MIN_GPU_PCT


def scan_entries(state: dict, entries: list[dict]) -> None:
    """Deep scan the candidates within the budget, then classify every entry again."""
    deadline = None
    if state["scan_budget"] is not None:
        deadline = time.perf_counter() + state["scan_budget"]
    inspector = state["inspector"]
    for position, entry in enumerate(entries):
        gpu = entry["gpu"]
        detail = entry["detail"]
        in_time = deadline is None or time.perf_counter() < deadline
        if is_scan_candidate(position, gpu) and in_time:
            mem = get_scan_mem(gpu)
            if procinfo.needs_scan(detail, mem):
                procinfo.scan_process(inspector, detail, mem)
        entry["kind"] = classify.classify_process(detail, gpu["engines"])


def get_required_counters(snapshot: dict) -> tuple:
    """Return the counters a row must have measured before a breakdown is requested."""
    if snapshot["backend"] == constants.BACKEND_NVML or not snapshot["shared_is_spill"]:
        return COUNTERS_DEVICE_ONLY
    return COUNTERS_WITH_SPILL


def is_llm_candidate(entry: dict, required: tuple) -> bool:
    """Tell whether an entry qualifies for an LLM memory breakdown."""
    detail = entry["detail"]
    kind = entry["kind"]
    if cmdline.is_diffusion_cmdline(detail["cmdline"]):
        return False
    if kind not in constants.AI_KINDS:
        return False
    if not classify.has_detail_evidence(detail) and kind != constants.KIND_LLM:
        return False
    available = entry["gpu"]["available"]
    return all(counter in available for counter in required)


def find_ollama_runner_pid(entries: list[dict]) -> int | None:
    """Return the pid of the Ollama runner when exactly one entry is one, else None."""
    runner_pids = []
    for entry in entries:
        if classify.is_ollama_runner(entry["detail"], entry["kind"]):
            runner_pids.append(entry["pid"])
    if len(runner_pids) == 1:
        return runner_pids[0]
    return None


def build_entry_view(state: dict, entry: dict, spill_mode: bool, log_path: str | None) -> dict:
    """Request the LLM view of one entry with its held and spilled bytes."""
    gpu = entry["gpu"]
    detail = entry["detail"]
    shown = records.get_shown_vram(gpu)
    held = shown or 0.0
    spill = 0.0
    if spill_mode:
        spill = gpu["shared"]
    else:
        held += gpu["shared"]
    attributed = shown is not None
    parents = procinfo.get_parent_targets(state["inspector"], detail)
    return llm.build_llm_view(
        state["llm_state"], detail, parents, held, spill, attributed, log_path
    )


def build_views(state: dict, snapshot: dict, entries: list[dict]) -> None:
    """Fill the LLM view of every entry that passes the gate."""
    required = get_required_counters(snapshot)
    runner_pid = find_ollama_runner_pid(entries)
    spill_mode = snapshot["shared_is_spill"]
    for entry in entries:
        if not is_llm_candidate(entry, required):
            continue
        log_path = None
        if runner_pid is not None and entry["pid"] == runner_pid:
            log_path = llm.get_ollama_log_path()
        entry["view"] = build_entry_view(state, entry, spill_mode, log_path)


def build_local_row(entry: dict) -> dict:
    """Turn a finished entry into a row that owns a copy of the detail."""
    detail = entry["detail"]
    kind = entry["kind"]
    view = entry["view"]
    row = records.create_row(
        entry["pid"], detail["name"], kind, entry["gpu"], records.copy_detail(detail)
    )
    row["llm"] = view
    row["inside"] = classify.describe_inside(detail, kind, view)
    return row


def build_outside_row(pid: int, gpu: dict) -> dict:
    """Create the row of a process that lives in another PID namespace."""
    detail = records.create_detail(pid, constants.OUTSIDE_NAME)
    row = records.create_row(
        pid, constants.OUTSIDE_NAME, constants.KIND_EXTERNAL, gpu, detail, outside=True
    )
    row["inside"] = constants.KIND_INSIDE_TEXT[constants.KIND_EXTERNAL]
    return row


def sample_collector(state: dict) -> dict:
    """Take one snapshot of the selected adapter. May raise an unexpected exception."""
    sample = read_backend(state)
    snapshot = build_snapshot(state, sample)
    live = list_live_procs(sample["procs"])
    entries = build_entries(state, live)
    scan_entries(state, entries)
    build_views(state, snapshot, entries)
    rows = [build_local_row(entry) for entry in entries]
    for pid, gpu in sample["outside"].items():
        rows.append(build_outside_row(pid, gpu))
    snapshot["procs"] = rows
    procinfo.prune_inspector(state["inspector"], set(sample["procs"]))
    return snapshot


def main():
    pass


if __name__ == "__main__":
    main()
