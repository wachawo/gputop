#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System sampler: machine CPU, RAM and swap, per-process RSS and CPU share (psutil)."""

import time

import psutil

# Local imports
from gputop import constants

TOP_FIELDS = ("rss", "cpu_pct")
DEFAULT_TOP_COUNT = 5
MILLISECONDS = 1000.0


def create_system_proc(
    pid: int,
    name: str,
    rss: int | None,
    cpu_pct: float | None,
    created: float | None,
) -> dict:
    """Create the reading of one OS process.

    pid: int. name: str, "?" when unreadable.
    rss: int or None, resident set size in bytes.
    cpu_pct: float or None, percent of the whole machine; None until a second reading exists.
    created: float or None, process start time in Unix seconds.
    """
    return {
        "pid": pid,
        "name": name,
        "rss": rss,
        "cpu_pct": cpu_pct,
        "created": created,
    }


def open_system_sampler() -> dict:
    """Create the sampler state and prime the machine-wide CPU rate.

    logical: int, logical CPU count (1 when unknown).
    physical: int or None, physical core count.
    rated: set[tuple[int, float]], (pid, start time) pairs seen in the previous sample.
    """
    logical = psutil.cpu_count() or 1
    physical = psutil.cpu_count(logical=False)
    psutil.cpu_percent(percpu=True)
    return {
        "logical": logical,
        "physical": physical,
        "rated": set(),
    }


def read_process(process, logical: int, rated: set, seen: set) -> dict:
    """Read one process in a single pass and note its identity in the seen set."""
    with process.oneshot():
        name = process.name()
        memory = process.memory_info()
        busy = process.cpu_percent(None)
        created = process.create_time()
    identity = (process.pid, created)
    seen.add(identity)
    cpu_pct = busy / logical if identity in rated else None
    return create_system_proc(process.pid, name, memory.rss, cpu_pct, created)


def read_denied_process(process) -> dict:
    """Build the reading of a process whose details are not accessible."""
    try:
        name = process.name()
    except psutil.Error:
        name = constants.UNKNOWN_NAME
    return create_system_proc(process.pid, name, None, None, None)


def read_processes(sampler: dict) -> tuple[list[dict], int]:
    """Read every running process; return the readings and the access-denied count."""
    logical = sampler["logical"]
    rated = sampler["rated"]
    seen = set()
    readings = []
    denied = 0
    for process in psutil.process_iter():
        try:
            if not process.is_running():
                continue
            readings.append(read_process(process, logical, rated, seen))
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            denied += 1
            readings.append(read_denied_process(process))
    sampler["rated"] = seen
    return readings, denied


def compute_mean(values: list[float]) -> float:
    """Return the arithmetic mean of the values, 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_system(sampler: dict) -> dict:
    """Take one system sample.

    cpu_pct: float, mean of per_core. per_core: list[float], percent per logical CPU.
    logical: int. physical: int or None.
    ram_total, ram_used, ram_available, swap_total, swap_used: int, bytes.
    procs: list[system proc]. denied: int, processes that refused access.
    cost_ms: float, wall time of this sample in milliseconds.
    """
    started = time.perf_counter()
    per_core = list(psutil.cpu_percent(percpu=True))
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    readings, denied = read_processes(sampler)
    elapsed = time.perf_counter() - started
    return {
        "cpu_pct": compute_mean(per_core),
        "per_core": per_core,
        "logical": sampler["logical"],
        "physical": sampler["physical"],
        "ram_total": int(memory.total),
        "ram_used": int(memory.total - memory.available),
        "ram_available": int(memory.available),
        "swap_total": int(swap.total),
        "swap_used": int(swap.used),
        "procs": readings,
        "denied": denied,
        "cost_ms": elapsed * MILLISECONDS,
    }


def find_system_proc(sample: dict, pid: int | None, created: float = 0.0) -> dict | None:
    """Return the reading of a pid, or None when absent or when the pid was reused."""
    if pid is None:
        return None
    found = None
    for reading in sample["procs"]:
        if reading["pid"] == pid:
            found = reading
    if found is None:
        return None
    if created and found["created"]:
        if abs(found["created"] - created) > constants.PID_REUSE_TOLERANCE:
            return None
    return found


def get_top_procs(sample: dict, field: str, count: int = DEFAULT_TOP_COUNT) -> list[dict]:
    """Return the readings with the largest known value of the field, largest first."""
    if field not in TOP_FIELDS:
        raise ValueError(f"unknown system field: {field}")
    known = [reading for reading in sample["procs"] if reading[field] is not None]
    ordered = sorted(known, key=lambda reading: reading[field], reverse=True)
    return ordered[:count]


def main():
    pass


if __name__ == "__main__":
    main()
