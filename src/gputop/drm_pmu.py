#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whole-GPU busy counters of the i915 PMU through the perf_event_open syscall."""

import ctypes
import errno
import os
import re
import struct
import time

# Local imports
from gputop import constants
from gputop import drm_cards

# PMU discovery
EVENT_SOURCE_DIR = "/sys/bus/event_source/devices"
PMU_NAME = "i915"
BUSY_SUFFIX = "-busy"
VIDEO_PREFIXES = ("vcs", "vecs")
CPU_PATTERN = re.compile(r"\d+")

# perf_event_open details
SYSCALL_NUMBERS = {"x86_64": 298, "aarch64": 241}
PERF_ATTR_SIZE = 64
PERF_ATTR_HEAD = "=IIQ"
COUNTER_FORMAT = "=Q"
COUNTER_SIZE = 8

# Why the PMU gives no reading
REASON_DENIED = "denied"
REASON_MISSING = "missing"
REASON_SHARED = "shared"


def create_pmu(reason: str = "") -> dict:
    """Create the PMU state of one card.

    Keys:
        reason: str, "" while usable, else REASON_DENIED, REASON_MISSING or REASON_SHARED;
            a reason is final for the life of the state.
        spec: dict or None, decoded PMU: {"type": int, "cpu": int,
            "events": dict[str, int] engine instance name to event config}.
        fds: dict[str, int], engine instance name to open perf descriptor.
        baseline: dict[str, tuple[int, int]], engine instance name to
            (busy nanoseconds, monotonic nanoseconds) of the previous read.
    """
    return {"reason": reason, "spec": None, "fds": {}, "baseline": {}}


def create_pmu_reading() -> dict:
    """Create one PMU reading.

    Keys:
        util: float or None, busiest engine in percent, None = unknown.
        video: float or None, busiest video engine in percent, None = unknown.
    """
    return {"util": None, "video": None}


def find_pmu_path(card: dict) -> str | None:
    """Return the event source directory that serves the card; None when there is none."""
    if card["driver"] != constants.DRIVER_I915:
        return None
    suffix = card["pdev"].replace(":", "_")
    named_path = os.path.join(EVENT_SOURCE_DIR, f"{PMU_NAME}_{suffix}")
    if os.path.isdir(named_path):
        return named_path
    plain_path = os.path.join(EVENT_SOURCE_DIR, PMU_NAME)
    if card["integrated"] and os.path.isdir(plain_path):
        return plain_path
    return None


def decode_events(events_dir: str) -> dict | None:
    """Read the config of every *-busy event; None when one is malformed or none exists."""
    try:
        names = sorted(os.listdir(events_dir))
    except OSError:
        return None
    events = {}
    for name in names:
        if not name.endswith(BUSY_SUFFIX):
            continue
        content = drm_cards.read_text(os.path.join(events_dir, name))
        if not content or "=" not in content:
            return None
        assignment = content.split("=", 1)[1]
        config_text = assignment.split(",", 1)[0]
        value = config_text.strip()
        try:
            events[name[:-len(BUSY_SUFFIX)]] = int(value, 16)
        except ValueError:
            return None
    return events if events else None


def decode_pmu(path: str) -> dict | None:
    """Decode type, CPU and busy events of a PMU directory; None on any failure."""
    type_text = drm_cards.read_text(os.path.join(path, "type"))
    cpumask = drm_cards.read_text(os.path.join(path, "cpumask"))
    if not type_text or not type_text.isdigit() or not cpumask:
        return None
    cpu_match = CPU_PATTERN.match(cpumask)
    if cpu_match is None:
        return None
    events = decode_events(os.path.join(path, "events"))
    if events is None:
        return None
    return {"type": int(type_text), "cpu": int(cpu_match.group(0)), "events": events}


def open_perf_event(pmu_type: int, config: int, cpu: int) -> int:
    """Open one system-wide counting event; raises OSError when the syscall fails."""
    number = SYSCALL_NUMBERS.get(os.uname().machine)
    if number is None:
        raise OSError(errno.ENOSYS, "perf_event_open is not known for this machine type")
    head = struct.pack(PERF_ATTR_HEAD, pmu_type, PERF_ATTR_SIZE, config)
    attr = ctypes.create_string_buffer(head.ljust(PERF_ATTR_SIZE, b"\0"), PERF_ATTR_SIZE)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(
        ctypes.c_long(number),
        attr,
        ctypes.c_long(-1),
        ctypes.c_long(cpu),
        ctypes.c_long(-1),
        ctypes.c_ulong(0),
    )
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def close_counters(pmu: dict) -> None:
    """Close every perf descriptor and forget the baselines."""
    for fd in pmu["fds"].values():
        try:
            os.close(fd)
        except OSError:
            pass
    pmu["fds"] = {}
    pmu["baseline"] = {}


def open_counters(pmu: dict) -> None:
    """Open one counter per engine instance; a failure closes all and sets the reason."""
    spec = pmu["spec"]
    for name, config in spec["events"].items():
        try:
            pmu["fds"][name] = open_perf_event(spec["type"], config, spec["cpu"])
        except OSError as exc:
            close_counters(pmu)
            if exc.errno in (errno.EACCES, errno.EPERM):
                pmu["reason"] = REASON_DENIED
            else:
                pmu["reason"] = REASON_MISSING
            return


def open_pmu(card: dict, cards: list[dict]) -> dict:
    """Open the PMU of the card; the state carries a reason when it cannot be used."""
    path = find_pmu_path(card)
    if path is None:
        return create_pmu(REASON_MISSING)
    for other in cards:
        if other["pdev"] != card["pdev"] and other["intel"] and find_pmu_path(other) == path:
            return create_pmu(REASON_SHARED)
    pmu = create_pmu()
    pmu["spec"] = decode_pmu(path)
    if pmu["spec"] is None:
        pmu["reason"] = REASON_MISSING
        return pmu
    open_counters(pmu)
    return pmu


def read_counter(pmu: dict, name: str) -> tuple[bool, float | None]:
    """Read one counter: (read succeeded, busy percent or None without a usable delta)."""
    try:
        raw = os.read(pmu["fds"][name], COUNTER_SIZE)
    except OSError:
        return False, None
    stamp = time.monotonic_ns()
    if len(raw) != COUNTER_SIZE:
        return False, None
    busy = struct.unpack(COUNTER_FORMAT, raw)[0]
    previous = pmu["baseline"].get(name)
    pmu["baseline"][name] = (busy, stamp)
    if previous is None:
        return True, None
    busy_delta = busy - previous[0]
    elapsed = stamp - previous[1]
    if busy_delta < 0 or elapsed <= 0:
        return True, None
    return True, min(100.0, 100.0 * busy_delta / elapsed)


def summarize_busy(shares: dict) -> dict:
    """Reduce per-engine busy shares (None = no value) to GPU and video utilisation."""
    reading = create_pmu_reading()
    values = list(shares.values())
    if values and all(value is not None for value in values):
        reading["util"] = max(values)
    video_values = [value for name, value in shares.items() if name.startswith(VIDEO_PREFIXES)]
    if video_values and all(value is not None for value in video_values):
        reading["video"] = max(video_values)
    return reading


def read_pmu(pmu: dict) -> dict:
    """Take one PMU reading; unknown values are None. Never raises."""
    if pmu["reason"] or pmu["spec"] is None:
        return create_pmu_reading()
    if not pmu["fds"]:
        open_counters(pmu)
        if pmu["reason"]:
            return create_pmu_reading()
    shares = {}
    for name in list(pmu["fds"]):
        succeeded, share = read_counter(pmu, name)
        if not succeeded:
            close_counters(pmu)
            return create_pmu_reading()
        shares[name] = share
    return summarize_busy(shares)


def close_pmu(pmu: dict) -> None:
    """Close the perf descriptors. Idempotent, never raises."""
    close_counters(pmu)


def main():
    pass


if __name__ == "__main__":
    main()
