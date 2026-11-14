#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PID-namespace detection, local CUDA candidates and the host-to-local PID matcher."""
import logging
import os

# Local imports
from gputop import constants

PROC_ROOT = "/proc"
NAMESPACE_LINK = "/proc/self/ns/pid"
INITIAL_NAMESPACE_INODE = 0xEFFFFFFC
NVIDIA_DEVICE_PREFIX = "/dev/nvidia"
CUDA_LIBRARY_MARK = "/libcuda.so"

# Process-wide cache of the namespace kind, filled on first use under the key "kind".
PID_NAMESPACE = {}

logger = logging.getLogger(__name__)


def parse_namespace_inode(target: str) -> int | None:
    """Return the inode of a 'pid:[<inode>]' link target, or None when it does not parse."""
    opening = target.find("[")
    closing = target.find("]", opening + 1)
    if opening < 0 or closing < 0:
        return None
    digits = target[opening + 1:closing]
    if not digits.isdigit():
        return None
    return int(digits)


def read_pid_namespace() -> str:
    """Read the namespace link once and name the kind of PID namespace gputop runs in."""
    try:
        target = os.readlink(NAMESPACE_LINK)
    except OSError as exc:
        logger.debug(f"PID namespace link unreadable: {type(exc).__name__}: {exc}")
        return constants.PIDNS_UNKNOWN
    inode = parse_namespace_inode(target)
    if inode is None:
        return constants.PIDNS_UNKNOWN
    if inode == INITIAL_NAMESPACE_INODE:
        return constants.PIDNS_INITIAL
    return constants.PIDNS_CHILD


def detect_pid_namespace() -> str:
    """Return PIDNS_INITIAL, PIDNS_CHILD or PIDNS_UNKNOWN; computed once per process."""
    kind = PID_NAMESPACE.get("kind")
    if kind is None:
        kind = read_pid_namespace()
        PID_NAMESPACE["kind"] = kind
    return kind


def has_proc_entry(pid: int) -> bool:
    """Tell whether /proc/<pid> exists in the local namespace."""
    return os.path.exists(f"{PROC_ROOT}/{pid}")


def find_foreign_pids(reported_pids: set[int]) -> set[int]:
    """Return the driver-reported PIDs that do not name a process of the local namespace.

    Initial namespace: none. Child namespace: all of them, also one that happens to equal a
    local PID. Unknown namespace: those without a /proc/<pid> entry.
    """
    kind = detect_pid_namespace()
    if kind == constants.PIDNS_INITIAL:
        return set()
    if kind == constants.PIDNS_CHILD:
        return set(reported_pids)
    foreign = set()
    for pid in reported_pids:
        if not has_proc_entry(pid):
            foreign.add(pid)
    return foreign


def has_nvidia_descriptor(pid: int) -> bool:
    """Tell whether the process holds a descriptor on a /dev/nvidia* device file."""
    directory = f"{PROC_ROOT}/{pid}/fd"
    try:
        names = os.listdir(directory)
    except OSError:
        return False
    for name in names:
        try:
            target = os.readlink(f"{directory}/{name}")
        except OSError:
            continue
        if target.startswith(NVIDIA_DEVICE_PREFIX):
            return True
    return False


def has_cuda_mapping(pid: int) -> bool:
    """Tell whether the process maps libcuda; an unreadable maps file counts as no."""
    try:
        with open(f"{PROC_ROOT}/{pid}/maps", "r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if CUDA_LIBRARY_MARK in line:
                    return True
    except OSError:
        return False
    return False


def list_proc_pids() -> list[int]:
    """Return every numeric entry of /proc as an int, unordered."""
    try:
        names = os.listdir(PROC_ROOT)
    except OSError:
        return []
    pids = []
    for name in names:
        if name.isdigit():
            pids.append(int(name))
    return pids


def list_cuda_candidates(excluded_pids: set[int]) -> list[int]:
    """Return local PIDs with a /dev/nvidia* descriptor and a libcuda mapping, ascending.

    PIDs of excluded_pids and gputop's own PID never appear. Unreadable processes are skipped.
    """
    own_pid = os.getpid()
    candidates = []
    for pid in list_proc_pids():
        if pid == own_pid or pid in excluded_pids:
            continue
        if has_nvidia_descriptor(pid) and has_cuda_mapping(pid):
            candidates.append(pid)
    candidates.sort()
    return candidates


def create_matcher() -> dict:
    """Create the matcher state kept between samples.

    Keys:
      pairs          dict[int, int]  confirmed host pid to local pid
      previous_host  set[int]        host pids of the previous call
      previous_local set[int]        local candidates of the previous call
      first_call     bool            True until match_foreign_pids ran once
    """
    return {
        "pairs": {},
        "previous_host": set(),
        "previous_local": set(),
        "first_call": True,
    }


def keep_live_pairs(pairs: dict, host_set: set[int], local_set: set[int]) -> dict:
    """Return the confirmed pairs whose both ends are still present."""
    kept = {}
    for host_pid, local_pid in pairs.items():
        if host_pid in host_set and local_pid in local_set:
            kept[host_pid] = local_pid
    return kept


def confirm_new_pair(matcher: dict, pairs: dict, host_set: set[int], local_set: set[int]) -> None:
    """Pair the single new host pid with the single new local pid, when exactly one of each."""
    paired_locals = set(pairs.values())
    new_host = []
    for pid in host_set:
        if pid not in matcher["previous_host"] and pid not in pairs:
            new_host.append(pid)
    new_local = []
    for pid in local_set:
        if pid not in matcher["previous_local"] and pid not in paired_locals:
            new_local.append(pid)
    if len(new_host) == 1 and len(new_local) == 1:
        pairs[new_host[0]] = new_local[0]


def match_foreign_pids(matcher: dict, host_pids: list[int],
                       local_pids: list[int]) -> dict[int, tuple[int, str]]:
    """Map host pids to (local pid, confidence) by creation order.

    Confirmed pairs are always INFERRED_CONSISTENT. The rest is zipped newest first:
    INFERRED_CONSISTENT when both remainders have the same length, else INFERRED_ORDER.
    Host pids beyond the zip are absent from the result. Call it every sample, with two
    empty lists when nothing is foreign.
    """
    host_set = set(host_pids)
    local_set = set(local_pids)
    pairs = keep_live_pairs(matcher["pairs"], host_set, local_set)
    if not matcher["first_call"]:
        confirm_new_pair(matcher, pairs, host_set, local_set)
    matcher["pairs"] = pairs
    matcher["previous_host"] = host_set
    matcher["previous_local"] = local_set
    matcher["first_call"] = False
    matches = {}
    for host_pid, local_pid in pairs.items():
        matches[host_pid] = (local_pid, constants.INFERRED_CONSISTENT)
    rest_host = sorted(host_set - set(pairs), reverse=True)
    rest_local = sorted(local_set - set(pairs.values()), reverse=True)
    confidence = constants.INFERRED_ORDER
    if len(rest_host) == len(rest_local):
        confidence = constants.INFERRED_CONSISTENT
    for host_pid, local_pid in zip(rest_host, rest_local):
        matches[host_pid] = (local_pid, confidence)
    return matches


def main():
    pass


if __name__ == "__main__":
    main()
