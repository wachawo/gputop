#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Budgeted /proc walk for DRM clients, engine rates and per-process memory attribution."""

import bisect
import os
import time

# Local imports
from gputop import constants
from gputop import drm_fdinfo
from gputop import records

# procfs locations and descriptor targets
PROC_DIR = "/proc"
KFD_TARGET = "/dev/kfd"
DRI_PREFIX = "/dev/dri/"
RENDER_PREFIX = "/dev/dri/renderD"

# Scan tuning
FD_DEADLINE_STEP = 64
# A descriptor list that still verifies as unchanged is listed again in full after this many
# walks; the exact walk is spread by PID so the relists never fall on one refresh.
RELIST_WALKS = 8
START_TIME_TOKEN = 19
STAT_READ_SIZE = 4096
BASELINE_KEEP_NS = 60 * 1_000_000_000
PROC_STATUS = "/proc is not readable"

# What a walk decides about the cached descriptor list of one live process
VERDICT_CURRENT = "current"
VERDICT_AGED = "aged"
VERDICT_UNVERIFIABLE = "unverifiable"
VERDICT_CHANGED = "changed"


def create_scan_state() -> dict:
    """Create the state of the process scan, kept between refreshes.

    Keys:
        cache: dict[int, dict], PID to cache entry (create_cache_entry) of the last walk.
        cursor: int, PID the next walk starts at, 0 after a complete walk.
        walk: int, number of walks taken so far.
        rotation: int, number of the walk in which the cursor last came round to the
            start of the PID list.
        previous_rotation: int, the same for the rotation before that one.
        baselines: dict[tuple, dict], client key to {"engines": dict[str, int] of
            high-water busy nanoseconds, "time": int monotonic nanoseconds of the read}.
        clients: dict[tuple, dict], client key to client record (create_client) of the
            last scan.
        kfd_pids: set[int], processes holding /dev/kfd in the last scan.
        incomplete_pids: set[int], processes with an unreadable descriptor or fdinfo.
        partial: bool, the last walk left a process whose descriptor list is not known:
            never listed, known to have changed, or not listed for a whole rotation.
        status: str, scan status line, "" when the scan saw everything.
    """
    return {
        "cache": {},
        "cursor": 0,
        "walk": 0,
        "rotation": 0,
        "previous_rotation": 0,
        "baselines": {},
        "clients": {},
        "kfd_pids": set(),
        "incomplete_pids": set(),
        "partial": False,
        "status": "",
    }


def create_cache_entry(start: str, walk_number: int = 0, fd_count: int = 0) -> dict:
    """Create the cached descriptor list of one process.

    Keys:
        start: str, start time of the process (field 22 of its stat file).
        listed: int, number of the walk that last listed the descriptors completely.
        fd_count: int, descriptor count the kernel reported right before that listing,
            0 = not reported (kernels that leave the directory size at zero).
        fds: list[tuple[str, str]], (descriptor name, link target) of /dev/dri descriptors.
        kfd: bool, the process holds /dev/kfd.
        incomplete: bool, at least one descriptor link could not be read.
        denied: bool, the descriptor directory is not readable at all.
    """
    return {
        "start": start,
        "listed": walk_number,
        "fd_count": fd_count,
        "fds": [],
        "kfd": False,
        "incomplete": False,
        "denied": False,
    }


def create_walk(budget: float | None) -> dict:
    """Create the bookkeeping of one walk over the process list.

    Keys:
        deadline: float or None, monotonic seconds at which the budget ends, None = no
            budget.
        exhausted: bool, the deadline has passed.
        free_new: bool, the deadline-free listing of a never-listed process is unused.
        free_known: bool, the deadline-free listing of a cached process is unused.
        deferred: list[int], processes that needed a listing and did not get one.
        unknown: bool, one of the deferred processes has no trustworthy cached list.
        aged: list[tuple[int, str]], (PID, start time) of verified entries that are due
            for their periodic relist; served last, from what is left of the budget.
    """
    deadline = None if budget is None else time.monotonic() + budget
    return {
        "deadline": deadline,
        "exhausted": False,
        "free_new": True,
        "free_known": True,
        "deferred": [],
        "unknown": False,
        "aged": [],
    }


def create_client(info: dict, stamp: int) -> dict:
    """Create the record of one DRM client seen in a scan.

    Keys:
        pdev: str, PCI address of the card.
        pids: set[int], processes holding the client.
        info: dict, parsed fdinfo of the first descriptor that showed the client.
        time: int, monotonic nanoseconds right after that first read.
        render_only: bool, every descriptor of the client points at a render node.
        rates: dict[str, float] or None, busy share in percent per raw engine name,
            None = unknown this refresh.
    """
    return {
        "pdev": info["pdev"],
        "pids": set(),
        "info": info,
        "time": stamp,
        "render_only": True,
        "rates": None,
    }


def list_pids() -> list[int] | None:
    """Return every PID of /proc in ascending order; None when /proc cannot be listed."""
    try:
        names = os.listdir(PROC_DIR)
    except OSError:
        return None
    return sorted(int(name) for name in names if name.isascii() and name.isdigit())


def read_start_time(pid: int) -> str | None:
    """Return the start time field of a process; None when the process is gone."""
    try:
        stat_fd = os.open(f"{PROC_DIR}/{pid}/stat", os.O_RDONLY)
    except OSError:
        return None
    try:
        raw = os.read(stat_fd, STAT_READ_SIZE)
    except OSError:
        return None
    finally:
        os.close(stat_fd)
    tail = raw[raw.rfind(b")") + 1:]
    tokens = tail.split()
    if len(tokens) <= START_TIME_TOKEN:
        return None
    return tokens[START_TIME_TOKEN].decode("ascii", errors="replace")


def open_fd_dir(pid: int, entry: dict) -> int | None:
    """Open the descriptor directory of a process; None (entry marked) when it cannot be."""
    try:
        return os.open(f"{PROC_DIR}/{pid}/fd", os.O_RDONLY | os.O_DIRECTORY)
    except PermissionError:
        entry["denied"] = True
    except OSError:
        pass
    return None


def collect_fd_targets(dir_fd: int, entry: dict, deadline: float | None) -> bool:
    """Record the GPU descriptors of an open descriptor directory; False when deferred."""
    try:
        names = os.listdir(dir_fd)
    except OSError:
        return True
    for position, fd_name in enumerate(names):
        if deadline is not None and position and position % FD_DEADLINE_STEP == 0:
            if time.monotonic() >= deadline:
                return False
        try:
            target = os.readlink(fd_name, dir_fd=dir_fd)
        except PermissionError:
            entry["incomplete"] = True
            continue
        except OSError:
            continue
        if target == KFD_TARGET:
            entry["kfd"] = True
        elif target.startswith(DRI_PREFIX):
            entry["fds"].append((fd_name, target))
    return True


def read_fd_count(pid: int) -> int:
    """Return the descriptor count the kernel reports as the size of the fd directory.

    Readable without access to the descriptors themselves. 0 when the kernel does not
    report it or the process is gone.
    """
    try:
        return os.stat(f"{PROC_DIR}/{pid}/fd").st_size
    except OSError:
        return 0


def is_fd_dir_denied(pid: int) -> bool:
    """Tell whether the descriptor directory of a process is still closed to this user."""
    return not os.access(f"{PROC_DIR}/{pid}/fd", os.R_OK, effective_ids=True)


def judge_entry(scan: dict, pid: int, cached: dict | None, unlimited: bool) -> str:
    """Decide whether the cached entry of a live process can stand in for a new listing.

    The directory size is the descriptor count, so an unchanged count lets the walk skip
    the listing; such an entry is still listed again every RELIST_WALKS walks, because a
    close followed by an open leaves the count alone.
    """
    if cached is None or unlimited:
        return VERDICT_CHANGED
    if cached["denied"]:
        return VERDICT_CURRENT if is_fd_dir_denied(pid) else VERDICT_CHANGED
    fd_count = read_fd_count(pid)
    if fd_count == 0:
        return VERDICT_UNVERIFIABLE
    if fd_count != cached["fd_count"]:
        return VERDICT_CHANGED
    if scan["walk"] - cached["listed"] >= RELIST_WALKS + pid % RELIST_WALKS:
        return VERDICT_AGED
    return VERDICT_CURRENT


def list_process_fds(scan: dict, walk: dict, pid: int, start: str, known: bool) -> dict | None:
    """List the GPU descriptors of a process; None when the budget does not allow it.

    The first never-listed process and the first cached process whose directory opens are
    each listed without a deadline, so one huge descriptor table can neither stay unknown
    nor hold the cursor in place. A denied or vanished process does not use that up.
    """
    free_key = "free_known" if known else "free_new"
    if walk["deadline"] is not None and not walk["exhausted"]:
        walk["exhausted"] = time.monotonic() >= walk["deadline"]
    if walk["exhausted"] and not walk[free_key]:
        return None
    entry = create_cache_entry(start, scan["walk"], read_fd_count(pid))
    dir_fd = open_fd_dir(pid, entry)
    if dir_fd is None:
        return entry
    deadline = None if walk[free_key] else walk["deadline"]
    try:
        finished = collect_fd_targets(dir_fd, entry, deadline)
    finally:
        os.close(dir_fd)
    walk[free_key] = False
    if finished:
        return entry
    walk["exhausted"] = True
    return None


def order_pids(pids: list[int], cache: dict, cursor: int) -> list[int]:
    """Rotate the PID list to the cursor and move never-scanned PIDs to the front."""
    split = bisect.bisect_left(pids, cursor)
    rotated = pids[split:] + pids[:split]
    new_pids = [pid for pid in rotated if pid not in cache]
    known_pids = [pid for pid in rotated if pid in cache]
    return new_pids + known_pids


def is_deferral_unknown(scan: dict, cached: dict | None, verdict: str) -> bool:
    """Tell whether a process left without a listing has no trustworthy cached list.

    An entry that cannot be verified by its descriptor count is trusted for one rotation of
    the cursor, which is how long the walk needs to come back to it.
    """
    if cached is None or verdict == VERDICT_CHANGED:
        return True
    return cached["listed"] < scan["previous_rotation"]


def visit_process(scan: dict, walk: dict, pid: int, new_cache: dict) -> None:
    """Carry one live process into the new cache: verified, listed again, or deferred."""
    start = read_start_time(pid)
    if start is None:
        return
    cached = scan["cache"].get(pid)
    if cached is not None and cached["start"] != start:
        cached = None
    verdict = judge_entry(scan, pid, cached, walk["deadline"] is None)
    if verdict == VERDICT_AGED:
        walk["aged"].append((pid, start))
    if verdict in (VERDICT_CURRENT, VERDICT_AGED):
        new_cache[pid] = cached
        return
    entry = list_process_fds(scan, walk, pid, start, cached is not None)
    if entry is None:
        walk["deferred"].append(pid)
        if is_deferral_unknown(scan, cached, verdict):
            walk["unknown"] = True
        entry = cached
    if entry is not None:
        new_cache[pid] = entry


def relist_aged(scan: dict, walk: dict, new_cache: dict) -> None:
    """Spend what is left of the budget on the periodic relists of verified entries.

    An entry that has to wait keeps its verified list and is not counted as deferred.
    """
    for pid, start in walk["aged"]:
        entry = list_process_fds(scan, walk, pid, start, True)
        if entry is None:
            return
        new_cache[pid] = entry


def update_cursor(scan: dict, deferred: list[int], old_cache: dict) -> None:
    """Move the cursor to the first cached PID that was not rescanned.

    A cursor that returns to the start of the PID list, or passes it, closes a rotation.
    """
    previous_cursor = scan["cursor"]
    if not deferred:
        scan["cursor"] = 0
    for pid in deferred:
        if pid in old_cache:
            scan["cursor"] = pid
            break
    if scan["cursor"] == 0 or scan["cursor"] < previous_cursor:
        scan["previous_rotation"] = scan["rotation"]
        scan["rotation"] = scan["walk"]


def walk_processes(scan: dict, pids: list[int], budget: float | None) -> int:
    """Rebuild the descriptor cache within the budget; return how many PIDs were deferred."""
    walk = create_walk(budget)
    scan["walk"] += 1
    old_cache = scan["cache"]
    new_cache = {}
    for pid in order_pids(pids, old_cache, scan["cursor"]):
        visit_process(scan, walk, pid, new_cache)
    relist_aged(scan, walk, new_cache)
    scan["cache"] = new_cache
    scan["partial"] = walk["unknown"]
    update_cursor(scan, walk["deferred"], old_cache)
    return len(walk["deferred"])


def read_fdinfo(pid: int, fd_name: str) -> tuple[str | None, bool, int]:
    """Read one fdinfo file: (text or None, permission denied, monotonic ns after the read)."""
    try:
        with open(f"{PROC_DIR}/{pid}/fdinfo/{fd_name}", "rb") as handle:
            raw = handle.read()
    except PermissionError:
        return None, True, 0
    except OSError:
        return None, False, 0
    stamp = time.monotonic_ns()
    return raw.decode("utf-8", errors="replace"), False, stamp


def add_client(clients: dict, pid: int, descriptor: tuple, info: dict, stamp: int) -> None:
    """Register one parsed descriptor under its client identity."""
    fd_name, target = descriptor
    if info["client_id"] is not None:
        key = (info["pdev"], info["client_id"])
    else:
        key = (info["pdev"], pid, fd_name)
    client = clients.get(key)
    if client is None:
        client = create_client(info, stamp)
        clients[key] = client
    client["pids"].add(pid)
    if not target.startswith(RENDER_PREFIX):
        client["render_only"] = False


def read_clients(scan: dict, card_pdevs: set[str]) -> None:
    """Read the fdinfo of every cached descriptor and rebuild the client map."""
    clients = {}
    kfd_pids = set()
    incomplete_pids = set()
    for pid, entry in scan["cache"].items():
        if entry["kfd"]:
            kfd_pids.add(pid)
        if entry["incomplete"]:
            incomplete_pids.add(pid)
        for descriptor in entry["fds"]:
            text, denied, stamp = read_fdinfo(pid, descriptor[0])
            if denied:
                incomplete_pids.add(pid)
            if text is None:
                continue
            info = drm_fdinfo.parse_fdinfo(text)
            if info is None or info["driver"] not in constants.DRM_DRIVERS:
                continue
            if info["pdev"] not in card_pdevs:
                continue
            add_client(clients, pid, descriptor, info, stamp)
    scan["clients"] = clients
    scan["kfd_pids"] = kfd_pids
    scan["incomplete_pids"] = incomplete_pids


def compute_client_rates(client: dict, previous: dict | None, high_water: dict) -> dict | None:
    """Busy share per raw engine since the previous read; None without a usable interval."""
    if previous is None:
        return None
    elapsed = client["time"] - previous["time"]
    if elapsed <= 0:
        return None
    rates = {}
    for name in client["info"]["engines"]:
        busy = high_water[name] - previous["engines"].get(name, 0)
        rates[name] = min(max(100.0 * busy / elapsed, 0.0), 100.0)
    return rates


def update_engine_rates(scan: dict) -> None:
    """Compute engine rates of every client and advance the high-water baselines."""
    baselines = scan["baselines"]
    for key, client in scan["clients"].items():
        engines = client["info"]["engines"]
        if engines is None:
            continue
        previous = baselines.get(key)
        high_water = dict(previous["engines"]) if previous is not None else {}
        for name, busy in engines.items():
            high_water[name] = max(high_water.get(name, 0), busy)
        client["rates"] = compute_client_rates(client, previous, high_water)
        baselines[key] = {"engines": high_water, "time": client["time"]}
    oldest = time.monotonic_ns() - BASELINE_KEEP_NS
    for key in list(baselines):
        if key not in scan["clients"] and baselines[key]["time"] < oldest:
            del baselines[key]


def build_status(deferred: int, denied: int, incomplete: int) -> str:
    """Compose the scan status line; "" when the scan saw everything."""
    parts = []
    if deferred:
        parts.append(
            f"scan budget reached, {deferred} processes not rescanned this refresh "
            "(last known GPU fds kept)"
        )
    if denied:
        parts.append(
            f"{denied} processes not inspectable "
            "(other users' or kernel threads; sudo includes them)"
        )
    if incomplete:
        parts.append(f"{incomplete} processes whose fds or fdinfo couldn't all be read")
    return "; ".join(parts)


def scan_processes(scan: dict, cards: list[dict], budget: float | None) -> None:
    """Run one process scan for the given cards; budget in seconds, None = unlimited."""
    pids = list_pids()
    if pids is None:
        scan["cache"] = {}
        scan["clients"] = {}
        scan["kfd_pids"] = set()
        scan["incomplete_pids"] = set()
        scan["partial"] = False
        scan["status"] = PROC_STATUS
        return
    deferred = walk_processes(scan, pids, budget)
    card_pdevs = {card["pdev"] for card in cards}
    read_clients(scan, card_pdevs)
    update_engine_rates(scan)
    denied = sum(1 for entry in scan["cache"].values() if entry["denied"])
    scan["status"] = build_status(deferred, denied, len(scan["incomplete_pids"]))


def map_amd_engine(name: str, unified_video: bool) -> str:
    """Map a raw amdgpu engine name to an engine category."""
    lowered = name.lower()
    if unified_video and ("enc" in lowered or "dec" in lowered):
        return constants.ENGINE_VIDEO
    if lowered == "gfx":
        return constants.ENGINE_3D
    if lowered == "compute":
        return constants.ENGINE_COMPUTE
    if "dec" in lowered or "jpeg" in lowered:
        return constants.ENGINE_DECODE
    if "enc" in lowered:
        return constants.ENGINE_ENCODE
    if lowered == "dma":
        return constants.ENGINE_COPY
    return constants.ENGINE_OTHER


def map_intel_engine(name: str) -> str:
    """Map a raw i915 engine name to an engine category."""
    lowered = name.lower()
    if lowered == "render" or lowered.startswith("rcs"):
        return constants.ENGINE_3D
    if lowered == "compute" or lowered.startswith("ccs"):
        return constants.ENGINE_COMPUTE
    if lowered == "copy" or lowered.startswith("bcs"):
        return constants.ENGINE_COPY
    if lowered.startswith(("video", "vcs", "vecs")):
        return constants.ENGINE_VIDEO
    return constants.ENGINE_OTHER


def group_clients(scan: dict, pdev: str) -> dict:
    """Return PID to the list of its clients on the card with the given PCI address."""
    by_pid = {}
    for client in scan["clients"].values():
        if client["pdev"] != pdev:
            continue
        for pid in client["pids"]:
            holder_clients = by_pid.setdefault(pid, [])
            holder_clients.append(client)
    return by_pid


def sum_resident(clients: list[dict], field: str, incomplete: bool) -> float | None:
    """Add one resident figure over the clients of a process; None when any is unknown."""
    if incomplete:
        return None
    total = 0
    for client in clients:
        size = client["info"][field]
        if size is None:
            return None
        total += size
    return float(total)


def compute_engines(scan: dict, card: dict, clients: list[dict], blocked: bool) -> dict | None:
    """Busy share per engine category of one process; None when it is unknown."""
    if blocked or scan["partial"]:
        return None
    engines = {}
    for client in clients:
        if len(client["pids"]) > 1 or client["rates"] is None:
            return None
        for name, rate in client["rates"].items():
            if card["intel"]:
                category = map_intel_engine(name)
            else:
                category = map_amd_engine(name, card["unified_video"])
            engines[category] = min(engines.get(category, 0.0) + rate, 100.0)
    return engines


def find_reason(scan: dict, card: dict, clients: list[dict], gpu: dict, incomplete: bool) -> str:
    """First reason of the attribution gate that applies; "" when the memory is private."""
    if incomplete:
        return constants.DRM_REASON_INCOMPLETE
    if any(len(client["pids"]) > 1 for client in clients):
        return constants.DRM_REASON_SHARED_CLIENT
    if any(client["info"]["shared"] is None for client in clients):
        return constants.DRM_REASON_LEGACY
    if any(client["info"]["shared"] != 0 for client in clients):
        return constants.DRM_REASON_SHARED_BUFFERS
    if os.geteuid() != 0 and not all(client["render_only"] for client in clients):
        return constants.DRM_REASON_PRIMARY
    intel_integrated = card["intel"] and card["integrated"]
    if gpu["resident_gtt"] is None:
        return constants.DRM_REASON_LEGACY
    if gpu["resident_vram"] is None and not intel_integrated:
        return constants.DRM_REASON_LEGACY
    if scan["partial"]:
        return constants.DRM_REASON_PARTIAL_SCAN
    return ""


def apply_attribution(gpu: dict, card: dict, reason: str) -> None:
    """Charge the resident bytes to the process, or mark its memory as not attributed."""
    if reason:
        gpu["drm_reason"] = reason
        gpu["attributed"] = False
        gpu["available"].discard(constants.COUNTER_DEDICATED)
        gpu["available"].discard(constants.COUNTER_SHARED)
        return
    if card["intel"] and card["integrated"]:
        gpu["dedicated"] = gpu["resident_gtt"]
        gpu["available"].discard(constants.COUNTER_SHARED)
        return
    gpu["dedicated"] = gpu["resident_vram"]
    gpu["shared"] = gpu["resident_gtt"]


def build_proc_gpu(scan: dict, card: dict, pid: int, clients: list[dict]) -> dict:
    """Build the proc_gpu record of one process on one card."""
    counters = {constants.COUNTER_DEDICATED, constants.COUNTER_SHARED, constants.COUNTER_ENGINE}
    gpu = records.create_proc_gpu(counters)
    incomplete = pid in scan["incomplete_pids"]
    gpu["kfd"] = not card["intel"] and pid in scan["kfd_pids"]
    gpu["resident_vram"] = sum_resident(clients, "vram", incomplete)
    gpu["resident_gtt"] = sum_resident(clients, "gtt", incomplete)
    engines = compute_engines(scan, card, clients, incomplete or gpu["kfd"])
    if engines is None:
        gpu["available"].discard(constants.COUNTER_ENGINE)
    else:
        gpu["engines"] = engines
    reason = find_reason(scan, card, clients, gpu, incomplete)
    apply_attribution(gpu, card, reason)
    return gpu


def build_proc_counters(scan: dict, card: dict) -> dict[int, dict]:
    """Return PID to proc_gpu for every process holding a client on the card."""
    by_pid = group_clients(scan, card["pdev"])
    counters = {}
    for pid in sorted(by_pid):
        counters[pid] = build_proc_gpu(scan, card, pid, by_pid[pid])
    return counters


def sum_client_memory(scan: dict, card: dict) -> dict:
    """Device memory of an Intel card from its clients, each client counted once.

    Returns {"used": int or None, "has_local": bool}: used bytes (None = unknown) and
    whether any client reports local memory, which proves the card is discrete.
    """
    clients = [client for client in scan["clients"].values() if client["pdev"] == card["pdev"]]
    local_sizes = [client["info"]["vram"] for client in clients]
    has_local = any(size is not None for size in local_sizes)
    if has_local:
        sizes = local_sizes
    elif card["integrated"]:
        sizes = [client["info"]["gtt"] for client in clients]
    else:
        sizes = []
    used = None
    if sizes and all(size is not None for size in sizes):
        used = sum(sizes)
    return {"used": used, "has_local": has_local}


def main():
    pass


if __name__ == "__main__":
    main()
