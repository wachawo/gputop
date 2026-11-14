#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process inspector: cheap details, listening ports, rescan rule, mapped-file scan, parents."""
import logging
import os
import time
import traceback

import psutil

# Local imports
from gputop import constants
from gputop import modelfiles
from gputop import records
from gputop import runtimes

IDLE_PID = 0
IDLE_NAME = "System Idle"
PORT_TABLE_LIFETIME = 5.0
# Bind addresses a connection to 127.0.0.1 reaches: the wildcards and IPv4 loopback, also in
# its IPv4-mapped form. A socket bound to ::1 alone does not answer there.
WILDCARD_ADDRESSES = ("0.0.0.0", "::")
LOOPBACK_PREFIXES = ("127.", "::ffff:127.")
SCAN_MIN_INTERVAL = 5.0
SCAN_MAX_INTERVAL = 120.0
SCAN_MIN_DELTA = 64 * constants.MIB
SCAN_DELTA_SHARE = 0.10
CREATE_TIME_TOLERANCE = 0.001
MAX_PARENT_TARGETS = 2

logger = logging.getLogger(__name__)


def open_inspector() -> dict:
    """Create the inspector state.

    Keys: "details" (pid to cached detail), "port_table" (pid to ascending port list),
    "port_owners" (port to set of listening pids), "local_ports" (pid to the set of its
    ports bound to a wildcard or loopback address), "ports_time" (monotonic seconds of the
    last port table build, None before the first one).
    """
    return {
        "details": {},
        "port_table": {},
        "port_owners": {},
        "local_ports": {},
        "ports_time": None,
    }


def read_create_time(pid: int) -> float:
    """Return the live start time of a process in Unix seconds; 0.0 when it is unreadable."""
    try:
        process = psutil.Process(pid)
        return float(process.create_time())
    except (psutil.Error, OSError, ValueError):
        return 0.0


def fill_identity(detail: dict, process: psutil.Process) -> None:
    """Read name, then exe, command line and start time; flag the detail on failure."""
    try:
        detail["name"] = process.name() or constants.UNKNOWN_NAME
    except (psutil.Error, OSError):
        pass
    try:
        detail["exe"] = process.exe() or ""
        detail["cmdline"] = list(process.cmdline())
        detail["create_time"] = float(process.create_time())
    except (psutil.Error, OSError):
        detail["access_denied"] = True


def build_detail(pid: int) -> dict:
    """Build the detail of one process from its cheap, always-read properties."""
    detail = records.create_detail(pid)
    if pid == IDLE_PID:
        detail["name"] = IDLE_NAME
        return detail
    try:
        process = psutil.Process(pid)
    except (psutil.Error, OSError, ValueError):
        detail["access_denied"] = True
        return detail
    fill_identity(detail, process)
    return detail


def is_local_address(address: str) -> bool:
    """Tell whether a socket bound to the address answers a connection to 127.0.0.1."""
    if address in WILDCARD_ADDRESSES:
        return True
    return address.startswith(LOOPBACK_PREFIXES)


def read_listening_ports() -> tuple[dict, dict]:
    """Return (pid to listening TCP ports, pid to the ports among them bound locally).

    The second map holds only ports bound to a wildcard or loopback address, the ones a
    connection to 127.0.0.1 can reach. Both are empty when enumeration is not possible.
    """
    listeners = {}
    local_listeners = {}
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return {}, {}
    for connection in connections:
        if connection.status != psutil.CONN_LISTEN:
            continue
        if not connection.pid or not connection.laddr:
            continue
        ports = listeners.setdefault(connection.pid, set())
        ports.add(connection.laddr.port)
        if is_local_address(connection.laddr.ip):
            local_ports = local_listeners.setdefault(connection.pid, set())
            local_ports.add(connection.laddr.port)
    return listeners, local_listeners


def refresh_port_table(inspector: dict) -> None:
    """Rebuild the listening-port table when it is older than its lifetime."""
    now = time.monotonic()
    built = inspector["ports_time"]
    if built is not None and now - built < PORT_TABLE_LIFETIME:
        return
    inspector["ports_time"] = now
    table = {}
    owners = {}
    listeners, local_listeners = read_listening_ports()
    for pid, ports in listeners.items():
        table[pid] = sorted(ports)
        for port in ports:
            port_pids = owners.setdefault(port, set())
            port_pids.add(pid)
    inspector["port_table"] = table
    inspector["port_owners"] = owners
    inspector["local_ports"] = local_listeners


def list_exclusive_ports(inspector: dict, pid: int) -> list[int]:
    """Return the ports of a pid that no other pid listens on, ascending.

    A port bound only to a non-loopback address is left out: these ports are what gets
    probed on 127.0.0.1, where such a port belongs to somebody else or to nobody.
    """
    owners = inspector["port_owners"]
    local_ports = inspector["local_ports"].get(pid, set())
    exclusive = []
    for port in inspector["port_table"].get(pid, []):
        if owners.get(port) == {pid} and port in local_ports:
            exclusive.append(port)
    return exclusive


def is_reused_pid(cached: dict) -> bool:
    """Tell whether the process behind a cached detail was replaced or is gone."""
    known = cached["create_time"]
    if not known:
        return False
    live = read_create_time(cached["pid"])
    return abs(live - known) > CREATE_TIME_TOLERANCE


def get_detail(inspector: dict, pid: int) -> dict:
    """Return the cached detail of a pid with fresh ports; build or rebuild it when needed."""
    details = inspector["details"]
    detail = details.get(pid)
    if detail is None or is_reused_pid(detail):
        detail = build_detail(pid)
        details[pid] = detail
    refresh_port_table(inspector)
    detail["ports"] = list(inspector["port_table"].get(pid, []))
    detail["exclusive_ports"] = list_exclusive_ports(inspector, pid)
    return detail


def needs_scan(detail: dict, mem: float) -> bool:
    """Apply the rescan rule: first scan, minimum interval, backstop, memory change."""
    if detail["access_denied"] or detail["pid"] == IDLE_PID:
        return False
    if not detail["maps_time"]:
        return True
    age = time.time() - detail["maps_time"]
    if 0 <= age < SCAN_MIN_INTERVAL:
        return False
    if age < 0 or age > SCAN_MAX_INTERVAL:
        return True
    scan_mem = detail["scan_mem"]
    threshold = max(SCAN_MIN_DELTA, SCAN_DELTA_SHARE * scan_mem)
    return abs(mem - scan_mem) > threshold


def read_mapped_files(pid: int) -> list[tuple]:
    """Return (path, resident bytes) per mapped backing path; raises psutil.Error or OSError."""
    process = psutil.Process(pid)
    mapped = []
    for region in process.memory_maps(grouped=True):
        mapped.append((region.path, int(region.rss)))
    return mapped


def read_working_directory(pid: int) -> str:
    """Return the working directory of a process; "" when it cannot be read."""
    try:
        process = psutil.Process(pid)
        return process.cwd() or ""
    except (psutil.Error, OSError, ValueError):
        return ""


def add_mapped_path(path: str, resident: int, library_names: set, known: dict) -> None:
    """Sort one mapped path into the library set or the model table (keyed by file identity)."""
    if not path or modelfiles.is_pseudo_path(path):
        return
    basename = os.path.basename(path)
    library_name = runtimes.normalize_library_name(basename)
    if library_name is not None:
        library_names.add(library_name)
        return
    model = modelfiles.create_mapped_model(path, resident)
    if model is None:
        return
    identity = modelfiles.get_file_identity(path)
    existing = known.get(identity)
    if existing is None:
        known[identity] = model
    else:
        existing["resident"] = (existing["resident"] or 0) + resident


def add_model_source(known: dict, model: dict, source: str) -> None:
    """Add a model to the table, or only its source when the same file is already there."""
    identity = modelfiles.get_file_identity(model["path"])
    existing = known.get(identity)
    if existing is None:
        known[identity] = model
    elif source not in existing["sources"]:
        existing["sources"].append(source)


def add_argument_models(detail: dict, cwd: str, known: dict) -> None:
    """Add the models named by launch flags, then the path-argument hints."""
    arguments = detail["cmdline"]
    for path in modelfiles.find_cmdline_models(arguments, cwd):
        model = modelfiles.create_model_from_path(path, constants.EVIDENCE_CMDLINE)
        if model is not None:
            add_model_source(known, model, constants.EVIDENCE_CMDLINE)
    for model in modelfiles.find_path_arg_models(arguments, cwd):
        add_model_source(known, model, constants.EVIDENCE_PATH_ARG)


def get_model_order(model: dict) -> tuple:
    """Sort key: models with evidence first, then larger files first."""
    return (not modelfiles.has_model_evidence(model), -model["size"])


def has_ai_runtime(runtime_labels: list[str]) -> bool:
    """Tell whether any of the labels is an AI runtime."""
    for label in runtime_labels:
        if label in runtimes.AI_RUNTIMES:
            return True
    return False


def find_candidates(detail: dict, runtime_labels: list[str], models: list[dict]) -> list[dict]:
    """Return installed candidates when an AI runtime is loaded and nothing is proven."""
    if not has_ai_runtime(runtime_labels):
        return []
    for model in models:
        if modelfiles.has_model_evidence(model):
            return []
    return modelfiles.find_installed_models(detail["exe"])


def apply_scan(detail: dict, mapped: list[tuple]) -> None:
    """Turn the mapped files of a process into new runtimes, models and candidates lists."""
    library_names = set()
    known = {}
    for path, resident in mapped:
        add_mapped_path(path, resident, library_names, known)
    cwd = read_working_directory(detail["pid"])
    add_argument_models(detail, cwd, known)
    runtime_labels = runtimes.match_runtimes(library_names)
    models = sorted(known.values(), key=get_model_order)
    candidates = find_candidates(detail, runtime_labels, models)
    detail["runtimes"] = runtime_labels
    detail["models"] = models
    detail["candidates"] = candidates


def scan_process(inspector: dict, detail: dict, mem: float) -> None:
    """Run the mapped-file scan of one process and replace its scan results."""
    detail["maps_time"] = time.time()
    detail["scan_mem"] = float(mem)
    try:
        mapped = read_mapped_files(detail["pid"])
    except (psutil.Error, OSError, ValueError):
        detail["access_denied"] = True
        return
    try:
        apply_scan(detail, mapped)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")


def read_parent_pids(pid: int) -> list[int]:
    """Return the ancestor pids of a process, nearest first; [] when they cannot be read."""
    try:
        process = psutil.Process(pid)
        ancestors = process.parents()
        return [ancestor.pid for ancestor in ancestors]
    except (psutil.Error, OSError, ValueError):
        return []


def get_parent_targets(inspector: dict, detail: dict) -> list[dict]:
    """Return up to two ancestors as {"pid", "create_time", "exclusive_ports"}, nearest first."""
    targets = []
    if detail["parents"] is None:
        detail["parents"] = read_parent_pids(detail["pid"])
    refresh_port_table(inspector)
    for parent_pid in detail["parents"][:MAX_PARENT_TARGETS]:
        target = {
            "pid": parent_pid,
            "create_time": read_create_time(parent_pid),
            "exclusive_ports": list_exclusive_ports(inspector, parent_pid),
        }
        targets.append(target)
    return targets


def prune_inspector(inspector: dict, live_pids: set[int]) -> None:
    """Drop every cached detail whose pid is not in the live set."""
    details = inspector["details"]
    for pid in list(details):
        if pid not in live_pids:
            del details[pid]


def main():
    pass


if __name__ == "__main__":
    main()
