#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVML session, NVIDIA adapter list, device telemetry, per-process memory and load."""
import logging
import time
import traceback

try:
    import pynvml
except ImportError:
    pynvml = None

# Local imports
from gputop import constants
from gputop import pidns
from gputop import records

# NVML error codes after which the session has to be rebuilt:
# library not initialised, kernel module not loaded, GPU lost.
RESET_ERROR_CODES = (1, 9, 15)
REFRESH_SECONDS = 5.0
# Per-process memory at or above this value is the "not available" sentinel of the driver.
MEMORY_NOT_AVAILABLE = 1 << 63
# A per-process load sample stays on screen this long without a newer one: the driver
# publishes samples on its own clock, so a refresh can fall between two of them.
LOAD_HOLD_SECONDS = 2.0

# Throttle bits worth reporting, in display order. Idle, sync boost and the display clock
# setting are normal operating states and are left out.
THROTTLE_BITS = (
    (0x2, "app-clocks"),
    (0x4, "sw-power-cap"),
    (0x8, "hw-slowdown"),
    (0x20, "sw-thermal"),
    (0x40, "hw-thermal"),
    (0x80, "power-brake"),
)

logger = logging.getLogger(__name__)


def decode_text(value) -> str:
    """Return a driver string as text; older bindings hand out bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def convert_float(value) -> float | None:
    """Return the reading as float, None staying None."""
    if value is None:
        return None
    return float(value)


def convert_int(value) -> int | None:
    """Return the reading as int, None staying None."""
    if value is None:
        return None
    return int(value)


def is_reset_error(exc: BaseException) -> bool:
    """Tell whether the exception carries an NVML code that asks for a new session."""
    return getattr(exc, "value", None) in RESET_ERROR_CODES


def extract_bus_id(pci_info) -> str:
    """Return the PCI bus id of an NVML PCI info structure, '' when absent."""
    if pci_info is None:
        return ""
    return decode_text(pci_info.busId)


def compute_handle_key(index: int, bus_id: str) -> int:
    """Return the adapter key of a device: from the bus id, else index + 1."""
    if bus_id:
        return records.compute_adapter_key(bus_id)
    return index + 1


def read_nvml(state: dict, function_name: str, *arguments):
    """Run one guarded NVML query and return its result, None on any failure.

    Unsupported, forbidden, failed and missing-in-this-binding all read as None. A reset
    error additionally flags the state so the next refresh rebuilds the session.
    """
    try:
        function = getattr(pynvml, function_name)
        return function(*arguments)
    except (pynvml.NVMLError, AttributeError) as exc:
        if is_reset_error(exc):
            state["needs_reset"] = True
        return None


def read_first_element(state: dict, function_name: str, handle) -> float | None:
    """Return the first element of a pair-valued query (value, sampling period) as float."""
    result = read_nvml(state, function_name, handle)
    if result is None:
        return None
    return float(result[0])


def build_adapter(index: int) -> dict | None:
    """Build the adapter record of one NVML index; None when it cannot be described."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    except pynvml.NVMLError as exc:
        logger.debug(f"NVML device {index} skipped: {type(exc).__name__}: {exc}")
        return None
    try:
        name = decode_text(pynvml.nvmlDeviceGetName(handle))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = int(memory.total)
    except pynvml.NVMLError as exc:
        logger.debug(f"NVML device {index} not listed: {type(exc).__name__}: {exc}")
        return None
    key = create_handle_entry(index, handle)["key"]
    return records.create_adapter(key, name, constants.VENDOR_NVIDIA, total, 0)


def read_adapter_list() -> list[dict]:
    """Enumerate the devices of an open session in NVML index order."""
    found = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        adapter = build_adapter(index)
        if adapter is not None:
            found.append(adapter)
    return found


def list_nvidia_adapters() -> list[dict]:
    """Return the NVIDIA adapters in NVML index order through a short session.

    [] when the binding is missing or the library does not initialise. Never raises.
    """
    if pynvml is None:
        return []
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.debug(f"NVML not available: {type(exc).__name__}: {exc}")
        return []
    try:
        return read_adapter_list()
    except pynvml.NVMLError as exc:
        logger.debug(f"NVML enumeration failed: {type(exc).__name__}: {exc}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def create_shared_reads() -> dict:
    """Create the per-tick reads that collectors of one program share.

    The process lists of every device and the PID-namespace resolution are machine-wide,
    so the collectors of several NVIDIA cards read them once per tick instead of once each.

    Keys:
      readers     set[int]            ids of the states that already used the current reads
      procs       dict[int, dict]     NVML index to the device process map of this tick
      compute     set[int]            pids in the compute list of any device
      foreign     set[int]            pids of another PID namespace
      matches     dict[int, tuple]    host pid to (local pid, confidence)
      matcher     dict                pidns matcher state
    """
    return {
        "readers": set(),
        "procs": {},
        "compute": set(),
        "foreign": set(),
        "matches": {},
        "matcher": pidns.create_matcher(),
    }


def create_nvidia_state(adapter: dict | None, gpu_index: int, shared: dict | None) -> dict:
    """Create the long-lived NVML state of one collector.

    Keys:
      adapter     dict or None    selected adapter record, None when nothing was enumerated
      gpu_index   int             requested index, used only without an adapter
      session     bool            True while nvmlInit is in effect for this state
      handles     list[dict]      handle entries: index (int), handle, bus_id (str), key (int)
      driver      str             driver version, '' when unknown
      cuda        str             CUDA driver version as major.minor, '' when unknown
      cursors     dict[int, int]  adapter key to the last consumed utilisation timestamp
      loads       dict[int, dict] pid to its last load sample: time (monotonic), sm, enc, dec
      shared      dict            shared per-tick reads, see create_shared_reads
      needs_reset bool            a reset error was seen; the next refresh rebuilds the session
      refreshed   float or None   monotonic time of the last refresh, None before the first
    """
    return {
        "adapter": adapter,
        "gpu_index": gpu_index,
        "session": False,
        "handles": [],
        "driver": "",
        "cuda": "",
        "cursors": {},
        "loads": {},
        "shared": shared if shared is not None else create_shared_reads(),
        "needs_reset": False,
        "refreshed": None,
    }


def stop_session(state: dict) -> None:
    """Shut the session down when it is open and forget its handles."""
    if state["session"]:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
    state["session"] = False
    state["handles"] = []


def start_session(state: dict) -> bool:
    """Initialise the library for this state; False when it is not available yet."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.debug(f"NVML init failed: {type(exc).__name__}: {exc}")
        return False
    state["session"] = True
    return True


def create_handle_entry(index: int, handle) -> dict:
    """Describe one device handle: index, handle, bus id and adapter key."""
    try:
        bus_id = extract_bus_id(pynvml.nvmlDeviceGetPciInfo(handle))
    except pynvml.NVMLError:
        bus_id = ""
    return {
        "index": index,
        "handle": handle,
        "bus_id": bus_id,
        "key": compute_handle_key(index, bus_id),
    }


def load_handles(state: dict) -> bool:
    """Re-read the handle list; False when the session had to be dropped."""
    try:
        count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as exc:
        logger.debug(f"NVML device count failed: {type(exc).__name__}: {exc}")
        stop_session(state)
        return False
    handles = []
    reset_seen = False
    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except pynvml.NVMLError as exc:
            reset_seen = reset_seen or is_reset_error(exc)
            continue
        handles.append(create_handle_entry(index, handle))
    state["handles"] = handles
    if not handles and reset_seen:
        state["needs_reset"] = True
    return True


def format_cuda_version(encoded: int | None) -> str:
    """Turn major * 1000 + minor * 10 into 'major.minor'; '' when unknown."""
    if not encoded:
        return ""
    return f"{encoded // 1000}.{(encoded % 1000) // 10}"


def load_versions(state: dict) -> None:
    """Read the driver and CUDA driver versions; CUDA keeps its previous value on failure."""
    state["driver"] = decode_text(read_nvml(state, "nvmlSystemGetDriverVersion"))
    encoded = read_nvml(state, "nvmlSystemGetCudaDriverVersion_v2")
    if encoded is None:
        encoded = read_nvml(state, "nvmlSystemGetCudaDriverVersion")
    version = format_cuda_version(encoded)
    if version:
        state["cuda"] = version


def refresh_session(state: dict) -> None:
    """Low-frequency refresh: rebuild after a reset error, re-read handles and versions."""
    state["refreshed"] = time.monotonic()
    if state["needs_reset"]:
        stop_session(state)
        state["needs_reset"] = False
    if not state["session"] and not start_session(state):
        return
    if not load_handles(state):
        return
    load_versions(state)


def is_refresh_due(state: dict) -> bool:
    """Tell whether the 5-second refresh has to run before this sample."""
    if state["refreshed"] is None:
        return True
    return time.monotonic() - state["refreshed"] >= REFRESH_SECONDS


def open_nvidia(adapter: dict | None, gpu_index: int, shared: dict | None = None) -> dict:
    """Open the NVML state for one adapter (None: the device with NVML index gpu_index).

    `shared` is the create_shared_reads record of the program when several collectors are
    open at once; None gives this state its own. Never raises. Without the binding the
    state stays inert for good; when the library does not initialise, the state has no
    session and the refresh keeps retrying.
    """
    state = create_nvidia_state(adapter, gpu_index, shared)
    if pynvml is None:
        return state
    try:
        refresh_session(state)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
    return state


def close_nvidia(state: dict) -> None:
    """Shut the session down when it is open. Idempotent, never raises."""
    if pynvml is None:
        return
    stop_session(state)


def fold_process_entry(device_procs: dict, entry, is_compute: bool) -> None:
    """Fold one running-process entry into the per-device process map."""
    pid = int(entry.pid)
    gpu = device_procs["procs"].get(pid)
    if gpu is None:
        gpu = records.create_proc_gpu({constants.COUNTER_DEDICATED, constants.COUNTER_ENGINE})
        device_procs["procs"][pid] = gpu
    if is_compute:
        device_procs["compute"].add(pid)
    used = entry.usedGpuMemory
    if used is None or used >= MEMORY_NOT_AVAILABLE:
        gpu["attributed"] = False
        return
    gpu["dedicated"] = max(gpu["dedicated"], float(used))


def read_device_procs(state: dict, handle) -> dict:
    """Read the compute and graphics process lists of one device.

    Returns {"procs": dict[int, proc_gpu], "compute": set[int]}. A process in both lists
    keeps the larger memory value; a failing list reads as empty.
    """
    device_procs = {"procs": {}, "compute": set()}
    process_lists = (
        ("nvmlDeviceGetComputeRunningProcesses", True),
        ("nvmlDeviceGetGraphicsRunningProcesses", False),
    )
    for function_name, is_compute in process_lists:
        entries = read_nvml(state, function_name, handle)
        for entry in entries or []:
            fold_process_entry(device_procs, entry, is_compute)
    return device_procs


def reduce_utilisation(samples) -> dict:
    """Keep the newest utilisation sample per pid."""
    newest = {}
    for sample in samples:
        pid = int(sample.pid)
        kept = newest.get(pid)
        if kept is None or sample.timeStamp > kept.timeStamp:
            newest[pid] = sample
    return newest


def add_engine_load(gpu: dict, engine: str, value) -> None:
    """Add a utilisation percentage to one engine of a proc_gpu record."""
    if not value:
        return
    gpu["engines"][engine] = gpu["engines"].get(engine, 0.0) + float(value)


def apply_utilisation(
    state: dict, entry: dict, device_procs: dict, compute_pids: set
) -> None:
    """Read the per-process utilisation samples newer than the cursor and apply the loads.

    No samples, an unsupported call or an error leave the cursor alone. A process without a
    new sample keeps its last one for LOAD_HOLD_SECONDS, so a refresh that falls between
    two driver samples does not read 0 %. compute_pids is the compute-list set of this
    sample over every device; it decides the engine the SM load is booked on.
    """
    cursor = state["cursors"].get(entry["key"], 0)
    samples = read_nvml(state, "nvmlDeviceGetProcessUtilization", entry["handle"], cursor)
    newest = reduce_utilisation(samples or [])
    now = time.monotonic()
    loads = state["loads"]
    for pid, sample in newest.items():
        loads[pid] = {
            "time": now, "sm": sample.smUtil, "enc": sample.encUtil, "dec": sample.decUtil
        }
    if newest:
        timestamps = [int(sample.timeStamp) for sample in newest.values()]
        state["cursors"][entry["key"]] = max(timestamps)
    for pid in list(loads):
        gpu = device_procs["procs"].get(pid)
        load = loads[pid]
        if gpu is None or now - load["time"] > LOAD_HOLD_SECONDS:
            del loads[pid]
            continue
        sm_engine = constants.ENGINE_3D
        if pid in compute_pids:
            sm_engine = constants.ENGINE_COMPUTE
        add_engine_load(gpu, sm_engine, load["sm"])
        add_engine_load(gpu, constants.ENGINE_ENCODE, load["enc"])
        add_engine_load(gpu, constants.ENGINE_DECODE, load["dec"])


def read_identity(state: dict, handle, device: dict) -> None:
    """Fill name and PCI bus id; read first so identity survives other failures."""
    name = read_nvml(state, "nvmlDeviceGetName", handle)
    if name is not None:
        device["name"] = decode_text(name)
    device["pci_bus"] = extract_bus_id(read_nvml(state, "nvmlDeviceGetPciInfo", handle))


def read_memory_info(state: dict, handle):
    """Return (memory info, has_reserved): the v2 structure, else the plain one."""
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
    except (pynvml.NVMLError, AttributeError, TypeError) as exc:
        if is_reset_error(exc):
            state["needs_reset"] = True
        info = None
    if info is not None:
        return info, True
    return read_nvml(state, "nvmlDeviceGetMemoryInfo", handle), False


def read_memory(state: dict, handle, device: dict) -> None:
    """Fill total, used and reserved device memory."""
    info, has_reserved = read_memory_info(state, handle)
    if info is None:
        return
    device["mem_total"] = int(info.total)
    device["mem_used"] = int(info.used)
    device["mem_reserved"] = int(info.reserved) if has_reserved else 0


def read_load(state: dict, handle, device: dict) -> None:
    """Fill GPU utilisation, memory activity, encoder and decoder load."""
    rates = read_nvml(state, "nvmlDeviceGetUtilizationRates", handle)
    if rates is not None:
        device["util_gpu"] = float(rates.gpu)
        device["util_mem"] = float(rates.memory)
    device["enc"] = read_first_element(state, "nvmlDeviceGetEncoderUtilization", handle)
    device["dec"] = read_first_element(state, "nvmlDeviceGetDecoderUtilization", handle)


def read_milliwatts(state: dict, function_name: str, handle) -> float | None:
    """Return a milliwatt query in watts."""
    milliwatts = read_nvml(state, function_name, handle)
    if milliwatts is None:
        return None
    return float(milliwatts) / 1000.0


def read_performance_state(state: dict, handle) -> int | None:
    """Return the performance state number; None when unreadable or reported as unknown."""
    pstate = convert_int(read_nvml(state, "nvmlDeviceGetPerformanceState", handle))
    if pstate == pynvml.NVML_PSTATE_UNKNOWN:
        return None
    return pstate


def read_thermal(state: dict, handle, device: dict) -> None:
    """Fill temperature, power draw and limit, fan and performance state."""
    sensor = pynvml.NVML_TEMPERATURE_GPU
    device["temp"] = convert_float(read_nvml(state, "nvmlDeviceGetTemperature", handle, sensor))
    device["power_w"] = read_milliwatts(state, "nvmlDeviceGetPowerUsage", handle)
    device["power_limit_w"] = read_milliwatts(state, "nvmlDeviceGetEnforcedPowerLimit", handle)
    device["fan"] = convert_int(read_nvml(state, "nvmlDeviceGetFanSpeed", handle))
    device["pstate"] = read_performance_state(state, handle)


def read_clock(state: dict, function_name: str, handle, clock_id: int) -> int | None:
    """Return one clock in MHz; None when the reading is missing."""
    return convert_int(read_nvml(state, function_name, handle, clock_id))


def read_clocks(state: dict, handle, device: dict) -> None:
    """Fill the SM, memory and video clocks."""
    current = "nvmlDeviceGetClockInfo"
    device["sm_clock"] = read_clock(state, current, handle, pynvml.NVML_CLOCK_SM)
    device["sm_clock_max"] = read_clock(
        state, "nvmlDeviceGetMaxClockInfo", handle, pynvml.NVML_CLOCK_SM
    )
    device["mem_clock"] = read_clock(state, current, handle, pynvml.NVML_CLOCK_MEM)
    device["video_clock"] = read_clock(state, current, handle, pynvml.NVML_CLOCK_VIDEO)


def read_throughput(state: dict, handle, counter_id: int) -> int | None:
    """Return one PCIe throughput counter in KB/s."""
    return convert_int(read_nvml(state, "nvmlDeviceGetPcieThroughput", handle, counter_id))


def read_pcie(state: dict, handle, device: dict) -> None:
    """Fill PCIe throughput, link generation and link width."""
    device["pcie_rx_kbs"] = read_throughput(state, handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
    device["pcie_tx_kbs"] = read_throughput(state, handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
    device["pcie_gen"] = convert_int(
        read_nvml(state, "nvmlDeviceGetCurrPcieLinkGeneration", handle)
    )
    device["pcie_width"] = convert_int(
        read_nvml(state, "nvmlDeviceGetCurrPcieLinkWidth", handle)
    )


def decode_throttle(mask: int) -> list[str]:
    """Return the names of the reported throttle bits set in the mask, in table order."""
    names = []
    for bit, name in THROTTLE_BITS:
        if mask & bit:
            names.append(name)
    return names


def read_throttle(state: dict, handle) -> list[str] | None:
    """Return the active throttle reasons; None when neither NVML call answers."""
    mask = read_nvml(state, "nvmlDeviceGetCurrentClocksEventReasons", handle)
    if mask is None:
        mask = read_nvml(state, "nvmlDeviceGetCurrentClocksThrottleReasons", handle)
    if mask is None:
        return None
    return decode_throttle(int(mask))


def read_compute_capability(state: dict, handle) -> str:
    """Return the CUDA compute capability as 'major.minor', '' when unavailable."""
    capability = read_nvml(state, "nvmlDeviceGetCudaComputeCapability", handle)
    if capability is None:
        return ""
    return f"{int(capability[0])}.{int(capability[1])}"


def read_device(state: dict, entry: dict) -> dict:
    """Read one telemetry record; every query is guarded on its own."""
    handle = entry["handle"]
    device = records.create_device(entry["index"], constants.SOURCE_NVML)
    read_identity(state, handle, device)
    read_memory(state, handle, device)
    read_load(state, handle, device)
    read_thermal(state, handle, device)
    read_clocks(state, handle, device)
    read_pcie(state, handle, device)
    device["throttle"] = read_throttle(state, handle)
    device["compute_cap"] = read_compute_capability(state, handle)
    return device


def find_selected_position(state: dict) -> int | None:
    """Return the position in the handle list of the selected device, None without one.

    With an adapter: the handle whose key equals the adapter key, only when exactly one
    matches. Without an adapter: the handle whose NVML index equals gpu_index.
    """
    adapter = state["adapter"]
    positions = []
    for position, entry in enumerate(state["handles"]):
        if adapter is not None and entry["key"] == adapter["key"]:
            positions.append(position)
        if adapter is None and entry["index"] == state["gpu_index"]:
            positions.append(position)
    if len(positions) != 1:
        return None
    return positions[0]


def read_used_memory(state: dict, handle) -> int | None:
    """Return device-wide used memory from the plain memory query, None on failure."""
    info = read_nvml(state, "nvmlDeviceGetMemoryInfo", handle)
    if info is None:
        return None
    return int(info.used)


def read_selected_device(
    state: dict, entry: dict, device_procs: dict, compute_pids: set, sample: dict
) -> None:
    """Fill used memory, per-process load and telemetry of the selected device."""
    sample["vram_used"] = read_used_memory(state, entry["handle"])
    apply_utilisation(state, entry, device_procs, compute_pids)
    device = read_device(state, entry)
    if state["adapter"] is not None and not device["pci_bus"]:
        return
    sample["device"] = device


def resolve_foreign_pids(shared: dict) -> None:
    """Run the PID-namespace resolution over every device of the shared reads.

    Fills foreign (pids) and matches (host pid to (local pid, confidence)). The matcher is
    fed on every call, with two empty lists when nothing is foreign.
    """
    reported = set()
    for device_procs in shared["procs"].values():
        reported.update(device_procs["procs"])
    shared["foreign"] = pidns.find_foreign_pids(reported)
    if not shared["foreign"]:
        pidns.match_foreign_pids(shared["matcher"], [], [])
        shared["matches"] = {}
        return
    host_pids = sorted(shared["foreign"] & shared["compute"])
    local_pids = pidns.list_cuda_candidates(reported - shared["foreign"])
    shared["matches"] = pidns.match_foreign_pids(shared["matcher"], host_pids, local_pids)


def refresh_shared_reads(state: dict) -> dict:
    """Return the shared reads of this tick, reading them when this state starts a new one.

    A state that already used the current reads is on its next sample, which is a new tick
    for every collector; the other states of the program then reuse what it read.
    """
    shared = state["shared"]
    reader = id(state)
    if shared["readers"] and reader not in shared["readers"]:
        shared["readers"].add(reader)
        return shared
    shared["readers"] = {reader}
    shared["procs"] = {}
    shared["compute"] = set()
    for entry in state["handles"]:
        device_procs = read_device_procs(state, entry["handle"])
        shared["procs"][entry["index"]] = device_procs
        shared["compute"].update(device_procs["compute"])
    resolve_foreign_pids(shared)
    return shared


def get_outside_order(item: tuple) -> tuple:
    """Sort key of outside rows: larger memory first, then smaller host pid."""
    return (-item[1]["dedicated"], item[0])


def split_foreign_procs(procs: dict, foreign: set, matches: dict) -> tuple[dict, dict]:
    """Split one device map into (local map, outside map).

    Matched foreign records move to their local pid with inferred_by set; the others go to
    the outside map under the host pid. Local records are placed first, so a local pid equal
    to a host pid that still has to move cannot be overwritten.
    """
    local = {}
    moved = []
    outside_items = []
    for pid, gpu in procs.items():
        if pid not in foreign:
            local[pid] = gpu
        elif pid in matches:
            moved.append((pid, gpu))
        else:
            outside_items.append((pid, gpu))
    for host_pid, gpu in moved:
        local_pid, confidence = matches[host_pid]
        gpu["inferred_by"] = confidence
        local[local_pid] = gpu
    outside_items.sort(key=get_outside_order)
    return local, dict(outside_items)


def fill_sample(state: dict, sample: dict) -> None:
    """Take one sample from an open session into a fresh backend sample."""
    sample["driver"] = state["driver"]
    sample["cuda"] = state["cuda"]
    shared = refresh_shared_reads(state)
    position = find_selected_position(state)
    if position is None:
        return
    selected_entry = state["handles"][position]
    device_procs = shared["procs"].get(selected_entry["index"])
    if device_procs is None:
        return
    read_selected_device(state, selected_entry, device_procs, shared["compute"], sample)
    local, outside = split_foreign_procs(
        device_procs["procs"], shared["foreign"], shared["matches"]
    )
    sample["procs"] = local
    sample["outside"] = outside


def sample_nvidia(state: dict) -> dict:
    """Return one backend sample for the selected NVIDIA device. Never raises.

    A state without a session (no binding, no driver yet) yields the default sample.
    """
    if pynvml is None:
        return records.create_backend_sample()
    sample = records.create_backend_sample()
    try:
        if is_refresh_due(state):
            refresh_session(state)
        if not state["session"]:
            return sample
        fill_sample(state, sample)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        return records.create_backend_sample()
    return sample


def main():
    pass


if __name__ == "__main__":
    main()
