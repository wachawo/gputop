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
# Performance state number the driver reports when it cannot tell the state.
PSTATE_UNKNOWN = 32

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
    return decode_text(getattr(pci_info, "busId", None))


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
    except Exception as exc:
        if is_reset_error(exc):
            state["needs_reset"] = True
        return None


def read_first_element(state: dict, function_name: str, handle) -> float | None:
    """Return the first element of a pair-valued query (value, sampling period) as float."""
    result = read_nvml(state, function_name, handle)
    if result is None:
        return None
    try:
        return float(result[0])
    except (TypeError, IndexError, ValueError):
        return None


def build_adapter(index: int) -> dict | None:
    """Build the adapter record of one NVML index; None when it cannot be described."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as exc:
        logger.debug(f"NVML device {index} skipped: {type(exc).__name__}: {exc}")
        return None
    try:
        bus_id = extract_bus_id(pynvml.nvmlDeviceGetPciInfo(handle))
    except Exception:
        bus_id = ""
    try:
        name = decode_text(pynvml.nvmlDeviceGetName(handle))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = int(memory.total)
    except Exception as exc:
        logger.debug(f"NVML device {index} not listed: {type(exc).__name__}: {exc}")
        return None
    key = compute_handle_key(index, bus_id)
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
    except Exception as exc:
        logger.debug(f"NVML not available: {type(exc).__name__}: {exc}")
        return []
    try:
        return read_adapter_list()
    except Exception as exc:
        logger.debug(f"NVML enumeration failed: {type(exc).__name__}: {exc}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def create_nvidia_state(adapter: dict | None, gpu_index: int) -> dict:
    """Create the long-lived NVML state of one collector.

    Keys:
      adapter     dict or None    selected adapter record, None when nothing was enumerated
      gpu_index   int             requested index, used only without an adapter
      session     bool            True while nvmlInit is in effect for this state
      handles     list[dict]      handle entries: index (int), handle, bus_id (str), key (int)
      driver      str             driver version, '' when unknown
      cuda        str             CUDA driver version as major.minor, '' when unknown
      cursors     dict[int, int]  adapter key to the last consumed utilisation timestamp
      matcher     dict            pidns matcher state
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
        "matcher": pidns.create_matcher(),
        "needs_reset": False,
        "refreshed": None,
    }


def stop_session(state: dict) -> None:
    """Shut the session down when it is open and forget its handles."""
    if state["session"]:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    state["session"] = False
    state["handles"] = []


def start_session(state: dict) -> bool:
    """Initialise the library for this state; False when it is not available yet."""
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug(f"NVML init failed: {type(exc).__name__}: {exc}")
        return False
    state["session"] = True
    return True


def create_handle_entry(index: int, handle) -> dict:
    """Describe one device handle: index, handle, bus id and adapter key."""
    try:
        bus_id = extract_bus_id(pynvml.nvmlDeviceGetPciInfo(handle))
    except Exception:
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
    except Exception as exc:
        logger.debug(f"NVML device count failed: {type(exc).__name__}: {exc}")
        stop_session(state)
        return False
    handles = []
    reset_seen = False
    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            reset_seen = reset_seen or is_reset_error(exc)
            continue
        handles.append(create_handle_entry(index, handle))
    state["handles"] = handles
    if not handles and reset_seen:
        state["needs_reset"] = True
    return True


def format_cuda_version(encoded) -> str:
    """Turn major * 1000 + minor * 10 into 'major.minor'; '' for zero or garbage."""
    try:
        number = int(encoded)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    return f"{number // 1000}.{(number % 1000) // 10}"


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


def open_nvidia(adapter: dict | None, gpu_index: int) -> dict:
    """Open the NVML state for one adapter (None: the device with NVML index gpu_index).

    Never raises. Without the binding the state stays inert for good; when the library does
    not initialise, the state has no session and the refresh keeps retrying.
    """
    state = create_nvidia_state(adapter, gpu_index)
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
    pid = getattr(entry, "pid", None)
    if pid is None:
        return
    pid = int(pid)
    gpu = device_procs["procs"].get(pid)
    if gpu is None:
        gpu = records.create_proc_gpu({constants.COUNTER_DEDICATED, constants.COUNTER_ENGINE})
        device_procs["procs"][pid] = gpu
    if is_compute:
        device_procs["compute"].add(pid)
    used = getattr(entry, "usedGpuMemory", None)
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
    """Read the per-process utilisation samples newer than the cursor and apply them.

    No samples, an unsupported call or an error leave the cursor alone; processes without a
    new sample keep an empty engine map and so read 0 %. compute_pids is the compute-list
    set of this sample over every device; it decides the engine the SM load is booked on.
    """
    cursor = state["cursors"].get(entry["key"], 0)
    samples = read_nvml(state, "nvmlDeviceGetProcessUtilization", entry["handle"], cursor)
    if not samples:
        return
    newest = reduce_utilisation(samples)
    if not newest:
        return
    timestamps = []
    for sample in newest.values():
        timestamps.append(int(sample.timeStamp))
    state["cursors"][entry["key"]] = max(timestamps)
    for pid, sample in newest.items():
        gpu = device_procs["procs"].get(pid)
        if gpu is None:
            continue
        sm_engine = constants.ENGINE_3D
        if pid in compute_pids:
            sm_engine = constants.ENGINE_COMPUTE
        add_engine_load(gpu, sm_engine, getattr(sample, "smUtil", 0))
        add_engine_load(gpu, constants.ENGINE_ENCODE, getattr(sample, "encUtil", 0))
        add_engine_load(gpu, constants.ENGINE_DECODE, getattr(sample, "decUtil", 0))


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
    except Exception as exc:
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
    device["mem_total"] = convert_int(getattr(info, "total", None))
    device["mem_used"] = convert_int(getattr(info, "used", None))
    reserved = 0
    if has_reserved:
        reserved = getattr(info, "reserved", 0) or 0
    device["mem_reserved"] = int(reserved)


def read_load(state: dict, handle, device: dict) -> None:
    """Fill GPU utilisation, memory activity, encoder and decoder load."""
    rates = read_nvml(state, "nvmlDeviceGetUtilizationRates", handle)
    if rates is not None:
        device["util_gpu"] = convert_float(getattr(rates, "gpu", None))
        device["util_mem"] = convert_float(getattr(rates, "memory", None))
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
    if pstate == getattr(pynvml, "NVML_PSTATE_UNKNOWN", PSTATE_UNKNOWN):
        return None
    return pstate


def read_thermal(state: dict, handle, device: dict) -> None:
    """Fill temperature, power draw and limit, fan and performance state."""
    sensor = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
    device["temp"] = convert_float(read_nvml(state, "nvmlDeviceGetTemperature", handle, sensor))
    device["power_w"] = read_milliwatts(state, "nvmlDeviceGetPowerUsage", handle)
    device["power_limit_w"] = read_milliwatts(state, "nvmlDeviceGetEnforcedPowerLimit", handle)
    device["fan"] = convert_int(read_nvml(state, "nvmlDeviceGetFanSpeed", handle))
    device["pstate"] = read_performance_state(state, handle)


def read_clock(state: dict, function_name: str, handle, clock_name: str) -> int | None:
    """Return one clock in MHz; None when the clock id or the reading is missing."""
    clock_id = getattr(pynvml, clock_name, None)
    if clock_id is None:
        return None
    return convert_int(read_nvml(state, function_name, handle, clock_id))


def read_clocks(state: dict, handle, device: dict) -> None:
    """Fill the SM, memory and video clocks."""
    device["sm_clock"] = read_clock(state, "nvmlDeviceGetClockInfo", handle, "NVML_CLOCK_SM")
    device["sm_clock_max"] = read_clock(
        state, "nvmlDeviceGetMaxClockInfo", handle, "NVML_CLOCK_SM"
    )
    device["mem_clock"] = read_clock(state, "nvmlDeviceGetClockInfo", handle, "NVML_CLOCK_MEM")
    device["video_clock"] = read_clock(
        state, "nvmlDeviceGetClockInfo", handle, "NVML_CLOCK_VIDEO"
    )


def read_throughput(state: dict, handle, counter_name: str) -> int | None:
    """Return one PCIe throughput counter in KB/s."""
    counter_id = getattr(pynvml, counter_name, None)
    if counter_id is None:
        return None
    return convert_int(read_nvml(state, "nvmlDeviceGetPcieThroughput", handle, counter_id))


def read_pcie(state: dict, handle, device: dict) -> None:
    """Fill PCIe throughput, link generation and link width."""
    device["pcie_rx_kbs"] = read_throughput(state, handle, "NVML_PCIE_UTIL_RX_BYTES")
    device["pcie_tx_kbs"] = read_throughput(state, handle, "NVML_PCIE_UTIL_TX_BYTES")
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
    try:
        return f"{int(capability[0])}.{int(capability[1])}"
    except (TypeError, IndexError, ValueError):
        return ""


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
    return convert_int(getattr(info, "used", None))


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


def collect_compute_pids(all_device_procs: list[dict]) -> set:
    """Return the pids found in the compute list of any device in this sample."""
    compute_pids = set()
    for device_procs in all_device_procs:
        compute_pids.update(device_procs["compute"])
    return compute_pids


def resolve_foreign_pids(
    state: dict, all_device_procs: list[dict], compute_pids: set
) -> tuple[set, dict]:
    """Run the PID-namespace resolution over every device.

    Returns (foreign pids, host pid to (local pid, confidence)). The matcher is fed on every
    call, with two empty lists when nothing is foreign.
    """
    reported = set()
    for device_procs in all_device_procs:
        reported.update(device_procs["procs"])
    foreign = pidns.find_foreign_pids(reported)
    if not foreign:
        pidns.match_foreign_pids(state["matcher"], [], [])
        return foreign, {}
    host_pids = sorted(foreign & compute_pids)
    local_pids = pidns.list_cuda_candidates(reported - foreign)
    matches = pidns.match_foreign_pids(state["matcher"], host_pids, local_pids)
    return foreign, matches


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
    all_device_procs = []
    for entry in state["handles"]:
        all_device_procs.append(read_device_procs(state, entry["handle"]))
    compute_pids = collect_compute_pids(all_device_procs)
    position = find_selected_position(state)
    if position is not None:
        selected_entry = state["handles"][position]
        read_selected_device(
            state, selected_entry, all_device_procs[position], compute_pids, sample
        )
    foreign, matches = resolve_foreign_pids(state, all_device_procs, compute_pids)
    if position is None:
        return
    selected_procs = all_device_procs[position]["procs"]
    local, outside = split_foreign_procs(selected_procs, foreign, matches)
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
