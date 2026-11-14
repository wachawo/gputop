#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Device telemetry of AMD and Intel cards from sysfs, hwmon, the i915 PMU and RAPL."""

import os
import re
import time

# Local imports
from gputop import constants
from gputop import drm_cards
from gputop import drm_pmu
from gputop import records

# sysfs locations
HWMON_CLASS_DIR = "/sys/class/hwmon"
POWERCAP_DIR = "/sys/class/powercap"
PARANOID_PATH = "/proc/sys/kernel/perf_event_paranoid"

# hwmon labels and unit conversions
TEMP_LABEL_EDGE = "edge"
CLOCK_LABEL_GFX = "sclk"
CLOCK_LABEL_MEM = "mclk"
CORETEMP_NAME = "coretemp"
PACKAGE_LABEL = "Package id"
PACKAGE_LABEL_FIRST = "Package id 0"
MILLI = 1000
MICRO = 1_000_000
NANO = 1_000_000_000

# Text labels of fallback sensors and power sources
TEMP_KIND_CPU = "CPU package"
POWER_KIND_BOARD = "Board"
POWER_KIND_RAPL = "RAPL uncore"

# PCIe link speed in GT/s to generation
LINK_GENERATIONS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
DPM_CLOCK_PATTERN = re.compile(r"(\d+)\s*mhz", re.IGNORECASE)

# RAPL zone discovery
RAPL_PACKAGE_PATTERN = re.compile(r"intel-rapl:\d+")
RAPL_ZONE_NAME = "uncore"
RAPL_DENIED = "denied"
RAPL_MISSING = "missing"

# Intel throttle files of gt0 with their display labels, in display order
THROTTLE_FILES = (
    ("throttle_reason_pl1", "PL1"),
    ("throttle_reason_pl2", "PL2"),
    ("throttle_reason_pl4", "PL4"),
    ("throttle_reason_prochot", "PROCHOT"),
    ("throttle_reason_ratl", "RATL"),
    ("throttle_reason_thermal", "thermal"),
    ("throttle_reason_vr_tdc", "VR-TDC"),
    ("throttle_reason_vr_thermalert", "VR-therm"),
)

# Parts of the Intel metrics explanation line
STATUS_PMU_DENIED = (
    "GPU and video load need perf privileges this user lacks (perf_event_paranoid={value})"
)
STATUS_PMU_MISSING = "no readable supported i915 performance counters for this GPU"
STATUS_PMU_SHARED = "performance counters can't attribute load with multiple Intel GPUs"
STATUS_RAPL_DENIED = "GPU power needs root (RAPL energy is root-only)"
STATUS_RAPL_MISSING = "no RAPL uncore zone found"
STATUS_LINE = "Intel device metrics unavailable: {parts}. N/A is unknown, not zero."

# Process-wide RAPL latch: zone path once found, failure reason once failed (never retried)
RAPL_LATCH = {"searched": False, "path": "", "reason": ""}


def create_intel_state(card: dict, cards: list[dict]) -> dict:
    """Create the telemetry state of one Intel card and open its PMU.

    Keys:
        card: dict, the card record the state belongs to.
        pmu: dict, PMU state of drm_pmu.open_pmu.
        energy: dict[str, tuple[int, int]], energy file path to (microjoules,
            monotonic nanoseconds) of the previous read.
    """
    return {"card": card, "pmu": drm_pmu.open_pmu(card, cards), "energy": {}}


def list_sorted(path: str) -> list[str]:
    """Return the sorted entry names of a directory, [] when it cannot be listed."""
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def find_hwmon_dir(device_dir: str) -> str:
    """Return the first hwmon directory of a device, "" when it has none."""
    hwmon_root = os.path.join(device_dir, "hwmon")
    names = list_sorted(hwmon_root)
    return os.path.join(hwmon_root, names[0]) if names else ""


def list_sensor_labels(hwmon_dir: str, prefix: str) -> list[tuple[str, str]]:
    """Return (sensor name, label text) of every <prefix>N_label file, sorted by name."""
    labels = []
    for name in list_sorted(hwmon_dir):
        if not name.startswith(prefix) or not name.endswith("_label"):
            continue
        label = drm_cards.read_text(os.path.join(hwmon_dir, name))
        if label is not None:
            labels.append((name[:-len("_label")], label))
    return labels


def read_labelled_input(hwmon_dir: str, prefix: str, wanted: str) -> tuple[bool, int | None]:
    """Find the sensor with the wanted label: (label found, value of its input file)."""
    for sensor, label in list_sensor_labels(hwmon_dir, prefix):
        if label == wanted:
            return True, drm_cards.read_number(os.path.join(hwmon_dir, f"{sensor}_input"))
    return False, None


def read_temperature(hwmon_dir: str) -> float | None:
    """Temperature in degrees C from the edge sensor, else from temp1."""
    if not hwmon_dir:
        return None
    found, reading = read_labelled_input(hwmon_dir, "temp", TEMP_LABEL_EDGE)
    if not found:
        reading = drm_cards.read_number(os.path.join(hwmon_dir, "temp1_input"))
    return None if reading is None else float(reading // MILLI)


def read_clock(hwmon_dir: str, wanted: str) -> int | None:
    """Clock in MHz of the frequency sensor with the wanted label."""
    if not hwmon_dir:
        return None
    reading = read_labelled_input(hwmon_dir, "freq", wanted)[1]
    return None if reading is None else reading // MICRO


def read_fan(hwmon_dir: str) -> int | None:
    """Fan speed as a percentage of its maximum; None without both readings."""
    if not hwmon_dir:
        return None
    speed = drm_cards.read_number(os.path.join(hwmon_dir, "fan1_input"))
    speed_max = drm_cards.read_number(os.path.join(hwmon_dir, "fan1_max"))
    if speed is None or speed_max is None or speed_max <= 0:
        return None
    return round(100 * speed / speed_max)


def fill_amd_power(device: dict, hwmon_dir: str) -> None:
    """Fill power draw, its label and the power limit of an amdgpu card."""
    if not hwmon_dir:
        return
    power = drm_cards.read_number(os.path.join(hwmon_dir, "power1_average"))
    if power is None or power == 0:
        instant = drm_cards.read_number(os.path.join(hwmon_dir, "power1_input"))
        if instant is not None:
            power = instant
    if power is not None:
        device["power_w"] = power / MICRO
        device["power_kind"] = drm_cards.read_text(os.path.join(hwmon_dir, "power1_label")) or ""
    limit = drm_cards.read_number(os.path.join(hwmon_dir, "power1_cap"))
    if limit is not None:
        device["power_limit_w"] = limit / MICRO


def read_max_clock(device_dir: str) -> int | None:
    """Highest graphics clock level of pp_dpm_sclk in MHz."""
    content = drm_cards.read_text(os.path.join(device_dir, "pp_dpm_sclk"))
    if not content:
        return None
    levels = [int(level) for level in DPM_CLOCK_PATTERN.findall(content)]
    return max(levels) if levels else None


def read_link_generation(device_dir: str) -> int | None:
    """PCIe generation of the current link speed; None when it is not a known speed."""
    content = drm_cards.read_text(os.path.join(device_dir, "current_link_speed"))
    if not content:
        return None
    try:
        speed = round(float(content.split()[0]), 1)
    except ValueError:
        return None
    return LINK_GENERATIONS.get(speed)


def create_card_device(card: dict, gpu_index: int, source: str) -> dict:
    """Create a device record carrying the identity of a card."""
    device = records.create_device(gpu_index, source)
    device["name"] = card["name"]
    device["pci_bus"] = card["pdev"]
    device["driver_model"] = card["driver"]
    return device


def read_amd_device(card: dict, gpu_index: int) -> dict:
    """Read one telemetry record of an amdgpu card."""
    device_dir = card["device_dir"]
    hwmon_dir = find_hwmon_dir(device_dir)
    device = create_card_device(card, gpu_index, constants.SOURCE_AMD)
    busy = drm_cards.read_number(os.path.join(device_dir, "gpu_busy_percent"))
    mem_busy = drm_cards.read_number(os.path.join(device_dir, "mem_busy_percent"))
    device["util_gpu"] = None if busy is None else float(busy)
    device["util_mem"] = None if mem_busy is None else float(mem_busy)
    device["mem_total"] = drm_cards.read_number(os.path.join(device_dir, "mem_info_vram_total"))
    device["mem_used"] = drm_cards.read_number(os.path.join(device_dir, "mem_info_vram_used"))
    device["temp"] = read_temperature(hwmon_dir)
    fill_amd_power(device, hwmon_dir)
    device["fan"] = read_fan(hwmon_dir)
    device["gfx_clock"] = read_clock(hwmon_dir, CLOCK_LABEL_GFX)
    device["mem_clock"] = read_clock(hwmon_dir, CLOCK_LABEL_MEM)
    device["sm_clock_max"] = read_max_clock(device_dir)
    device["pcie_gen"] = read_link_generation(device_dir)
    device["pcie_width"] = drm_cards.read_number(os.path.join(device_dir, "current_link_width"))
    return device


def read_amd_shared_used(card: dict) -> int | None:
    """Bytes of system RAM the amdgpu card currently uses (GTT), None when unreadable."""
    return drm_cards.read_number(os.path.join(card["device_dir"], "mem_info_gtt_used"))


def read_package_temperature() -> float | None:
    """CPU package temperature in degrees C from the coretemp hwmon device."""
    for name in list_sorted(HWMON_CLASS_DIR):
        hwmon_dir = os.path.join(HWMON_CLASS_DIR, name)
        if drm_cards.read_text(os.path.join(hwmon_dir, "name")) != CORETEMP_NAME:
            continue
        readings = []
        for sensor, label in list_sensor_labels(hwmon_dir, "temp"):
            reading = drm_cards.read_number(os.path.join(hwmon_dir, f"{sensor}_input"))
            if reading is not None and label.startswith(PACKAGE_LABEL):
                readings.append((label, reading))
        for label, reading in readings:
            if label == PACKAGE_LABEL_FIRST:
                return float(reading // MILLI)
        if readings:
            return float(readings[0][1] // MILLI)
    return None


def compute_power(state: dict, path: str, energy: int) -> float | None:
    """Watts from a cumulative microjoule counter; None on the first read, a wrap or a reset."""
    stamp = time.monotonic_ns()
    previous = state["energy"].get(path)
    state["energy"][path] = (energy, stamp)
    if previous is None:
        return None
    energy_delta = energy - previous[0]
    elapsed = stamp - previous[1]
    if energy_delta < 0 or elapsed <= 0:
        return None
    return energy_delta / MICRO / (elapsed / NANO)


def find_rapl_zone() -> str:
    """Return the energy file of the uncore zone of the first package that has one."""
    for package in list_sorted(POWERCAP_DIR):
        if not RAPL_PACKAGE_PATTERN.fullmatch(package):
            continue
        package_dir = os.path.join(POWERCAP_DIR, package)
        for child in list_sorted(package_dir):
            zone_dir = os.path.join(package_dir, child)
            energy_path = os.path.join(zone_dir, "energy_uj")
            if drm_cards.read_text(os.path.join(zone_dir, "name")) != RAPL_ZONE_NAME:
                continue
            if os.path.exists(energy_path):
                return energy_path
    return ""


def read_rapl_energy() -> int | None:
    """Read the RAPL uncore counter in microjoules; a failure is latched process-wide."""
    if RAPL_LATCH["reason"]:
        return None
    if not RAPL_LATCH["searched"]:
        RAPL_LATCH["searched"] = True
        RAPL_LATCH["path"] = find_rapl_zone()
    if not RAPL_LATCH["path"]:
        RAPL_LATCH["reason"] = RAPL_MISSING
        return None
    try:
        with open(RAPL_LATCH["path"], "r", encoding="ascii", errors="replace") as handle:
            content = handle.read()
        return int(content.strip())
    except PermissionError:
        RAPL_LATCH["reason"] = RAPL_DENIED
    except (OSError, ValueError):
        RAPL_LATCH["reason"] = RAPL_MISSING
    return None


def fill_intel_power(device: dict, state: dict, hwmon_dir: str) -> None:
    """Fill power from the hwmon energy counter, else from RAPL on an integrated card."""
    energy_path = os.path.join(hwmon_dir, "energy1_input") if hwmon_dir else ""
    energy = drm_cards.read_number(energy_path) if energy_path else None
    if energy is not None:
        device["power_kind"] = POWER_KIND_BOARD
        device["power_w"] = compute_power(state, energy_path, energy)
        return
    if not state["card"]["integrated"]:
        return
    energy = read_rapl_energy()
    if energy is not None:
        device["power_kind"] = POWER_KIND_RAPL
        device["power_w"] = compute_power(state, RAPL_LATCH["path"], energy)


def fill_intel_frequency(device: dict, gt_dir: str) -> None:
    """Fill actual, requested and maximum GT frequency in MHz."""
    actual = drm_cards.read_number(os.path.join(gt_dir, "rps_act_freq_mhz"))
    requested = drm_cards.read_number(os.path.join(gt_dir, "rps_cur_freq_mhz"))
    device["sm_clock"] = requested if actual is None else actual
    device["sm_clock_requested"] = requested
    device["sm_clock_max"] = drm_cards.read_number(os.path.join(gt_dir, "rps_RP0_freq_mhz"))


def read_throttle(gt_dir: str) -> list[str] | None:
    """Labels of the active throttle reasons; None when no throttle file is readable."""
    readable = 0
    active = []
    for file_name, label in THROTTLE_FILES:
        flag = drm_cards.read_number(os.path.join(gt_dir, file_name))
        if flag is None:
            continue
        readable += 1
        if flag:
            active.append(label)
    return active if readable else None


def read_intel_device(card: dict, gpu_index: int, state: dict) -> dict:
    """Read one telemetry record of an Intel card; memory in use is filled by the caller."""
    hwmon_dir = find_hwmon_dir(card["device_dir"])
    gt_dir = os.path.join(card["card_dir"], "gt", "gt0")
    device = create_card_device(card, gpu_index, constants.SOURCE_INTEL)
    device["mem_total"] = card["vram_total"] or None
    device["temp"] = read_temperature(hwmon_dir)
    if device["temp"] is None and card["integrated"]:
        device["temp"] = read_package_temperature()
        if device["temp"] is not None:
            device["temp_kind"] = TEMP_KIND_CPU
    fill_intel_power(device, state, hwmon_dir)
    device["fan"] = read_fan(hwmon_dir)
    fill_intel_frequency(device, gt_dir)
    device["throttle"] = read_throttle(gt_dir)
    reading = drm_pmu.read_pmu(state["pmu"])
    device["util_gpu"] = reading["util"]
    device["video"] = reading["video"]
    return device


def describe_pmu_reason(reason: str) -> str:
    """Explain why the PMU gives no load figures; "" when it works."""
    if reason == drm_pmu.REASON_DENIED:
        paranoid = drm_cards.read_text(PARANOID_PATH)
        return STATUS_PMU_DENIED.format(value=paranoid if paranoid else "?")
    if reason == drm_pmu.REASON_MISSING:
        return STATUS_PMU_MISSING
    if reason == drm_pmu.REASON_SHARED:
        return STATUS_PMU_SHARED
    return ""


def describe_intel_status(state: dict, device: dict) -> str:
    """Compose the explanation line for missing Intel metrics; "" when nothing is missing."""
    parts = []
    pmu_part = describe_pmu_reason(state["pmu"]["reason"])
    if pmu_part:
        parts.append(pmu_part)
    if state["card"]["integrated"] and device["power_kind"] != POWER_KIND_BOARD:
        if RAPL_LATCH["reason"] == RAPL_DENIED:
            parts.append(STATUS_RAPL_DENIED)
        elif RAPL_LATCH["reason"] == RAPL_MISSING:
            parts.append(STATUS_RAPL_MISSING)
    if not parts:
        return ""
    return STATUS_LINE.format(parts="; ".join(parts))


def main():
    pass


if __name__ == "__main__":
    main()
