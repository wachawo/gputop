#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Facade of the DRM backend: adapter list and open, sample, close for one AMD or Intel card."""

import logging
import os
import traceback

# Local imports
from gputop import drm_cards
from gputop import drm_pmu
from gputop import drm_scan
from gputop import drm_telemetry
from gputop import records

DRIVER_PREFIX = "kernel "

logger = logging.getLogger(__name__)


def create_drm_state(card: dict, gpu_index: int, scan_budget: float | None) -> dict:
    """Create the sampler state of one DRM card.

    Keys:
        card: dict, card record of drm_cards (its integrated flag can turn false later).
        gpu_index: int, display index reported in the device record.
        scan_budget: float or None, wall-time budget of one process scan in seconds,
            None = unlimited.
        scan: dict, process scan state of drm_scan.
        intel: dict or None, Intel telemetry state of drm_telemetry, None for an AMD card
            or after close.
    """
    return {
        "card": card,
        "gpu_index": gpu_index,
        "scan_budget": scan_budget,
        "scan": drm_scan.create_scan_state(),
        "intel": None,
    }


def log_failure(exc: BaseException) -> None:
    """Log an unexpected exception caught at the backend boundary."""
    logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")


def list_drm_adapters() -> list[dict]:
    """Return one adapter per supported DRM card, discrete first, then VRAM descending."""
    try:
        cards = drm_cards.list_cards()
        return [drm_cards.build_card_adapter(card) for card in cards]
    except Exception as exc:
        log_failure(exc)
        return []


def find_card(cards: list[dict], adapter: dict) -> dict | None:
    """Return the card whose key and vendor id equal the adapter's."""
    for card in cards:
        if card["key"] == adapter["key"] and card["vendor_id"] == adapter["vendor_id"]:
            return card
    return None


def arm_scan(state: dict) -> None:
    """Take the unbudgeted first scan that arms the engine baselines."""
    card = state["card"]
    drm_scan.scan_processes(state["scan"], [card], None)
    if card["intel"] and drm_scan.sum_client_memory(state["scan"], card)["has_local"]:
        card["integrated"] = False


def arm_intel(state: dict, cards: list[dict]) -> None:
    """Open the Intel PMU and arm the PMU and energy baselines with a first reading."""
    card = state["card"]
    if not card["intel"]:
        return
    state["intel"] = drm_telemetry.create_intel_state(card, cards)
    drm_telemetry.read_intel_device(card, state["gpu_index"], state["intel"])


def open_drm(adapter: dict, gpu_index: int, scan_budget: float | None) -> dict | None:
    """Open the sampler state of the adapter's card; None when no card matches."""
    try:
        cards = drm_cards.list_cards()
        card = find_card(cards, adapter)
    except Exception as exc:
        log_failure(exc)
        return None
    if card is None:
        return None
    state = create_drm_state(card, gpu_index, scan_budget)
    try:
        arm_scan(state)
    except Exception as exc:
        log_failure(exc)
    try:
        arm_intel(state, cards)
    except Exception as exc:
        log_failure(exc)
    return state


def fill_amd_sample(sample: dict, state: dict) -> None:
    """Add device memory and telemetry of an amdgpu card to the sample."""
    device = drm_telemetry.read_amd_device(state["card"], state["gpu_index"])
    sample["device"] = device
    sample["vram_used"] = device["mem_used"]
    sample["shared_used"] = drm_telemetry.read_amd_shared_used(state["card"])


def fill_intel_sample(sample: dict, state: dict, used: int | None) -> None:
    """Add device memory, telemetry and the metrics explanation of an Intel card."""
    sample["vram_used"] = used
    if state["intel"] is None:
        return
    device = drm_telemetry.read_intel_device(state["card"], state["gpu_index"], state["intel"])
    device["mem_used"] = used
    sample["device"] = device
    sample["counter_status"] = drm_telemetry.describe_intel_status(state["intel"], device)


def fill_sample(sample: dict, state: dict) -> None:
    """Run one scan and fill every part of the backend sample."""
    card = state["card"]
    scan = state["scan"]
    drm_scan.scan_processes(scan, [card], state["scan_budget"])
    sample["process_status"] = scan["status"]
    used = None
    if card["intel"]:
        memory = drm_scan.sum_client_memory(scan, card)
        if memory["has_local"]:
            card["integrated"] = False
        used = memory["used"]
    sample["shared_is_spill"] = not card["integrated"]
    sample["procs"] = drm_scan.build_proc_counters(scan, card)
    if card["intel"]:
        fill_intel_sample(sample, state, used)
    else:
        fill_amd_sample(sample, state)


def sample_drm(state: dict) -> dict:
    """Take one backend sample of the card. Never raises."""
    sample = records.create_backend_sample()
    sample["driver"] = DRIVER_PREFIX + os.uname().release
    sample["shared_is_spill"] = not state["card"]["integrated"]
    try:
        fill_sample(sample, state)
    except Exception as exc:
        log_failure(exc)
    return sample


def close_drm(state: dict) -> None:
    """Close the perf descriptors of the state. Idempotent, never raises."""
    intel_state = state["intel"]
    state["intel"] = None
    if intel_state is not None:
        drm_pmu.close_pmu(intel_state["pmu"])


def main():
    pass


if __name__ == "__main__":
    main()
