#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-vendor adapter enumeration, display order and --gpu selection."""
import logging

# Local imports
from gputop import constants
from gputop import drm
from gputop import nvidia
from gputop import records

logger = logging.getLogger(__name__)


def enumerate_adapters() -> list[dict]:
    """Return NVML adapters followed by the DRM adapters whose key is not listed yet."""
    found = list(nvidia.list_nvidia_adapters())
    known_keys = set()
    for adapter in found:
        known_keys.add(adapter["key"])
    for adapter in drm.list_drm_adapters():
        if adapter["key"] in known_keys:
            logger.debug(f"DRM adapter {adapter['name']} already listed by NVML")
            continue
        known_keys.add(adapter["key"])
        found.append(adapter)
    return found


def get_order_key(adapter: dict) -> tuple:
    """Sort key of the display order: NVIDIA first, larger memory first, smaller key first."""
    is_other_vendor = adapter["vendor_id"] != constants.VENDOR_NVIDIA
    return (is_other_vendor, -adapter["dedicated_vram"], adapter["key"])


def order_adapters(found: list[dict]) -> list[dict]:
    """Return a new list in display order; the input list is left untouched."""
    return sorted(found, key=get_order_key)


def list_ordered_adapters() -> list[dict]:
    """Enumerate every adapter and return the list in display order."""
    return order_adapters(enumerate_adapters())


def select_adapter(ordered: list[dict], gpu_index: int) -> dict | None:
    """Return the adapter at gpu_index of the ordered list.

    None only for an empty list with index 0. Raises ValueError for a negative index and for
    an index past the end of the list.
    """
    if gpu_index < 0:
        raise ValueError(constants.GPU_INDEX_MESSAGE)
    if gpu_index < len(ordered):
        return ordered[gpu_index]
    if not ordered and gpu_index == 0:
        return None
    raise ValueError(
        f"--gpu {gpu_index} is out of range; {len(ordered)} GPU adapter(s) available"
    )


def list_default_indexes(ordered: list[dict]) -> list[int]:
    """Indexes the live view shows without --gpu: the discrete adapters, else every adapter.

    An integrated GPU next to a discrete one is left out: without root it has almost nothing
    to report. A machine with integrated GPUs only shows them all; no adapters gives [0].
    """
    if not ordered:
        return [0]
    discrete = [index for index, adapter in enumerate(ordered) if not adapter["integrated"]]
    return discrete or list(range(len(ordered)))


def format_adapter_line(index: int, adapter: dict, is_hidden: bool) -> str:
    """Return one line of the --list-gpus output."""
    vendor = records.get_vendor_name(adapter["vendor_id"])
    line = f"{index}: {adapter['name']} ({vendor}, key 0x{adapter['key']:016x})"
    if is_hidden:
        line += f" - integrated, hidden in the live view without --gpu {constants.GPU_ALL}"
    return line


def main():
    pass


if __name__ == "__main__":
    main()
