#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover amdgpu, i915 and xe cards in sysfs and describe them as card records."""

import os
import re

# Local imports
from gputop import constants
from gputop import records

# sysfs locations
SYS_DRM_DIR = "/sys/class/drm"
PCI_IDS_PATHS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")
CARD_PATTERN = re.compile(r"card\d+")

# Classification thresholds
MIN_LOCAL_MEMORY = constants.GIB
UNIFIED_VIDEO_MAJOR = 4
BAR_FILES = ("resource2_wc", "resource2")

# Parsed pci.ids files, path to {(vendor id, device id): name}; each file is read once
PCI_IDS_CACHE = {}


def create_card(entry: str, driver: str) -> dict:
    """Create the record of one DRM card.

    Keys:
        entry: str, sysfs entry name, for example "card0".
        card_dir: str, /sys/class/drm/<entry>.
        device_dir: str, PCI device directory <card_dir>/device.
        driver: str, kernel driver name (amdgpu, i915 or xe).
        pdev: str, PCI address dddd:bb:dd.f.
        vendor_id: int, PCI vendor id, 0 = unknown.
        device_id: int, PCI device id, 0 = unknown.
        key: int, adapter key derived from the PCI address.
        name: str, marketing name with the vendor prefix.
        intel: bool, the card takes the Intel path (vendor 0x8086 on i915 or xe).
        integrated: bool, the card has no memory of its own; an Intel card can turn
            discrete later, when a client reports local memory.
        vram_total: int, bytes of device memory, 0 = unknown.
        gtt_total: int, bytes of GPU-accessible system RAM (AMD), 0 otherwise.
        unified_video: bool, AMD decode and encode share one ring (VCN 4 and later).
    """
    card_dir = os.path.join(SYS_DRM_DIR, entry)
    return {
        "entry": entry,
        "card_dir": card_dir,
        "device_dir": os.path.join(card_dir, "device"),
        "driver": driver,
        "pdev": "",
        "vendor_id": 0,
        "device_id": 0,
        "key": 0,
        "name": "",
        "intel": False,
        "integrated": False,
        "vram_total": 0,
        "gtt_total": 0,
        "unified_video": False,
    }


def read_text(path: str) -> str | None:
    """Read a small text file, stripped; None when it cannot be read."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return None
    return content.strip()


def read_number(path: str, base: int = 10) -> int | None:
    """Read an integer file in the given base; None when unreadable or unparseable."""
    content = read_text(path)
    if not content:
        return None
    try:
        return int(content, base)
    except ValueError:
        return None


def list_card_entries() -> list[str]:
    """Return the sorted cardN entry names of /sys/class/drm."""
    try:
        names = os.listdir(SYS_DRM_DIR)
    except OSError:
        return []
    return sorted(name for name in names if CARD_PATTERN.fullmatch(name))


def read_uevent(device_dir: str) -> dict:
    """Parse the uevent file of a device into a key to value map."""
    content = read_text(os.path.join(device_dir, "uevent"))
    values = {}
    if content is None:
        return values
    for line in content.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_driver(device_dir: str, uevent: dict) -> str:
    """Return the bound kernel driver name, "" when it cannot be found."""
    try:
        target = os.readlink(os.path.join(device_dir, "driver"))
    except OSError:
        return uevent.get("DRIVER", "")
    return os.path.basename(target)


def read_pdev(device_dir: str, uevent: dict) -> str:
    """Return the PCI address of the device."""
    slot = uevent.get("PCI_SLOT_NAME", "")
    if slot:
        return slot
    return os.path.basename(os.path.realpath(device_dir))


def parse_pci_ids(content: str) -> dict:
    """Build {(vendor id, device id): name} from the text of a pci.ids file."""
    names = {}
    vendor_id = None
    for line in content.splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("\t\t"):
            continue
        tokens = line.split(None, 1)
        try:
            number = int(tokens[0], 16)
        except ValueError:
            number = None
        if not line.startswith("\t"):
            vendor_id = number if len(tokens[0]) == 4 else None
        elif vendor_id is not None and number is not None and len(tokens) == 2:
            names[(vendor_id, number)] = tokens[1].strip()
    return names


def load_pci_ids(path: str) -> dict:
    """Return the parsed pci.ids file of the path, reading it at most once."""
    if path not in PCI_IDS_CACHE:
        content = read_text(path)
        PCI_IDS_CACHE[path] = parse_pci_ids(content) if content else {}
    return PCI_IDS_CACHE[path]


def shorten_pci_name(name: str) -> str:
    """Keep only the bracketed marketing part of a pci.ids device name when there is one."""
    opening = name.find("[")
    closing = name.rfind("]")
    if opening == -1 or closing == -1:
        return name
    inner = name[opening + 1:closing].strip()
    return inner if inner else name


def find_pci_name(vendor_id: int, device_id: int, pci_ids_path: str | None) -> str:
    """Look a device up in pci.ids; "" when no file knows it."""
    paths = (pci_ids_path,) if pci_ids_path else PCI_IDS_PATHS
    for path in paths:
        known_names = load_pci_ids(path)
        name = known_names.get((vendor_id, device_id))
        if name:
            return shorten_pci_name(name)
    return ""


def resolve_card_name(card: dict, pci_ids_path: str | None) -> str:
    """Return the display name of a card: product name, pci.ids entry or a generic label."""
    product = read_text(os.path.join(card["device_dir"], "product_name"))
    if product:
        return product
    prefix = records.get_vendor_name(card["vendor_id"])
    name = find_pci_name(card["vendor_id"], card["device_id"], pci_ids_path)
    if name:
        return f"{prefix} {name}"
    return f"{prefix} GPU {card['device_id']:04x}"


def has_unified_video(device_dir: str) -> bool:
    """Tell whether the amdgpu video block is VCN 4 or later."""
    major = read_number(os.path.join(device_dir, "ip_discovery", "die", "0", "UVD", "0", "major"))
    return major is not None and major >= UNIFIED_VIDEO_MAJOR


def fill_amd_memory(card: dict) -> None:
    """Fill memory totals and the integrated flag of an amdgpu card."""
    device_dir = card["device_dir"]
    card["vram_total"] = read_number(os.path.join(device_dir, "mem_info_vram_total")) or 0
    card["gtt_total"] = read_number(os.path.join(device_dir, "mem_info_gtt_total")) or 0
    card["integrated"] = not os.path.exists(os.path.join(device_dir, "mem_info_vram_vendor"))
    card["unified_video"] = has_unified_video(device_dir)


def read_bar_size(device_dir: str) -> int:
    """Return the size of the PCI BAR 2 resource file, 0 when there is none."""
    for file_name in BAR_FILES:
        try:
            size = os.stat(os.path.join(device_dir, file_name)).st_size
        except OSError:
            continue
        if size > 0:
            return size
    return 0


def fill_intel_memory(card: dict) -> None:
    """Fill the estimated local memory and the integrated flag of an Intel card."""
    total = read_number(os.path.join(card["device_dir"], "mem_info_vram_total")) or 0
    if total <= 0:
        bar_size = read_bar_size(card["device_dir"])
        total = bar_size if bar_size >= MIN_LOCAL_MEMORY else 0
    card["vram_total"] = total
    card["gtt_total"] = 0
    card["integrated"] = total == 0


def build_card(entry: str, pci_ids_path: str | None) -> dict | None:
    """Describe one cardN entry; None when its driver is not supported."""
    device_dir = os.path.join(SYS_DRM_DIR, entry, "device")
    uevent = read_uevent(device_dir)
    driver = read_driver(device_dir, uevent)
    if driver not in constants.DRM_DRIVERS:
        return None
    card = create_card(entry, driver)
    card["pdev"] = read_pdev(device_dir, uevent)
    card["vendor_id"] = read_number(os.path.join(device_dir, "vendor"), 16) or 0
    card["device_id"] = read_number(os.path.join(device_dir, "device"), 16) or 0
    card["key"] = records.compute_adapter_key(card["pdev"])
    intel_driver = driver != constants.DRIVER_AMDGPU
    card["intel"] = card["vendor_id"] == constants.VENDOR_INTEL and intel_driver
    card["name"] = resolve_card_name(card, pci_ids_path)
    if card["intel"]:
        fill_intel_memory(card)
    else:
        fill_amd_memory(card)
    return card


def get_card_order(card: dict) -> tuple:
    """Sort key: discrete cards first, then device memory descending."""
    return (card["integrated"], -card["vram_total"])


def list_cards(pci_ids_path: str | None = None) -> list[dict]:
    """Return every supported card, one per PCI device, in backend display order."""
    cards = []
    seen_pdevs = set()
    for entry in list_card_entries():
        card = build_card(entry, pci_ids_path)
        if card is None or card["pdev"] in seen_pdevs:
            continue
        seen_pdevs.add(card["pdev"])
        cards.append(card)
    return sorted(cards, key=get_card_order)


def build_card_adapter(card: dict) -> dict:
    """Convert a card record into an adapter record."""
    return records.create_adapter(
        card["key"],
        card["name"],
        card["vendor_id"],
        card["vram_total"],
        card["gtt_total"],
        card["integrated"],
    )


def main():
    pass


if __name__ == "__main__":
    main()
