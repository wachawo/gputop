#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse the text of one DRM fdinfo file into a client info record."""

# Local imports
from gputop import constants

# Size units of fdinfo values, lower-cased, mapped to bytes
SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000 ** 2,
    "mib": 1024 ** 2,
    "gb": 1000 ** 3,
    "gib": 1024 ** 3,
    "tb": 1000 ** 4,
    "tib": 1024 ** 4,
}

# fdinfo key names and prefixes
KEY_DRIVER = "drm-driver"
KEY_PDEV = "drm-pdev"
KEY_CLIENT_ID = "drm-client-id"
SHARED_PREFIX = "drm-shared-"
ENGINE_PREFIX = "drm-engine-"
CAPACITY_PREFIX = "drm-engine-capacity-"
AMD_SHARED_KEYS = ("drm-shared-vram", "drm-shared-gtt", "drm-shared-cpu")
INTEL_LOCAL_KEYS = ("drm-resident-local0", "drm-resident-stolen-local0")
INTEL_SYSTEM_KEYS = ("drm-resident-system0", "drm-resident-stolen-system0")
INTEL_SHARED_LOCAL = "drm-shared-local0"
INTEL_SHARED_SYSTEM = "drm-shared-system0"
INTEL_LOCAL_MARK = "local0"


def create_client_info(driver: str, pdev: str) -> dict:
    """Create the parsed form of one fdinfo file.

    Keys:
        driver: str, value of drm-driver.
        pdev: str, PCI address from drm-pdev, "" when absent.
        client_id: int or None, drm-client-id, None when absent or unparseable.
        vram: int or None, bytes resident in device memory, None = unknown.
        gtt: int or None, bytes resident in system RAM mapped for the GPU, None = unknown.
        cpu: int or None, bytes of the CPU-only region, None = unknown.
        shared: int or None, bytes in buffers with more than one handle, None = unknown.
        engines: dict[str, int] or None, raw engine name to accumulated busy nanoseconds,
            None = the engine set is unknown.
    """
    return {
        "driver": driver,
        "pdev": pdev,
        "client_id": None,
        "vram": None,
        "gtt": None,
        "cpu": None,
        "shared": None,
        "engines": None,
    }


def is_plain_integer(token: str) -> bool:
    """Tell whether the token is a non-empty run of ASCII digits."""
    return token.isascii() and token.isdigit()


def parse_size(value: str) -> int | None:
    """Convert '<integer> [unit]' to bytes; None when the value is not understood."""
    tokens = value.split()
    if not tokens or len(tokens) > 2:
        return None
    if not is_plain_integer(tokens[0]):
        return None
    unit = tokens[1].lower() if len(tokens) == 2 else ""
    multiplier = SIZE_UNITS.get(unit)
    if multiplier is None:
        return None
    return int(tokens[0]) * multiplier


def parse_fields(text: str) -> dict:
    """Split fdinfo text into a key to value map; a repeated key keeps its last value."""
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def read_region(fields: dict, preferred: str, fallback: str) -> int | None:
    """Read one memory region, using the legacy key only when the preferred one is absent."""
    if preferred in fields:
        return parse_size(fields[preferred])
    if fallback in fields:
        return parse_size(fields[fallback])
    return None


def sum_present_keys(fields: dict, keys: tuple) -> int | None:
    """Add the sizes of the listed keys that exist; None when none exists or one is unparseable."""
    present = [key for key in keys if key in fields]
    if not present:
        return None
    total = 0
    for key in present:
        size = parse_size(fields[key])
        if size is None:
            return None
        total += size
    return total


def sum_shared(fields: dict, required: tuple) -> int | None:
    """Add every drm-shared-* value; None unless all required keys exist and all values parse."""
    shared_keys = [key for key in fields if key.startswith(SHARED_PREFIX)]
    if not shared_keys:
        return None
    for key in required:
        if key not in fields:
            return None
    total = 0
    for key in shared_keys:
        size = parse_size(fields[key])
        if size is None:
            return None
        total += size
    return total


def get_intel_shared_keys(fields: dict, driver: str) -> tuple:
    """Return the drm-shared-* keys an Intel client must report for its sum to be trusted."""
    has_local = any(INTEL_LOCAL_MARK in key for key in fields)
    if driver == constants.DRIVER_I915 and not has_local:
        return (INTEL_SHARED_SYSTEM,)
    return (INTEL_SHARED_LOCAL, INTEL_SHARED_SYSTEM)


def parse_engines(fields: dict, intel: bool) -> dict | None:
    """Collect busy nanoseconds per raw engine name; None when the set is unknown."""
    engines = {}
    for key, value in fields.items():
        if not key.startswith(ENGINE_PREFIX) or key.startswith(CAPACITY_PREFIX):
            continue
        tokens = value.split()
        if not tokens or not is_plain_integer(tokens[0]):
            return None
        engines[key[len(ENGINE_PREFIX):]] = int(tokens[0])
    if intel and not engines:
        return None
    return engines


def fill_amd_memory(info: dict, fields: dict) -> None:
    """Fill the memory figures by the amdgpu key names."""
    info["vram"] = read_region(fields, "drm-resident-vram", "drm-memory-vram")
    info["gtt"] = read_region(fields, "drm-resident-gtt", "drm-memory-gtt")
    info["cpu"] = read_region(fields, "drm-resident-cpu", "drm-memory-cpu")
    info["shared"] = sum_shared(fields, AMD_SHARED_KEYS)


def fill_intel_memory(info: dict, fields: dict) -> None:
    """Fill the memory figures by the i915 region names."""
    info["vram"] = sum_present_keys(fields, INTEL_LOCAL_KEYS)
    info["gtt"] = sum_present_keys(fields, INTEL_SYSTEM_KEYS)
    info["cpu"] = None
    info["shared"] = sum_shared(fields, get_intel_shared_keys(fields, info["driver"]))


def parse_fdinfo(text: str) -> dict | None:
    """Parse one fdinfo text; None when the descriptor is not a DRM client."""
    fields = parse_fields(text)
    if KEY_DRIVER not in fields:
        return None
    info = create_client_info(fields[KEY_DRIVER], fields.get(KEY_PDEV, ""))
    client_id = fields.get(KEY_CLIENT_ID, "")
    if is_plain_integer(client_id):
        info["client_id"] = int(client_id)
    intel = info["driver"] in (constants.DRIVER_I915, constants.DRIVER_XE)
    if intel:
        fill_intel_memory(info, fields)
    else:
        fill_amd_memory(info, fields)
    info["engines"] = parse_engines(fields, intel)
    return info


def main():
    pass


if __name__ == "__main__":
    main()
