#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load-log tail parser for the buffer sizes llama.cpp and Ollama report at model load."""

import os
import re

# Local imports
from gputop import constants

# Which part of the file is read
OLLAMA_LOG_PATH = "~/.ollama/logs/server.log"
TAIL_BYTES = 512 * 1024

# Lines that open a load block (one per device)
BLOCK_MARKERS = ("model buffer size", 'msg="model weights"')

# Line shapes
LLAMA_LINE_PATTERN = re.compile(
    r"(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*(\d+(?:\.\d+)?)\s*MiB"
)
OLLAMA_LINE_PATTERN = re.compile(
    r'msg="(model weights|kv cache|compute graph)".*?\bdevice=(\S+)'
    r'.*?\bsize="(\d+(?:\.\d+)?)\s*(GiB|MiB|KiB|B)"'
)
UNIT_BYTES = {"GiB": constants.GIB, "MiB": constants.MIB, "KiB": 1024, "B": 1}
CATEGORY_NAMES = {
    "model": constants.SEGMENT_WEIGHTS,
    "model weights": constants.SEGMENT_WEIGHTS,
    "KV": constants.SEGMENT_KV,
    "RS": constants.SEGMENT_KV,
    "kv cache": constants.SEGMENT_KV,
    "compute": constants.SEGMENT_COMPUTE,
    "output": constants.SEGMENT_COMPUTE,
    "compute graph": constants.SEGMENT_COMPUTE,
}


def get_ollama_log_path() -> str:
    """Path of the log an Ollama server writes when it is not run by systemd."""
    return os.path.expanduser(OLLAMA_LOG_PATH)


def is_gpu_device(name: str) -> bool:
    """A log device is a GPU unless its name marks host memory."""
    upper_name = name.upper()
    if upper_name.startswith("CPU") or upper_name.startswith("MMAP"):
        return False
    return "HOST" not in upper_name


def read_log_tail(path: str, min_mtime: float) -> list[str]:
    """Lines of the last part of the log; empty when the file is stale or unreadable."""
    try:
        status = os.stat(path)
        if status.st_mtime < min_mtime:
            return []
        with open(path, "rb") as handle:
            handle.seek(max(status.st_size - TAIL_BYTES, 0))
            tail_bytes = handle.read(TAIL_BYTES)
    except OSError:
        return []
    text = tail_bytes.decode("utf-8", errors="replace")
    return text.splitlines()


def has_block_marker(line: str) -> bool:
    """True for a line that reports the weights buffer of one device."""
    for marker in BLOCK_MARKERS:
        if marker in line:
            return True
    return False


def find_block_start(lines: list[str]) -> int | None:
    """Index of the first line of the most recent load block, None without one."""
    start = None
    for index in range(len(lines) - 1, -1, -1):
        if has_block_marker(lines[index]):
            start = index
            break
    if start is None:
        return None
    while start > 0 and has_block_marker(lines[start - 1]):
        start -= 1
    return start


def parse_log_line(line: str) -> tuple[str, str, int] | None:
    """(device, category, bytes) of one buffer line, None for any other line."""
    llama_match = LLAMA_LINE_PATTERN.search(line)
    if llama_match:
        device_text = llama_match.group(1)
        device = device_text.rstrip(":")
        size = int(float(llama_match.group(3)) * constants.MIB)
        return device, CATEGORY_NAMES[llama_match.group(2)], size
    ollama_match = OLLAMA_LINE_PATTERN.search(line)
    if ollama_match:
        device_text = ollama_match.group(2)
        device = device_text.strip('"')
        size = int(float(ollama_match.group(3)) * UNIT_BYTES[ollama_match.group(4)])
        return device, CATEGORY_NAMES[ollama_match.group(1)], size
    return None


def parse_load_log(path: str, min_mtime: float) -> dict:
    """Buffer sizes of the most recent model load found in the log tail.

    Returns device name to {"weights", "kv", "compute"} bytes, repeated lines summed. Empty when
    the file is older than min_mtime, cannot be read, or holds no load block.
    """
    lines = read_log_tail(path, min_mtime)
    start = find_block_start(lines)
    if start is None:
        return {}
    devices = {}
    for line in lines[start:]:
        parsed = parse_log_line(line)
        if parsed is None:
            continue
        device, category, size = parsed
        if not device:
            continue
        sizes = devices.get(device)
        if sizes is None:
            sizes = {
                constants.SEGMENT_WEIGHTS: 0,
                constants.SEGMENT_KV: 0,
                constants.SEGMENT_COMPUTE: 0,
            }
            devices[device] = sizes
        sizes[category] += size
    return devices


def main():
    pass


if __name__ == "__main__":
    main()
