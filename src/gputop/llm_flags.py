#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch-flag parser for llama.cpp style runtimes."""

# Local imports
from gputop import cmdline

# Field kinds
FIELD_INT = "int"
FIELD_LAYERS = "layers"
FIELD_TEXT = "text"

# Flag spelling to (field name, field kind)
VALUE_FLAGS = {
    "-c": ("ctx", FIELD_INT),
    "--ctx-size": ("ctx", FIELD_INT),
    "--n-ctx": ("ctx", FIELD_INT),
    "-ngl": ("ngl", FIELD_LAYERS),
    "--n-gpu-layers": ("ngl", FIELD_LAYERS),
    "--gpu-layers": ("ngl", FIELD_LAYERS),
    "-np": ("parallel", FIELD_INT),
    "--parallel": ("parallel", FIELD_INT),
    "-ctk": ("ctk", FIELD_TEXT),
    "--cache-type-k": ("ctk", FIELD_TEXT),
    "-ctv": ("ctv", FIELD_TEXT),
    "--cache-type-v": ("ctv", FIELD_TEXT),
    "--port": ("port", FIELD_INT),
    "--log-file": ("log_file", FIELD_TEXT),
}
SWITCH_FLAGS = {"--no-mmap": "no_mmap"}
ALL_LAYERS_WORDS = ("all", "auto")
ALL_LAYERS = -1


def convert_flag_value(value: str, field_kind: str):
    """Typed value of a flag, None when an integer field does not parse."""
    if field_kind == FIELD_TEXT:
        return value
    text = value.strip()
    if field_kind == FIELD_LAYERS and text.lower() in ALL_LAYERS_WORDS:
        return ALL_LAYERS
    try:
        return int(text)
    except ValueError:
        return None


def parse_llm_flags(arguments: list[str]) -> dict:
    """Parse the launch flags that matter for the memory breakdown.

    Returns only the fields found: ctx, ngl, parallel, port (int), ctk, ctv, log_file (str),
    no_mmap (True). The last occurrence of a flag wins; an integer that does not parse is dropped.
    Long flags are read with underscores as hyphens; the value of a secret flag is skipped.
    """
    flags = {}
    position = 0
    while position < len(arguments):
        flag, value = cmdline.split_flag(arguments[position])
        flag = cmdline.normalize_flag(flag)
        position += 1
        if cmdline.is_secret_flag(flag):
            if value is None:
                position += 1
            continue
        if flag in SWITCH_FLAGS:
            flags[SWITCH_FLAGS[flag]] = True
            continue
        if flag not in VALUE_FLAGS:
            continue
        if value is None:
            if position >= len(arguments):
                continue
            value = arguments[position]
            position += 1
        field, field_kind = VALUE_FLAGS[flag]
        converted = convert_flag_value(value, field_kind)
        if converted is not None:
            flags[field] = converted
    return flags


def main():
    pass


if __name__ == "__main__":
    main()
