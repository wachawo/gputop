#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Number-to-text helpers shared by the report, the diff, the TUI and the notes."""

# Local imports
from gputop import constants

MISSING_SIZE = "N/A"
MISSING_METRIC = "N/A"
# Placeholder of a per-process value that is not measured
MISSING_CELL = "n/a"


def format_size(value: float) -> str:
    """Format a byte count as binary gigabytes (two decimals) or megabytes (no decimals)."""
    if abs(value) >= constants.GIB:
        return f"{value / constants.GIB:.2f} GB"
    return f"{value / constants.MIB:.0f} MB"


def format_size_or_na(value: float | None, missing: str = MISSING_SIZE) -> str:
    """Format a byte count, or return the placeholder when the value is unknown."""
    if value is None:
        return missing
    return format_size(value)


def format_metric(value: float | None, spec: str = "", unit: str = "") -> str:
    """Format an optional reading with a format spec and a unit; unknown prints without unit."""
    if value is None:
        return MISSING_METRIC
    return format(value, spec) + unit


def convert_to_int(value) -> int | None:
    """Integer form of a parsed (JSON or file metadata) number, None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return int(value)
    except (ValueError, OverflowError):
        return None


def format_megabytes_per_second(kilobytes: int | None) -> str:
    """A KB/s counter in MB/s without decimals, or the placeholder when unknown."""
    if kilobytes is None:
        return MISSING_METRIC
    return format(kilobytes / 1024, ".0f")


def format_error(exc: BaseException) -> str:
    """Return the one spelling of a stored error text: exception class name and message."""
    return f"{type(exc).__name__}: {exc}"


def main():
    pass


if __name__ == "__main__":
    main()
