#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparkline and stacked-bar drawing primitives of the gputop TUI."""

from collections.abc import Iterable

from rich.text import Text

# Local imports
from gputop import constants

# Glyphs
SPARK_GLYPHS = "▁▂▃▄▅▆▇█"
SPARK_GAP = " "
BAR_FILLED = "█"
BAR_EMPTY = "░"
LEGEND_SQUARE = "■ "

# Scales
DEFAULT_TOP = 100.0


def clamp_number(value: float, low: float, high: float) -> float:
    """Limit a number to the closed range low..high."""
    return max(low, min(high, value))


def get_spark_level(value: float, top: float = DEFAULT_TOP) -> int:
    """Return the glyph index 0..7 of a value on a scale that ends at top."""
    if top <= 0:
        return 0
    level = int(value / top * 7 + 0.5)
    return int(clamp_number(level, 0, len(SPARK_GLYPHS) - 1))


def get_spark_glyph(value: float | None, top: float = DEFAULT_TOP) -> str:
    """Return the glyph of one sample; a missing sample is a gap."""
    if value is None:
        return SPARK_GAP
    return SPARK_GLYPHS[get_spark_level(value, top)]


def draw_sparkline(
    history: Iterable[float | None], width: int, top: float = DEFAULT_TOP, style: str = ""
) -> Text:
    """Draw the last width samples of a history, oldest left; shorter histories stay shorter."""
    samples = list(history)
    visible = samples[-width:] if width > 0 else []
    glyphs = [get_spark_glyph(sample, top) for sample in visible]
    return Text("".join(glyphs), style=style)


def count_bar_cells(value: float, total: float | None, width: int, used: int) -> int:
    """Cells one part takes: rounded share of the width, limited to the cells still free."""
    if not total or total <= 0 or value <= 0:
        return 0
    cells = round(value / total * width)
    return int(clamp_number(cells, 0, width - used))


def draw_stacked_bar(parts: list[tuple[float, str]], total: float | None, width: int) -> Text:
    """Draw ordered (value, style) parts over a total; the unused tail is the free portion."""
    bar = Text()
    used = 0
    for value, style in parts:
        cells = count_bar_cells(value, total, width, used)
        if cells <= 0:
            continue
        bar.append(BAR_FILLED * cells, style=style)
        used += cells
    if width > used:
        bar.append(BAR_EMPTY * (width - used), style=constants.FREE_STYLE)
    return bar


def main():
    pass


if __name__ == "__main__":
    main()
