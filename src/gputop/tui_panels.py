#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Device panel, system panel, charts grid and scale note renderers of the gputop TUI."""

from collections import deque

from rich.text import Text

# Local imports
from gputop import constants
from gputop import formatting
from gputop import records
from gputop import system
from gputop import tui_draw

# Charts: order is also the cycle order and the option order
CHART_ALL = -1
CHART_CLOCK_KEY = "clock"
CHART_POWER_KEY = "power"
CHARTS = (
    {"key": "util_gpu", "name": "GPU", "colour": "#4ade80", "top": 100.0, "unit": "%"},
    {"key": CHART_CLOCK_KEY, "name": "SM clock", "colour": "#fb923c", "top": None, "unit": " MHz"},
    {"key": "vram", "name": "VRAM", "colour": "#60a5fa", "top": 100.0, "unit": "%"},
    {"key": "util_mem", "name": "Mem activity", "colour": "#a78bfa", "top": 100.0, "unit": "%"},
    {"key": "temp", "name": "Temperature", "colour": "#ef4444", "top": 120.0, "unit": "°C"},
    {"key": CHART_POWER_KEY, "name": "Power", "colour": "#facc15", "top": None, "unit": " W"},
    {"key": "enc", "name": "Encoder", "colour": "#e2e8f0", "top": 100.0, "unit": "%"},
    {"key": "dec", "name": "Decoder", "colour": "#22d3ee", "top": 100.0, "unit": "%"},
)
CHART_ALL_LABEL = "All charts"
CLOCK_NAME_NVML = "SM clock"
CLOCK_NAME_AMD = "GFX clock"
CLOCK_NAME_INTEL = "GT clock"
CLOCK_NAME_MIXED = "Clock"
CHART_NAME_WIDTH = 12
CHART_VALUE_WIDTH = 8
CHART_FIXED_WIDTH = 22
CHART_MIN_COLUMN_WIDTH = 46
CHART_MAX_COLUMNS = 3
CHART_COLUMN_GAP = 2
CHART_WIDTH_MARGIN = 6

# Device panel
DEVICE_PLACEHOLDER = "Sampling GPU counters..."
DEVICE_TITLE_FULL = "Full diagnostics"
DEVICE_TITLE_SLIM = "Slim monitoring"
BAR_WIDTH_MARGIN = 40
BAR_WIDTH_MIN = 20
BAR_WIDTH_MAX = 90
LEGEND_INDENT = "     "
LEGEND_MAX_KINDS = 7
LEGEND_MIN_UNATTRIBUTED = 32 * constants.MIB
LEGEND_MIN_SHARED = 64 * constants.MIB
SENSOR_STYLE = "grey70"
LOAD_STYLE = "bold #4ade80"
MEM_ACTIVITY_STYLE = "#a78bfa"
PAUSED_LABEL = "   [PAUSED]"

# System panel
SYSTEM_PLACEHOLDER = "Sampling RAM and CPU..."
SYSTEM_TITLE = "System (OS-reported)"
SYSTEM_WIDTH_MARGIN = 12
SYSTEM_MIN_BAR = 8
SYSTEM_MAX_BAR = 60
SYSTEM_RAM_STYLE = "dodger_blue2"
SYSTEM_CPU_STYLE = "green3"
SYSTEM_CAPTION_STYLE = "grey50"
SYSTEM_TOP_COUNT = 5
SYSTEM_MIN_CPU_PCT = 0.5
THREAD_HOT_PCT = 90.0
THREAD_WARM_PCT = 60.0


def create_history() -> dict:
    """Create the eight ring buffers of one GPU: chart key to a deque of float or None."""
    return {chart["key"]: deque(maxlen=constants.HISTORY_LENGTH) for chart in CHARTS}


def create_gpu_block(index: int, adapter: dict | None, heading: bool) -> dict:
    """Create the UI record of one shown GPU.

    Keys: index (int, position in the ordered adapter list), adapter (adapter or None),
    heading (bool, draw a "GPU <index>" line above the block), snapshot (snapshot or None,
    the last delivered one), error (str, text of the last failed refresh, "" when fine),
    history (dict chart key to deque, see create_history).
    """
    return {
        "index": index,
        "adapter": adapter,
        "heading": heading,
        "snapshot": None,
        "error": "",
        "history": create_history(),
    }


def compute_vram_percent(snapshot: dict) -> float | None:
    """Device memory in use as a percentage of the total; None when either is unknown."""
    used = snapshot["vram_used"]
    total = snapshot["vram_total"]
    if used is None or not total:
        return None
    return 100.0 * used / total


def compute_chart_values(snapshot: dict | None) -> dict:
    """Return chart key to the value one refresh adds to the history (None = no reading)."""
    values = {chart["key"]: None for chart in CHARTS}
    if snapshot is None:
        return values
    values["vram"] = compute_vram_percent(snapshot)
    device = snapshot["device"]
    if device is None:
        return values
    for key in ("util_gpu", "util_mem", "temp", "enc", "dec"):
        values[key] = device[key]
    values[CHART_CLOCK_KEY] = records.get_main_clock(device)[1]
    values[CHART_POWER_KEY] = device["power_w"]
    return values


def append_history(history: dict, snapshot: dict | None) -> None:
    """Append one sample to each of the eight histories; a failed refresh appends gaps."""
    values = compute_chart_values(snapshot)
    for key, value in values.items():
        history[key].append(value)


def get_clock_name(adapter: dict | None) -> str:
    """Name of chart 1 for one GPU: it follows the vendor of the adapter."""
    if adapter is None:
        return CLOCK_NAME_NVML
    if adapter["vendor_id"] == constants.VENDOR_AMD:
        return CLOCK_NAME_AMD
    if adapter["vendor_id"] == constants.VENDOR_INTEL:
        return CLOCK_NAME_INTEL
    return CLOCK_NAME_NVML


def get_clock_option_name(shown_adapters: list) -> str:
    """Dropdown name of chart 1: the common name when every shown GPU agrees."""
    names = {get_clock_name(adapter) for adapter in shown_adapters}
    if len(names) == 1:
        return names.pop()
    return CLOCK_NAME_MIXED


def list_chart_options(shown_adapters: list) -> list[tuple[str, int]]:
    """Options of the chart selector: all charts first, then one per chart."""
    options = [(CHART_ALL_LABEL, CHART_ALL)]
    for position, chart in enumerate(CHARTS):
        name = chart["name"]
        if chart["key"] == CHART_CLOCK_KEY:
            name = get_clock_option_name(shown_adapters)
        options.append((name, position))
    return options


def get_next_chart_mode(mode: int) -> int:
    """Advance the chart mode: all charts, then each chart, then wrap."""
    following = mode + 1
    if following >= len(CHARTS):
        return CHART_ALL
    return following


def get_history_peak(samples) -> float:
    """Largest known value of a history, 0 when it holds no reading."""
    known = [sample for sample in samples if sample is not None]
    return max(known) if known else 0.0


def get_chart_top(chart: dict, block: dict) -> float:
    """Scale top of one chart: fixed, or adaptive for the clock and power charts."""
    if chart["top"] is not None:
        return chart["top"]
    snapshot = block["snapshot"]
    device = snapshot["device"] if snapshot is not None else None
    limit = 0.0
    if device is not None:
        limit_key = "sm_clock_max" if chart["key"] == CHART_CLOCK_KEY else "power_limit_w"
        limit = device[limit_key] or 0.0
    return max(limit, get_history_peak(block["history"][chart["key"]]), 1.0)


def get_chart_name(chart: dict, block: dict) -> str:
    """Display name of one chart for one GPU."""
    if chart["key"] == CHART_CLOCK_KEY:
        return get_clock_name(block["adapter"])
    return chart["name"]


def render_chart_cell(chart: dict, block: dict, cell_width: int) -> Text:
    """One grid cell: name, current value and sparkline, cropped to the cell width."""
    samples = block["history"][chart["key"]]
    last = samples[-1] if samples else None
    colour = chart["colour"]
    cell = Text(no_wrap=True, overflow="crop")
    chart_name = get_chart_name(chart, block)
    cell.append(chart_name.ljust(CHART_NAME_WIDTH) + " ", style=colour)
    value_text = formatting.format_metric(last, ".0f", chart["unit"])
    cell.append(value_text.rjust(CHART_VALUE_WIDTH), style="bold")
    cell.append(" ")
    spark_width = max(1, cell_width - CHART_FIXED_WIDTH)
    cell.append_text(tui_draw.draw_sparkline(samples, spark_width, get_chart_top(chart, block), colour))
    cell.truncate(cell_width, pad=True)
    return cell


def count_chart_columns(chart_count: int, usable_width: int) -> int:
    """Grid columns: one below 90 usable cells, two from 90, three from 136."""
    by_width = max(1, (usable_width + CHART_COLUMN_GAP) // CHART_MIN_COLUMN_WIDTH)
    return max(1, min(chart_count, CHART_MAX_COLUMNS, by_width))


def render_chart_grid(block: dict, mode: int, width: int) -> list[Text]:
    """Grid lines of one GPU: charts fill rows left to right, top to bottom."""
    usable_width = max(1, width - CHART_WIDTH_MARGIN)
    charts = list(CHARTS) if mode == CHART_ALL else [CHARTS[mode]]
    columns = count_chart_columns(len(charts), usable_width)
    cell_width = max(1, (usable_width - CHART_COLUMN_GAP * (columns - 1)) // columns)
    lines = []
    for start in range(0, len(charts), columns):
        cells = [render_chart_cell(chart, block, cell_width) for chart in charts[start:start + columns]]
        line = Text(" " * CHART_COLUMN_GAP).join(cells)
        line.no_wrap = True
        line.overflow = "crop"
        line.rstrip()
        lines.append(line)
    return lines


def render_heading(block: dict) -> Text:
    """The bold "GPU <index>" line above a block when several GPUs are shown."""
    return Text(f"GPU {block['index']}", style="bold")


def render_charts(blocks: list[dict], mode: int, width: int) -> Text:
    """The chart grid of every shown GPU, stacked."""
    lines = []
    for block in blocks:
        if block["heading"]:
            lines.append(render_heading(block))
        lines.extend(render_chart_grid(block, mode, width))
    grid = Text("\n").join(lines)
    grid.no_wrap = True
    grid.overflow = "crop"
    return grid


def render_chart_note(blocks: list[dict]) -> Text:
    """The scale note: one line per GPU with the current adaptive scale tops."""
    lines = []
    for block in blocks:
        power_top = get_chart_top(CHARTS[5], block)
        clock_top = get_chart_top(CHARTS[1], block)
        prefix = f"GPU {block['index']}: " if block["heading"] else ""
        lines.append(
            f"{prefix}Recent samples - blank = no reading - power 0-{power_top:.0f} W"
            f" - {get_clock_name(block['adapter'])} 0-{clock_top:.0f} MHz"
        )
    return Text("\n".join(lines))


def render_name_line(snapshot: dict, slim: bool, paused: bool) -> Text:
    """Line 1 of a device block: name, driver strings, kernel driver, bus address."""
    line = Text()
    line.append(records.get_device_name(snapshot), style="bold")
    device = snapshot["device"]
    if not slim:
        if snapshot["driver"]:
            line.append(f"   driver {snapshot['driver']}")
        if snapshot["cuda"]:
            line.append(f"  CUDA {snapshot['cuda']}")
        if device is not None:
            line.append("  ")
            line.append(device["driver_model"], style="bold yellow")
            line.append("  ")
            line.append(device["pci_bus"])
    if paused:
        line.append(PAUSED_LABEL, style="bold red")
    return line


def get_bar_width(width: int) -> int:
    """Width of the VRAM bar for a terminal width."""
    return int(tui_draw.clamp_number(width - BAR_WIDTH_MARGIN, BAR_WIDTH_MIN, BAR_WIDTH_MAX))


def get_kind_style(kind: str) -> str:
    """Bar and legend style of a process kind."""
    return constants.KIND_STYLES.get(kind, constants.KIND_FALLBACK_STYLE)


def build_vram_parts(snapshot: dict, totals: dict) -> list[tuple[float, str]]:
    """Bar parts: kinds largest first (normalised), then the unattributed remainder."""
    if snapshot["vram_used"] is None:
        return []
    parts = [(amount * totals["scale"], get_kind_style(kind)) for kind, amount in totals["kinds"]]
    parts.append((totals["unattributed"], constants.UNATTRIBUTED_STYLE))
    return parts


def render_vram_line(snapshot: dict, totals: dict, width: int) -> Text:
    """Line 2 of a device block: the stacked VRAM bar with used and total."""
    line = Text("VRAM ")
    parts = build_vram_parts(snapshot, totals)
    line.append_text(tui_draw.draw_stacked_bar(parts, snapshot["vram_total"], get_bar_width(width)))
    used_label = formatting.format_size_or_na(snapshot["vram_used"])
    total_label = formatting.format_size(snapshot["vram_total"]) if snapshot["vram_total"] else formatting.MISSING_SIZE
    line.append(f" {used_label} / {total_label}", style="bold")
    return line


def render_legend_line(snapshot: dict, totals: dict) -> Text:
    """Line 3 of a device block: kind legend, unattributed, free, shared, normalisation."""
    line = Text(LEGEND_INDENT)
    for kind, amount in totals["kinds"][:LEGEND_MAX_KINDS]:
        line.append(tui_draw.LEGEND_SQUARE, style=get_kind_style(kind))
        line.append(f"{kind} {formatting.format_size(amount)}  ")
    if totals["unattributed"] > LEGEND_MIN_UNATTRIBUTED:
        line.append(tui_draw.LEGEND_SQUARE, style=constants.UNATTRIBUTED_STYLE)
        line.append(f"unattributed {formatting.format_size(totals['unattributed'])}  ")
    if totals["free"] is not None:
        line.append(f"free {formatting.format_size(totals['free'])}", style=constants.MUTED_STYLE)
    shared_used = snapshot["shared_used"]
    if shared_used is not None and shared_used > LEGEND_MIN_SHARED:
        line.append(f"   shared(RAM) {formatting.format_size(shared_used)}", style="yellow")
    if totals["scale"] < 1.0:
        line.append("   [bar normalized; labels are raw]", style=constants.MUTED_STYLE)
    return line


def list_legend_notes(snapshot: dict, error: str) -> list[str]:
    """Note lines under the legend: memory estimate, counter status, scan status, refresh error."""
    notes = [records.get_memory_note(snapshot), snapshot["counter_status"]]
    if snapshot["process_status"]:
        notes.append(f"DRM process scan: {snapshot['process_status']}.")
    notes.append(format_refresh_error(error))
    return [note for note in notes if note]


def format_refresh_error(error: str) -> str:
    """The legend note of a failed refresh; empty when the last refresh worked."""
    if not error:
        return ""
    return f"Refresh failed: {error} - retrying"


def render_load_line(device: dict) -> Text:
    """Line 4 of a device block: device load, memory activity, PCIe traffic, codec load."""
    line = Text()
    load = formatting.format_metric(device["util_gpu"], ".0f", "%")
    line.append(f"{records.get_device_label(device)} {load}", style=LOAD_STYLE)
    activity = formatting.format_metric(device["util_mem"], ".0f", "%")
    line.append(f"  Mem activity {activity}", style=MEM_ACTIVITY_STYLE)
    received = formatting.format_megabytes_per_second(device["pcie_rx_kbs"])
    sent = formatting.format_megabytes_per_second(device["pcie_tx_kbs"])
    line.append(f"  PCIe↓{received:>5} ↑{sent:>4} MB/s")
    if device["enc"] or device["dec"]:
        encoder = formatting.format_metric(device["enc"], ".0f", "%")
        decoder = formatting.format_metric(device["dec"], ".0f", "%")
        line.append(f"  enc {encoder} dec {decoder}")
    if device["video"] is not None:
        line.append(f"  video {formatting.format_metric(device['video'], '.0f', '%')}")
    return line


def format_temperature(device: dict) -> str:
    """Temperature with the fallback sensor label when one is used."""
    label = formatting.format_metric(device["temp"], ".0f", "°C")
    if device["temp_kind"]:
        label += f" ({device['temp_kind']})"
    return label


def format_clock(device: dict, with_maximum: bool) -> str:
    """Main clock of the device: SM, gfx or GT, with the maximum where one is drawn."""
    label, value = records.get_main_clock(device)
    current = formatting.format_metric(value, ".0f")
    if with_maximum and not records.is_amd_device(device):
        maximum = formatting.format_metric(device["sm_clock_max"], ".0f")
        return f"{label} {current}/{maximum} MHz"
    return f"{label} {current} MHz"


def list_full_sensor_parts(device: dict) -> list[str]:
    """Parts of the sensors line in full mode."""
    power = formatting.format_metric(device["power_w"], ".0f")
    power_limit = formatting.format_metric(device["power_limit_w"], ".0f")
    power_part = f"{power}/{power_limit} W"
    if device["power_kind"]:
        power_part += f" {device['power_kind']}"
    parts = [format_temperature(device), power_part]
    if device["fan"] is not None:
        parts.append(f"fan {device['fan']}%")
    parts.append(f"P{device['pstate']}" if device["pstate"] is not None else "P N/A")
    parts.append(format_clock(device, True))
    mem_clock = formatting.format_metric(device["mem_clock"], ".0f")
    video_clock = formatting.format_metric(device["video_clock"], ".0f")
    parts.append(f"mem {mem_clock} MHz")
    parts.append(f"video {video_clock} MHz")
    generation = formatting.format_metric(device["pcie_gen"])
    lanes = formatting.format_metric(device["pcie_width"])
    parts.append(f"PCIe gen{generation} x{lanes}")
    if device["source"] == constants.SOURCE_NVML:
        parts.append(f"CC {device['compute_cap'] or 'N/A'}")
    else:
        parts.append(device["source"])
    return parts


def list_slim_sensor_parts(device: dict) -> list[str]:
    """Parts of the sensors line in slim mode: temperature, power draw, main clock."""
    power_part = f"{formatting.format_metric(device['power_w'], '.0f')} W"
    if device["power_kind"]:
        power_part += f" {device['power_kind']}"
    return [format_temperature(device), power_part, format_clock(device, False)]


def render_sensor_line(device: dict, slim: bool) -> Text:
    """Line 5 of a device block: sensors, and the active throttle reasons."""
    parts = list_slim_sensor_parts(device) if slim else list_full_sensor_parts(device)
    line = Text("  ".join(parts), style=SENSOR_STYLE)
    if device["throttle"]:
        line.append(f"  throttle: {','.join(device['throttle'])}", style="yellow")
    return line


def render_device_block(block: dict, width: int, slim: bool, paused: bool) -> list[Text]:
    """All lines of one GPU in the device panel."""
    lines = [render_heading(block)] if block["heading"] else []
    snapshot = block["snapshot"]
    if snapshot is None:
        waiting = format_refresh_error(block["error"]) or DEVICE_PLACEHOLDER
        lines.append(Text(waiting, style=constants.MUTED_STYLE if block["error"] else ""))
        return lines
    totals = records.compute_kind_totals(snapshot)
    lines.append(render_name_line(snapshot, slim, paused))
    lines.append(render_vram_line(snapshot, totals, width))
    legend = render_legend_line(snapshot, totals)
    if legend.plain.strip():
        lines.append(legend)
    for note in list_legend_notes(snapshot, block["error"]):
        lines.append(Text(LEGEND_INDENT + note, style=constants.MUTED_STYLE))
    if snapshot["device"] is not None:
        lines.append(render_load_line(snapshot["device"]))
        lines.append(render_sensor_line(snapshot["device"], slim))
    return lines


def render_device_panel(blocks: list[dict], width: int, slim: bool, paused: bool) -> Text | None:
    """The device panel of every shown GPU; None while nothing was delivered yet."""
    has_content = any(block["snapshot"] is not None or block["error"] for block in blocks)
    if not has_content:
        return None
    lines = []
    for block in blocks:
        lines.extend(render_device_block(block, width, slim, paused))
    return Text("\n").join(lines)


def get_device_title(slim: bool) -> str:
    """Border title of the device panel."""
    return DEVICE_TITLE_SLIM if slim else DEVICE_TITLE_FULL


def compute_ram_percent(sample: dict | None) -> float | None:
    """RAM in use as a percentage; None without a sample or with an unknown total."""
    if sample is None or not sample["ram_total"]:
        return None
    return 100.0 * sample["ram_used"] / sample["ram_total"]


def append_tail_piece(tail: Text, piece: str, style: str, room: int) -> None:
    """Add an optional piece to a tail when the bar still keeps its minimum width."""
    if tail.cell_len + len(piece) + SYSTEM_MIN_BAR <= room:
        tail.append(piece, style=style)


def build_ram_tail(sample: dict, room: int) -> Text:
    """Text right of the RAM bar: used / total, percent, then free and swap when they fit."""
    tail = Text()
    used_label = formatting.format_size(sample["ram_used"])
    total_label = formatting.format_size(sample["ram_total"])
    tail.append(f" {used_label} / {total_label}", style="bold")
    tail.append(f"  {compute_ram_percent(sample) or 0.0:.0f}%", style=constants.MUTED_STYLE)
    free_piece = f"  free {formatting.format_size(sample['ram_available'])}"
    append_tail_piece(tail, free_piece, constants.MUTED_STYLE, room)
    if sample["swap_total"] > 0:
        swap_used = formatting.format_size(sample["swap_used"])
        swap_total = formatting.format_size(sample["swap_total"])
        heavy = sample["swap_used"] > sample["swap_total"] / 4
        append_tail_piece(tail, f"  swap {swap_used}/{swap_total}", "yellow" if heavy else constants.MUTED_STYLE, room)
    return tail


def build_cpu_tail(sample: dict, room: int, slim: bool) -> Text:
    """Text right of the CPU bar: machine load, then the thread and core counts in full mode."""
    tail = Text()
    tail.append(f" {sample['cpu_pct']:>4.0f}%", style="bold")
    if slim:
        return tail
    counts = f"  {sample['logical']} threads"
    if sample["physical"] is not None:
        counts += f"/{sample['physical']} cores"
    append_tail_piece(tail, counts, constants.MUTED_STYLE, room)
    return tail


def render_system_bars(sample: dict, room: int, slim: bool) -> list[Text]:
    """The RAM line and the CPU line; both bars share one width."""
    ram_tail = build_ram_tail(sample, room)
    cpu_tail = build_cpu_tail(sample, room, slim)
    longer_tail = max(ram_tail.cell_len, cpu_tail.cell_len)
    bar_width = max(SYSTEM_MIN_BAR, min(SYSTEM_MAX_BAR, room - longer_tail))
    ram_line = Text("RAM  ")
    ram_parts = [(sample["ram_used"], SYSTEM_RAM_STYLE)]
    ram_line.append_text(tui_draw.draw_stacked_bar(ram_parts, sample["ram_total"], bar_width))
    ram_line.append_text(ram_tail)
    cpu_line = Text("CPU  ")
    cpu_parts = [(sample["cpu_pct"], SYSTEM_CPU_STYLE)]
    cpu_line.append_text(tui_draw.draw_stacked_bar(cpu_parts, 100.0, bar_width))
    cpu_line.append_text(cpu_tail)
    return [ram_line, cpu_line]


def get_thread_style(load: float) -> str:
    """Colour of one per-thread glyph."""
    if load >= THREAD_HOT_PCT:
        return "red"
    if load >= THREAD_WARM_PCT:
        return "yellow"
    return SYSTEM_CPU_STYLE


def render_thread_line(sample: dict) -> Text:
    """One glyph per logical CPU, coloured by load."""
    line = Text(LEGEND_INDENT)
    for load in sample["per_core"]:
        line.append(tui_draw.get_spark_glyph(load), style=get_thread_style(load))
    line.append("  per-thread", style=SYSTEM_CAPTION_STYLE)
    return line


def render_system_history_line(cpu_history, ram_history, room: int) -> Text:
    """CPU and RAM sparklines of the recent samples."""
    spark_width = max(8, min(30, (room - 10) // 2))
    line = Text(LEGEND_INDENT)
    line.append_text(tui_draw.draw_sparkline(cpu_history, spark_width, 100.0, SYSTEM_CPU_STYLE))
    line.append(" CPU   ", style=SYSTEM_CAPTION_STYLE)
    line.append_text(tui_draw.draw_sparkline(ram_history, spark_width, 100.0, SYSTEM_RAM_STYLE))
    line.append(" RAM", style=SYSTEM_CAPTION_STYLE)
    return line


def render_top_ram_line(sample: dict) -> Text:
    """The five largest processes by resident memory."""
    line = Text(LEGEND_INDENT)
    line.append("top RAM: ", style=constants.MUTED_STYLE)
    largest = system.get_top_procs(sample, "rss", SYSTEM_TOP_COUNT)
    entries = [f"{proc['name']} {formatting.format_size(proc['rss'])}" for proc in largest]
    line.append("  ".join(entries))
    return line


def render_top_cpu_line(sample: dict) -> Text:
    """The busiest processes; idle when none is busy, the marker when none is rated yet."""
    line = Text(LEGEND_INDENT)
    line.append("top CPU: ", style=constants.MUTED_STYLE)
    rated = system.get_top_procs(sample, "cpu_pct", SYSTEM_TOP_COUNT)
    busy = [proc for proc in rated if proc["cpu_pct"] >= SYSTEM_MIN_CPU_PCT]
    if busy:
        line.append("  ".join(f"{proc['name']} {proc['cpu_pct']:.0f}%" for proc in busy))
    elif rated:
        line.append("idle")
    else:
        line.append(constants.NO_READING)
    return line


def render_system_panel(
    sample: dict | None,
    error: str,
    cpu_history: deque,
    ram_history: deque,
    width: int,
    slim: bool,
) -> Text | None:
    """The system panel; None when there is neither a sample nor an error (keep the old text)."""
    if sample is None:
        if not error:
            return None
        return Text(f"RAM/CPU unavailable: {error}", style="yellow", no_wrap=True, overflow="ellipsis")
    room = width - SYSTEM_WIDTH_MARGIN
    lines = render_system_bars(sample, room, slim)
    if not slim:
        lines.append(render_thread_line(sample))
        lines.append(render_system_history_line(cpu_history, ram_history, room))
        lines.append(render_top_ram_line(sample))
        lines.append(render_top_cpu_line(sample))
    panel = Text("\n").join(lines)
    panel.no_wrap = True
    panel.overflow = "ellipsis"
    return panel


def main():
    pass


if __name__ == "__main__":
    main()
