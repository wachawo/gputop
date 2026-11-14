#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Row filter, sort modes, column list and cell renderers of the gputop process table."""

from rich.text import Text

# Local imports
from gputop import constants
from gputop import formatting
from gputop import records

# Column titles
COLUMN_PID = "PID"
COLUMN_PROCESS = "Process"
COLUMN_KIND = "Kind"
COLUMN_RUNTIME = "Runtime"
COLUMN_VRAM = "VRAM"
COLUMN_SHARED = "Shared"
COLUMN_GPU_PCT = "GPU%"
COLUMN_3D = "3D"
COLUMN_COPY = "Copy"
COLUMN_VIDEO = "Video"
COLUMN_RAM = "RAM"
COLUMN_CPU = "CPU%"
COLUMN_INSIDE = "What's inside"
COLUMN_GPU = "GPU"

# Sort modes
SORT_VRAM = "VRAM"
SORT_GPU_PCT = "GPU%"
SORT_SHARED = "Shared"
SORT_NAME = "Process"
SORT_RAM = "RAM"
SORT_CPU = "CPU%"
# Every sort mode is named after the column it sorts by. The name sort runs A to Z, the
# others largest first; the header marks the sorted column with the matching arrow.
SORT_MARK_ASCENDING = "▲"
SORT_MARK_DESCENDING = "▼"
SORTED_LABEL_STYLE = "bold yellow"

# Top filter limits
TOP_MIN_DEDICATED = 32 * constants.MIB
TOP_MIN_GPU_PCT = 1.0
TOP_MIN_SHARED = 64 * constants.MIB

# Cell rules
SHARED_MIN_SHOWN = constants.MIB
PERCENT_MIN_SHOWN = 0.5
PERCENT_BOLD_FROM = 20.0
RUNTIME_STYLE = "grey70"


def create_column(title: str, justify: str) -> dict:
    """Create a column record: title (str, also the column key) and justify ("left" or "right")."""
    return {"title": title, "justify": justify}


def list_columns(show_system: bool, has_drm: bool, multi_gpu: bool) -> list[dict]:
    """The column list of a run; it never changes after the app has started."""
    columns = [
        create_column(COLUMN_PID, "right"),
        create_column(COLUMN_PROCESS, "left"),
        create_column(COLUMN_KIND, "left"),
        create_column(COLUMN_RUNTIME, "left"),
        create_column(COLUMN_VRAM, "right"),
    ]
    if has_drm:
        columns.append(create_column(COLUMN_SHARED, "right"))
    columns.append(create_column(COLUMN_GPU_PCT, "right"))
    columns.append(create_column(COLUMN_3D, "right"))
    if has_drm:
        columns.append(create_column(COLUMN_COPY, "right"))
    columns.append(create_column(COLUMN_VIDEO, "right"))
    if show_system:
        columns.append(create_column(COLUMN_RAM, "right"))
        columns.append(create_column(COLUMN_CPU, "right"))
    columns.append(create_column(COLUMN_INSIDE, "left"))
    if multi_gpu:
        columns.append(create_column(COLUMN_GPU, "left"))
    return columns


def render_column_label(column: dict, mode: str = "") -> Text:
    """Header label of a column, aligned like its cells; the sorted column carries an arrow."""
    if column["title"] != mode:
        return Text(column["title"], justify=column["justify"])
    mark = SORT_MARK_ASCENDING if mode == SORT_NAME else SORT_MARK_DESCENDING
    label = f"{column['title']}{mark}"
    return Text(label, style=SORTED_LABEL_STYLE, justify=column["justify"])


def list_sort_modes(show_system: bool, has_drm: bool) -> list[str]:
    """Sort modes in cycle order."""
    modes = [SORT_VRAM, SORT_GPU_PCT]
    if has_drm:
        modes.append(SORT_SHARED)
    modes.append(SORT_NAME)
    if show_system:
        modes.extend([SORT_RAM, SORT_CPU])
    return modes


def is_top_row(row: dict) -> bool:
    """Top filter: a row that holds memory, is busy, is unattributed or has an LLM view."""
    gpu = row["gpu"]
    if not gpu["attributed"]:
        return True
    if gpu["dedicated"] >= TOP_MIN_DEDICATED:
        return True
    if records.get_gpu_pct(gpu) >= TOP_MIN_GPU_PCT:
        return True
    if gpu["shared"] >= TOP_MIN_SHARED:
        return True
    return row["llm"] is not None


def filter_rows(rows: list[dict], show_all: bool) -> list[dict]:
    """Rows the table shows: all of them, or the Top selection."""
    if show_all:
        return list(rows)
    return [row for row in rows if is_top_row(row)]


def get_unknown_last_key(value: float | None) -> tuple:
    """Sort key for a descending order with unknown values last."""
    if value is None:
        return (1, 0.0)
    return (0, -value)


def get_sort_key(row: dict, mode: str):
    """Sort key of one row for a sort mode (ascending order of the key)."""
    gpu = row["gpu"]
    if mode == SORT_GPU_PCT:
        return -records.get_gpu_pct(gpu)
    if mode == SORT_SHARED:
        return -gpu["shared"]
    if mode == SORT_NAME:
        return row["name"].lower()
    if mode == SORT_RAM:
        return get_unknown_last_key(row["ram"])
    if mode == SORT_CPU:
        return get_unknown_last_key(row["cpu"])
    return -(records.get_shown_vram(gpu) or 0.0)


def sort_rows(rows: list[dict], mode: str) -> list[dict]:
    """New list in the order of a sort mode; the sort is stable."""
    return sorted(rows, key=lambda row: get_sort_key(row, mode))


def list_visible_rows(snapshots: list, show_all: bool, mode: str) -> list[dict]:
    """Rows of every shown GPU: filtered per GPU, concatenated, then sorted once."""
    visible = []
    for snapshot in snapshots:
        if snapshot is not None:
            visible.extend(filter_rows(snapshot["procs"], show_all))
    return sort_rows(visible, mode)


def find_llm_selection(snapshots: list) -> str | None:
    """Key of the LLM row with the largest VRAM across all GPUs, or None without LLM rows."""
    best_key = None
    best_vram = -1.0
    for snapshot in snapshots:
        rows = snapshot["procs"] if snapshot is not None else []
        for row in rows:
            vram = records.get_shown_vram(row["gpu"]) or 0.0
            if row["llm"] is not None and vram > best_vram:
                best_key = records.get_row_key(row)
                best_vram = vram
    return best_key


def format_table_status(mode: str, show_all: bool) -> str:
    """Text of the status line under the table."""
    return f"sort: {mode} - {'all' if show_all else 'top'} processes"


def render_percent_cell(value: float | None) -> Text:
    """Percent cell: n/a when unknown, blank when idle, bold when busy."""
    if value is None:
        return Text(formatting.MISSING_CELL, style=constants.MUTED_STYLE, justify="right")
    if value < PERCENT_MIN_SHOWN:
        return Text("", justify="right")
    style = "bold" if value >= PERCENT_BOLD_FROM else ""
    return Text(f"{value:.0f}", style=style, justify="right")


def render_pid_cell(row: dict, snapshot: dict) -> Text:
    """PID, marked for a process of another PID namespace."""
    style = constants.MUTED_STYLE if row["outside"] else ""
    return Text(records.get_pid_label(row), style=style, justify="right")


def render_process_cell(row: dict, snapshot: dict) -> Text:
    """Process name."""
    return Text(row["name"])


def render_kind_cell(row: dict, snapshot: dict) -> Text:
    """Kind in its colour; an unknown kind has no style."""
    return Text(row["kind"], style=constants.KIND_STYLES.get(row["kind"], ""))


def render_runtime_cell(row: dict, snapshot: dict) -> Text:
    """First runtime labels of the process."""
    return Text(records.get_runtime_label(row["detail"]), style=RUNTIME_STYLE)


def render_vram_cell(row: dict, snapshot: dict) -> Text:
    """Device memory of the process with its inference mark."""
    known = records.get_shown_vram(row["gpu"]) is not None
    style = "bold" if known else constants.MUTED_STYLE
    return Text(records.format_vram_label(row["gpu"]), style=style, justify="right")


def render_shared_cell(row: dict, snapshot: dict) -> Text:
    """GPU memory of the process in system RAM; red for a model spilling out of VRAM."""
    gpu = row["gpu"]
    if constants.COUNTER_SHARED not in gpu["available"]:
        is_local_drm = snapshot["backend"] == constants.BACKEND_DRM and not row["outside"]
        if is_local_drm:
            return Text(formatting.MISSING_CELL, style=constants.MUTED_STYLE, justify="right")
        return Text("", justify="right")
    if gpu["shared"] < SHARED_MIN_SHOWN:
        return Text("", justify="right")
    style = "bold red" if records.is_spill(row, snapshot) else "yellow"
    return Text(formatting.format_size(gpu["shared"]), style=style, justify="right")


def render_gpu_pct_cell(row: dict, snapshot: dict) -> Text:
    """Busiest engine of the process."""
    return render_percent_cell(records.get_engine_columns(row["gpu"])["gpu"])


def render_3d_cell(row: dict, snapshot: dict) -> Text:
    """3D plus compute load."""
    return render_percent_cell(records.get_engine_columns(row["gpu"])["3d"])


def render_copy_cell(row: dict, snapshot: dict) -> Text:
    """Copy engine load."""
    return render_percent_cell(records.get_engine_columns(row["gpu"])["copy"])


def render_video_cell(row: dict, snapshot: dict) -> Text:
    """Decode plus encode plus video engine load."""
    return render_percent_cell(records.get_engine_columns(row["gpu"])["video"])


def render_ram_cell(row: dict, snapshot: dict) -> Text:
    """Resident system memory of the process."""
    if row["ram"] is None:
        return Text(formatting.MISSING_CELL, style=constants.MUTED_STYLE, justify="right")
    return Text(formatting.format_size(row["ram"]), justify="right")


def render_cpu_cell(row: dict, snapshot: dict) -> Text:
    """CPU share of the whole machine; the marker while the process is not rated yet."""
    if row["cpu"] is None and row["ram"] is not None:
        return Text(constants.NO_READING, style=constants.MUTED_STYLE, justify="right")
    return render_percent_cell(row["cpu"])


def render_inside_cell(row: dict, snapshot: dict) -> Text:
    """One-line summary of what the process keeps in GPU memory."""
    is_llm = row["llm"] is not None or row["kind"] == constants.KIND_LLM
    style = "magenta" if is_llm else constants.MUTED_STYLE
    return Text(row["inside"], style=style, no_wrap=True, overflow="ellipsis")


def render_gpu_cell(row: dict, snapshot: dict) -> Text:
    """Index of the GPU the row belongs to."""
    index = row["gpu_index"]
    return Text("" if index is None else str(index))


CELL_RENDERERS = {
    COLUMN_PID: render_pid_cell,
    COLUMN_PROCESS: render_process_cell,
    COLUMN_KIND: render_kind_cell,
    COLUMN_RUNTIME: render_runtime_cell,
    COLUMN_VRAM: render_vram_cell,
    COLUMN_SHARED: render_shared_cell,
    COLUMN_GPU_PCT: render_gpu_pct_cell,
    COLUMN_3D: render_3d_cell,
    COLUMN_COPY: render_copy_cell,
    COLUMN_VIDEO: render_video_cell,
    COLUMN_RAM: render_ram_cell,
    COLUMN_CPU: render_cpu_cell,
    COLUMN_INSIDE: render_inside_cell,
    COLUMN_GPU: render_gpu_cell,
}


def render_row_cells(row: dict, snapshot: dict, columns: list[dict]) -> list[Text]:
    """Cells of one table row, one per column of the run."""
    return [CELL_RENDERERS[column["title"]](row, snapshot) for column in columns]


def main():
    pass


if __name__ == "__main__":
    main()
