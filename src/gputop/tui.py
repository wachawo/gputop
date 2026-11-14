#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Textual application of gputop: widgets, key bindings and the sampling worker."""

import concurrent.futures
import logging
import threading
import traceback
from collections import deque

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Markdown, Select, Static
from textual.widgets.data_table import CellDoesNotExist

# Local imports
from gputop import adapters
from gputop import clipboard
from gputop import collector
from gputop import constants
from gputop import formatting
from gputop import nvidia
from gputop import records
from gputop import system
from gputop import tui_help
from gputop import tui_inspector
from gputop import tui_panels
from gputop import tui_table

# Texts
APP_TITLE = "gputop"
CHART_LABEL = "Charts"
CHART_HINT = "Click to choose - c cycles"

# Styles
APP_CSS = """
Screen {
    layout: vertical;
}
#dashboard {
    height: auto;
    max-height: 50%;
}
#device {
    height: auto;
    border: round $accent;
    padding: 0 1;
}
#system {
    height: auto;
    border: round $success;
    padding: 0 1;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}
#history {
    height: auto;
    border: round $secondary;
    padding: 0 1;
}
#chart-controls {
    height: 1;
}
#chart-label {
    width: 8;
    text-style: bold;
}
#chart-mode {
    width: 24;
}
#chart-mode SelectOverlay {
    border: solid $accent;
    padding: 0 1;
}
#chart-hint {
    width: 1fr;
    color: $text-muted;
}
#charts {
    height: auto;
}
#chart-note {
    height: auto;
    color: $text-muted;
}
#hidden-note {
    height: 1;
    padding: 0 2;
    color: $text-muted;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}
#procs {
    height: 1fr;
    min-height: 4;
}
#table-status {
    height: 1;
    text-align: right;
    color: $text-muted;
}
#inspector {
    height: auto;
    max-height: 30%;
    overflow-y: auto;
    border: round $secondary;
    padding: 0 1;
}
"""
HELP_CSS = """
HelpScreen {
    align: center middle;
}
#help-body {
    width: 140;
    max-width: 95%;
    height: 85%;
    border: thick $accent;
    background: $surface;
    padding: 1 2;
}
"""

NO_CLIPBOARD_TOOL_NOTE = "Sent to the terminal only: install xclip, xsel or wl-clipboard"
# Click.chain of the second click of a double click
DOUBLE_CLICK = 2
# View state that survives a switch of the shown GPU set
CARRIED_KEYS = ("slim", "show_all", "cpu_history", "ram_history")

logger = logging.getLogger(__name__)


def select_shown_indexes(ordered_adapters: list[dict], gpu_choice: int | str | None) -> list[int]:
    """The shown GPU set: the asked index, every adapter for "all", else the default set."""
    if gpu_choice == constants.GPU_ALL:
        return list(range(len(ordered_adapters))) or [0]
    if gpu_choice is not None:
        return [gpu_choice]
    return adapters.list_default_indexes(ordered_adapters)


def format_hidden_note(ordered_adapters: list[dict], gpu_choice: int | str | None) -> str:
    """One line naming the integrated GPUs the default set left out; "" when none were."""
    if gpu_choice is not None:
        return ""
    shown = adapters.list_default_indexes(ordered_adapters)
    names = [
        adapter["name"] for index, adapter in enumerate(ordered_adapters) if index not in shown
    ]
    if not names:
        return ""
    noun = "GPU" if len(names) == 1 else "GPUs"
    return f"{len(names)} integrated {noun} hidden: {', '.join(names)} - press g to show"


def find_shown_adapter(ordered_adapters: list[dict], index: int) -> dict | None:
    """The adapter at a display index, or None when the list does not reach it."""
    if 0 <= index < len(ordered_adapters):
        return ordered_adapters[index]
    return None


def create_view_state(
    ordered_adapters: list[dict],
    shown_indexes: list[int],
    interval: float,
    show_system: bool,
    hidden_note: str = "",
) -> dict:
    """Create the UI state of one run.

    Keys: ordered_adapters (list of adapter, the enumeration of this run), shown_indexes
    (list of int, fixed), hidden_note (str, the line about GPUs left out, "" for none),
    can_switch_gpus (bool, g has two GPU sets to switch between), switch_gpus (bool, the
    app was left to reopen with the other GPU set), multi_gpu (bool, several GPUs are
    shown), interval (float, seconds between refreshes), show_system (bool), blocks (list of GPU block, one per shown GPU, see
    tui_panels.create_gpu_block), columns (list of column, fixed), sort_modes (list of str,
    fixed), sort_index (int), show_all (bool, False = Top filter), chart_mode (int, -1 = all
    charts), slim (bool), paused (bool, read by the worker), system_sample (system sample or
    None), system_error (str), cpu_history and ram_history (deque of float or None),
    row_index (dict row key to (row, snapshot)), selected_key (str or None),
    auto_selected (bool, the one-time LLM selection was done), stop_event (threading.Event).
    """
    multi_gpu = len(shown_indexes) > 1
    shown_adapters = [find_shown_adapter(ordered_adapters, index) for index in shown_indexes]
    foreign = [adapter for adapter in shown_adapters if adapter is not None
               and adapter["vendor_id"] != constants.VENDOR_NVIDIA]
    has_drm = bool(foreign)
    blocks = [
        tui_panels.create_gpu_block(index, adapter, multi_gpu)
        for index, adapter in zip(shown_indexes, shown_adapters)
    ]
    return {
        "ordered_adapters": ordered_adapters,
        "shown_indexes": shown_indexes,
        "hidden_note": hidden_note,
        "can_switch_gpus": False,
        "switch_gpus": False,
        "multi_gpu": multi_gpu,
        "interval": interval,
        "show_system": show_system,
        "blocks": blocks,
        "columns": tui_table.list_columns(show_system, has_drm, multi_gpu),
        "sort_modes": tui_table.list_sort_modes(show_system, has_drm),
        "sort_index": 0,
        "show_all": False,
        "chart_mode": tui_panels.CHART_ALL,
        "slim": False,
        "paused": False,
        "system_sample": None,
        "system_error": "",
        "cpu_history": deque(maxlen=constants.HISTORY_LENGTH),
        "ram_history": deque(maxlen=constants.HISTORY_LENGTH),
        "row_index": {},
        "selected_key": None,
        "auto_selected": False,
        "stop_event": threading.Event(),
    }


def log_failure(exc: BaseException) -> None:
    """Log an unexpected exception caught at a boundary."""
    logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")


def open_collectors(view_state: dict, collectors: list[dict]) -> None:
    """Open one collector per shown GPU, in order, sharing the deep-scan budget."""
    shown_indexes = view_state["shown_indexes"]
    nvidia_shared = nvidia.create_shared_reads()
    for index in shown_indexes:
        opened = collector.open_collector(
            index,
            budget_scale=1 / len(shown_indexes),
            ordered_adapters=view_state["ordered_adapters"],
            nvidia_shared=nvidia_shared,
        )
        collectors.append(opened)


def close_collectors(collectors: list[dict]) -> None:
    """Close the collectors in reverse order."""
    for opened in reversed(collectors):
        collector.close_collector(opened)


def sample_collectors(collectors: list[dict]) -> tuple[list, list[str]]:
    """Sample every collector in order; a failure keeps its slot empty and records the error."""
    snapshots = []
    errors = []
    for opened in collectors:
        try:
            snapshots.append(collector.sample_collector(opened))
            errors.append("")
        except Exception as exc:
            log_failure(exc)
            snapshots.append(None)
            errors.append(formatting.format_error(exc))
    return snapshots, errors


def build_payload(view_state: dict, collectors: list[dict], sampler_box: dict) -> dict:
    """One refresh: the snapshots of every shown GPU, then the system sample."""
    snapshots, errors = sample_collectors(collectors)
    system_sample = None
    system_error = ""
    if view_state["show_system"]:
        system_sample, system_error = system.take_system_reading(sampler_box)
    return {
        "snapshots": snapshots,
        "errors": errors,
        "system": system_sample,
        "system_error": system_error,
    }


def run_sampling_loop(view_state: dict, deliver) -> None:
    """Body of the sampling thread: it owns every collector and the system sampler."""
    stop_event = view_state["stop_event"]
    sampler_box = {"sampler": None}
    if view_state["show_system"]:
        system.take_system_reading(sampler_box)
    collectors = []
    try:
        open_collectors(view_state, collectors)
        if stop_event.wait(constants.TUI_WARMUP_SECONDS):
            return
        while not stop_event.is_set():
            if not view_state["paused"]:
                payload = build_payload(view_state, collectors, sampler_box)
                if not stop_event.is_set() and not view_state["paused"]:
                    deliver(payload)
            stop_event.wait(view_state["interval"])
    finally:
        close_collectors(collectors)


def deliver_payload(app, payload: dict) -> None:
    """Hand one refresh to the UI thread; a hand-off to a closing app is dropped.

    call_from_thread also re-raises what the refresh raised on the UI thread, so the same
    exception types from a running app are a rendering failure and are logged as one.
    """
    try:
        app.call_from_thread(app.apply_refresh, payload)
    except (RuntimeError, concurrent.futures.CancelledError) as exc:
        if app.view_state["stop_event"].is_set():
            logger.debug(f"Refresh dropped: {type(exc).__name__}: {str(exc)}")
        else:
            log_failure(exc)


def store_system_sample(view_state: dict, payload: dict) -> None:
    """Replace the stored system sample and extend the CPU and RAM histories."""
    sample = payload["system"]
    view_state["system_sample"] = sample
    view_state["system_error"] = payload["system_error"]
    view_state["cpu_history"].append(sample["cpu_pct"] if sample is not None else None)
    view_state["ram_history"].append(tui_panels.compute_ram_percent(sample))


def store_snapshots(view_state: dict, payload: dict) -> None:
    """A delivered snapshot replaces the stored one; a failed refresh keeps it and its error."""
    for position, block in enumerate(view_state["blocks"]):
        snapshot = payload["snapshots"][position]
        if snapshot is not None:
            block["snapshot"] = snapshot
            block["error"] = ""
        else:
            block["error"] = payload["errors"][position]


def join_system_readings(view_state: dict) -> None:
    """Set gpu_index, ram and cpu on every stored row and rebuild the row index."""
    sample = view_state["system_sample"]
    row_index = {}
    for position, block in enumerate(view_state["blocks"]):
        snapshot = block["snapshot"]
        rows = snapshot["procs"] if snapshot is not None else []
        for row in rows:
            row["gpu_index"] = position if view_state["multi_gpu"] else None
            reading = None
            if sample is not None and not row["outside"]:
                reading = system.find_system_proc(sample, row["pid"], row["detail"]["create_time"])
            row["ram"] = reading["rss"] if reading is not None else None
            row["cpu"] = reading["cpu_pct"] if reading is not None else None
            row_index[records.get_row_key(row)] = (row, snapshot)
    view_state["row_index"] = row_index


def extend_histories(view_state: dict, payload: dict) -> None:
    """Append one sample to the eight histories of every GPU; a failed refresh adds gaps."""
    for position, block in enumerate(view_state["blocks"]):
        tui_panels.append_history(block["history"], payload["snapshots"][position])


def select_llm_row_once(view_state: dict, payload: dict) -> None:
    """On the first refresh with data, select the LLM row with the largest VRAM."""
    if view_state["auto_selected"]:
        return
    if all(snapshot is None for snapshot in payload["snapshots"]):
        return
    view_state["auto_selected"] = True
    stored = [block["snapshot"] for block in view_state["blocks"]]
    llm_key = tui_table.find_llm_selection(stored)
    if llm_key is not None:
        view_state["selected_key"] = llm_key


def apply_payload(view_state: dict, payload: dict) -> None:
    """Fold one refresh into the UI state, in the order the contract fixes."""
    store_system_sample(view_state, payload)
    store_snapshots(view_state, payload)
    join_system_readings(view_state)
    extend_histories(view_state, payload)
    select_llm_row_once(view_state, payload)


def find_widget(app, selector: str, widget_type):
    """A widget of the main screen, or None when it is gone (shutdown, not mounted yet)."""
    try:
        main_screen = app.screen_stack[0]
        return main_screen.query_one(selector, widget_type)
    except (NoMatches, IndexError):
        return None


def render_device(app) -> None:
    """Redraw the device panel and its border title."""
    view_state = app.view_state
    device_widget = find_widget(app, "#device", Static)
    if device_widget is None:
        return
    device_widget.border_title = tui_panels.get_device_title(view_state["slim"])
    panel = tui_panels.render_device_panel(
        view_state["blocks"], app.size.width, view_state["slim"], view_state["paused"]
    )
    if panel is not None:
        device_widget.update(panel)


def render_system(app) -> None:
    """Redraw the system panel; the previous text stays without a sample and without an error."""
    view_state = app.view_state
    system_widget = find_widget(app, "#system", Static)
    if system_widget is None:
        return
    panel = tui_panels.render_system_panel(
        view_state["system_sample"],
        view_state["system_error"],
        view_state["cpu_history"],
        view_state["ram_history"],
        app.size.width,
        view_state["slim"],
    )
    if panel is not None:
        system_widget.update(panel)


def render_charts(app) -> None:
    """Redraw the chart grid and the scale note."""
    view_state = app.view_state
    charts_widget = find_widget(app, "#charts", Static)
    note_widget = find_widget(app, "#chart-note", Static)
    if charts_widget is None or note_widget is None:
        return
    grid = tui_panels.render_charts(view_state["blocks"], view_state["chart_mode"], app.size.width)
    charts_widget.update(grid)
    note_widget.update(tui_panels.render_chart_note(view_state["blocks"]))


def render_dashboard(app) -> None:
    """Redraw device panel, system panel, charts and chart note."""
    render_device(app)
    if app.view_state["show_system"]:
        render_system(app)
    render_charts(app)


def reset_columns(table, columns: list[dict], mode: str) -> None:
    """Relabel every column for the sort mode and size it to its label again.

    The label carries the sort arrow; the width reset lets the new rows alone decide it.
    """
    for column in columns:
        shown = table.columns[column["title"]]
        shown.label = tui_table.render_column_label(column, mode)
        shown.content_width = shown.label.cell_len


def restore_cursor(table, keys: list[str], selected_key: str | None, offset: tuple) -> None:
    """Move the cursor back to the remembered row and the view back to its old offset.

    Both happen in the same turn as the rebuild, so no frame is painted at offset 0; the
    previous virtual size is still in place then, so the offset is not cut short.
    """
    if selected_key in keys:
        table.move_cursor(row=keys.index(selected_key), animate=False, scroll=False)
    scroll_x, scroll_y = offset
    if scroll_x > 0 or scroll_y > 0:
        table.scroll_to(x=scroll_x, y=scroll_y, animate=False, immediate=True)


def render_table(app) -> None:
    """Rebuild the table rows from the stored snapshots and update the status line."""
    view_state = app.view_state
    table = find_widget(app, "#procs", DataTable)
    status_widget = find_widget(app, "#table-status", Static)
    if table is None or status_widget is None:
        return
    mode = view_state["sort_modes"][view_state["sort_index"]]
    stored = [block["snapshot"] for block in view_state["blocks"]]
    rows = tui_table.list_visible_rows(stored, view_state["show_all"], mode)
    keys = [records.get_row_key(row) for row in rows]
    offset = (table.scroll_x, table.scroll_y)
    table.clear()
    reset_columns(table, view_state["columns"], mode)
    for key, row in zip(keys, rows):
        snapshot = view_state["row_index"][key][1]
        cells = tui_table.render_row_cells(row, snapshot, view_state["columns"])
        table.add_row(*cells, key=key)
    restore_cursor(table, keys, view_state["selected_key"], offset)
    status_widget.update(tui_table.format_table_status(mode, view_state["show_all"]))


def render_inspector_panel(app) -> None:
    """Redraw the inspector for the selected row."""
    view_state = app.view_state
    inspector_widget = find_widget(app, "#inspector", Static)
    if inspector_widget is None:
        return
    row, snapshot = view_state["row_index"].get(view_state["selected_key"], (None, None))
    inspector_widget.border_title = tui_inspector.format_inspector_title(row, view_state["multi_gpu"])
    inspector_widget.update(tui_inspector.render_inspector(row, snapshot, app.size.width))


def copy_inspector(app) -> None:
    """Copy the inspector of the selected row to the clipboard as plain text.

    The desktop's clipboard tool is the reliable path; the terminal escape sequence is sent
    as well because it is the only one that reaches the user's machine over SSH.
    """
    view_state = app.view_state
    row, snapshot = view_state["row_index"].get(view_state["selected_key"], (None, None))
    if row is None:
        return
    title = tui_inspector.format_inspector_title(row, view_state["multi_gpu"])
    text = tui_inspector.format_inspector_text(title, row, snapshot, app.size.width)
    used_tool = clipboard.copy_text(text)
    app.copy_to_clipboard(text)
    if used_tool:
        app.notify(f"Copied: {title}", timeout=2)
    else:
        app.notify(NO_CLIPBOARD_TOOL_NOTE, severity="warning", timeout=6)


def render_all(app) -> None:
    """Redraw everything from the stored state: dashboard, table, inspector."""
    render_dashboard(app)
    render_table(app)
    render_inspector_panel(app)


def get_cursor_key(table, fallback: str | None) -> str | None:
    """Key of the row under the cursor right now; the fallback when the table cannot say."""
    try:
        cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
    except CellDoesNotExist:
        return fallback
    return cell_key.row_key.value


def sync_selected_key(app) -> None:
    """Take the selection from the cursor itself before the rows are rebuilt.

    A cursor move whose highlight message has not reached the app yet would otherwise be
    undone by the rebuild. Runs before the one-time LLM selection, which may still override.
    """
    table = find_widget(app, "#procs", DataTable)
    if table is None or table.row_count == 0:
        return
    view_state = app.view_state
    view_state["selected_key"] = get_cursor_key(table, view_state["selected_key"])


def handle_refresh(app, payload: dict) -> None:
    """Apply step of one refresh on the UI thread; late and paused calls are ignored."""
    if not app.is_running or app.view_state["paused"]:
        return
    sync_selected_key(app)
    apply_payload(app.view_state, payload)
    render_all(app)


def handle_row_highlighted(app, event) -> None:
    """Remember the highlighted row by key and show it in the inspector."""
    if not app.is_running:
        return
    app.view_state["selected_key"] = get_cursor_key(event.data_table, event.row_key.value)
    render_inspector_panel(app)


def handle_chart_changed(app, value) -> None:
    """Store the chosen chart mode, redraw the charts and give the keys back to the table."""
    if not isinstance(value, int):
        return
    app.view_state["chart_mode"] = value
    render_charts(app)
    table = find_widget(app, "#procs", DataTable)
    if table is not None and app.screen is app.screen_stack[0]:
        table.focus()


def cycle_sort(app) -> None:
    """Advance the sort mode and redraw the table from the stored snapshots."""
    view_state = app.view_state
    view_state["sort_index"] = (view_state["sort_index"] + 1) % len(view_state["sort_modes"])
    render_table(app)


def cycle_chart(app) -> None:
    """Advance the chart mode through the selector, so keyboard and mouse share one path."""
    selector = find_widget(app, "#chart-mode", Select)
    if selector is not None:
        selector.value = tui_panels.get_next_chart_mode(app.view_state["chart_mode"])


def toggle_filter(app) -> None:
    """Switch between the Top filter and all rows."""
    app.view_state["show_all"] = not app.view_state["show_all"]
    render_table(app)


def toggle_slim(app) -> None:
    """Switch between slim monitoring and full diagnostics."""
    app.view_state["slim"] = not app.view_state["slim"]
    render_dashboard(app)


def switch_gpus(app) -> None:
    """Leave the app so run_tui reopens it with the other GPU set."""
    app.view_state["switch_gpus"] = True
    quit_app(app)


def toggle_pause(app) -> None:
    """Freeze or resume sampling; the dashboard shows the paused mark at once."""
    app.view_state["paused"] = not app.view_state["paused"]
    render_dashboard(app)


def stop_sampling(app) -> None:
    """Wake the sampling thread so it ends without waiting a full interval."""
    app.view_state["stop_event"].set()


def quit_app(app) -> None:
    """Stop sampling and leave the app."""
    stop_sampling(app)
    app.exit(0)


def compose_chart_controls(view_state: dict) -> list:
    """Label, chart selector and hint of the charts panel."""
    shown_adapters = [block["adapter"] for block in view_state["blocks"]]
    selector = Select(
        tui_panels.list_chart_options(shown_adapters),
        allow_blank=False,
        value=tui_panels.CHART_ALL,
        compact=True,
        id="chart-mode",
    )
    return [
        Static(CHART_LABEL, id="chart-label"),
        selector,
        Static(CHART_HINT, id="chart-hint"),
    ]


def setup_widgets(app) -> None:
    """Border titles, table columns, the empty inspector and the initial focus."""
    view_state = app.view_state
    device_widget = app.query_one("#device", Static)
    device_widget.border_title = tui_panels.get_device_title(view_state["slim"])
    if view_state["show_system"]:
        system_widget = app.query_one("#system", Static)
        system_widget.border_title = tui_panels.SYSTEM_TITLE
    table = app.query_one("#procs", DataTable)
    for column in view_state["columns"]:
        table.add_column(tui_table.render_column_label(column), key=column["title"])
    mode = view_state["sort_modes"][view_state["sort_index"]]
    status_text = tui_table.format_table_status(mode, view_state["show_all"])
    status_line = app.query_one("#table-status", Static)
    status_line.update(status_text)
    render_inspector_panel(app)
    table.focus()


class HelpScreen(ModalScreen):
    """The help modal: one scrollable Markdown page."""

    CSS = HELP_CSS
    BINDINGS = [
        Binding("escape", "close_help", "Close"),
        Binding("q", "close_help", "Close"),
        Binding("question_mark", "close_help", "Close", key_display="?"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-body"):
            yield Markdown(tui_help.HELP_MARKDOWN)

    def action_close_help(self) -> None:
        self.dismiss()


class GpuTopApp(App):
    """The gputop terminal UI."""

    TITLE = APP_TITLE
    CSS = APP_CSS
    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("c", "cycle_chart", "Chart"),
        Binding("a", "toggle_filter", "All/Top"),
        Binding("m", "toggle_slim", "Slim/Full"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("y", "copy_inspector", "Copy"),
        Binding("g", "switch_gpus", "GPUs"),
        Binding("question_mark", "show_help", "Explain", key_display="?"),
    ]

    def __init__(self, view_state: dict) -> None:
        super().__init__()
        self.view_state = view_state

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dashboard"):
            yield Static(tui_panels.DEVICE_PLACEHOLDER, id="device")
            if self.view_state["hidden_note"]:
                yield Static(self.view_state["hidden_note"], id="hidden-note")
            if self.view_state["show_system"]:
                yield Static(tui_panels.SYSTEM_PLACEHOLDER, id="system")
            with Vertical(id="history"):
                with Horizontal(id="chart-controls"):
                    yield from compose_chart_controls(self.view_state)
                yield Static("", id="charts")
                yield Static("", id="chart-note")
        yield DataTable(id="procs", cursor_type="row", zebra_stripes=True)
        yield Static("", id="table-status")
        yield Static("", id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        setup_widgets(self)
        self.run_sampling()

    def on_unmount(self) -> None:
        stop_sampling(self)

    def on_resize(self, event) -> None:
        render_dashboard(self)
        render_inspector_panel(self)

    @work(thread=True, exclusive=True)
    def run_sampling(self) -> None:
        run_sampling_loop(self.view_state, self.deliver_refresh)

    def deliver_refresh(self, payload: dict) -> None:
        deliver_payload(self, payload)

    def apply_refresh(self, payload: dict) -> None:
        handle_refresh(self, payload)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        handle_row_highlighted(self, event)

    def on_select_changed(self, event: Select.Changed) -> None:
        handle_chart_changed(self, event.value)

    def on_click(self, event: Click) -> None:
        is_inspector = event.widget is not None and event.widget.id == "inspector"
        if is_inspector and event.chain == DOUBLE_CLICK:
            copy_inspector(self)

    def action_quit_app(self) -> None:
        quit_app(self)

    def action_cycle_sort(self) -> None:
        cycle_sort(self)

    def action_cycle_chart(self) -> None:
        cycle_chart(self)

    def action_toggle_filter(self) -> None:
        toggle_filter(self)

    def action_toggle_slim(self) -> None:
        toggle_slim(self)

    def action_toggle_pause(self) -> None:
        toggle_pause(self)

    def action_copy_inspector(self) -> None:
        copy_inspector(self)

    def action_switch_gpus(self) -> None:
        switch_gpus(self)

    def check_action(self, action: str, parameters: tuple) -> bool:
        if action == "switch_gpus":
            return self.view_state["can_switch_gpus"]
        return True

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())


def get_other_choice(gpu_choice: int | str | None) -> int | str | None:
    """The GPU set g switches to: every adapter from the default set, and back."""
    if gpu_choice is None:
        return constants.GPU_ALL
    return None


def can_switch_gpus(ordered_adapters: list[dict], gpu_choice: int | str | None) -> bool:
    """True when g changes the view: no single GPU was asked and one is hidden by default."""
    if gpu_choice not in (None, constants.GPU_ALL):
        return False
    return len(adapters.list_default_indexes(ordered_adapters)) < len(ordered_adapters)


def run_tui(gpu_choice: int | str | None, interval: float, show_system: bool) -> int:
    """Run the TUI for the asked GPU, for all of them, or for the default set.

    The shown GPU set is fixed for the life of one app, so g ends the app and this loop
    opens the next one with the other set; the view switches and the histories carry over.
    """
    ordered_adapters = adapters.list_ordered_adapters()
    carried = {}
    while True:
        shown_indexes = select_shown_indexes(ordered_adapters, gpu_choice)
        hidden_note = format_hidden_note(ordered_adapters, gpu_choice)
        view_state = create_view_state(
            ordered_adapters, shown_indexes, interval, show_system, hidden_note
        )
        view_state["can_switch_gpus"] = can_switch_gpus(ordered_adapters, gpu_choice)
        view_state.update(carried)
        app = GpuTopApp(view_state)
        app.run()
        if not view_state["switch_gpus"]:
            return app.return_code or 0
        gpu_choice = get_other_choice(gpu_choice)
        carried = {key: view_state[key] for key in CARRIED_KEYS}


def main():
    pass


if __name__ == "__main__":
    main()
