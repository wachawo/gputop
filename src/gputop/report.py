#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the plain-text report of the --once mode."""

# Local imports
from gputop import constants
from gputop import formatting
from gputop import modelfiles
from gputop import records
from gputop import system

# Column widths of the process table
PID_MIN_WIDTH = 7
NAME_WIDTH = 24
RUNTIME_WIDTH = 24

# Column width of the process name in the system table
SYSTEM_NAME_WIDTH = 28

# Fixed texts
SYSTEM_TITLE = "System RAM and CPU (OS-reported; not GPU memory)"
SYSTEM_TOP_TITLE = "Top processes by RAM, whole machine:"
SYSTEM_FAILED_PREFIX = "System RAM/CPU unavailable: "
PARENT_SOURCE_LINE = "   server source: parent process (API report, not worker residency)"
UNKNOWN_MARK = "?"


def render_identity_line(snapshot: dict) -> str:
    """Line 1: adapter name, driver, CUDA version and the driver model."""
    device = snapshot["device"]
    parts = [records.get_device_name(snapshot)]
    if snapshot["driver"]:
        parts.append(f"driver {snapshot['driver']}")
    if snapshot["cuda"]:
        parts.append(f"CUDA {snapshot['cuda']}")
    if device is not None and device["driver_model"]:
        parts.append(device["driver_model"])
    return "  ".join(parts)


def render_memory_line(snapshot: dict) -> str:
    """Line 2: device memory in use and in total, plus the shared figure on DRM."""
    used = formatting.format_size_or_na(snapshot["vram_used"])
    total = formatting.MISSING_SIZE
    if snapshot["vram_total"]:
        total = formatting.format_size(snapshot["vram_total"])
    line = f"VRAM {used} / {total}"
    if snapshot["backend"] == constants.BACKEND_DRM:
        shared = formatting.format_size_or_na(snapshot["shared_used"])
        line += f"   shared (in system RAM) {shared}"
    return line


def format_kind_suffix(kind: str) -> str:
    """The bracketed sensor label that follows a temperature or power figure."""
    if kind:
        return f" ({kind})"
    return ""


def render_telemetry_line(device: dict) -> str:
    """The device telemetry line."""
    util_gpu = formatting.format_metric(device["util_gpu"], ".0f", "%")
    util_mem = formatting.format_metric(device["util_mem"], ".0f", "%")
    temp = formatting.format_metric(device["temp"], ".0f", "C")
    power = formatting.format_metric(device["power_w"], ".0f")
    limit = formatting.format_metric(device["power_limit_w"], ".0f")
    pstate = formatting.format_metric(device["pstate"])
    clock_name, clock_value = records.get_main_clock(device)
    clock = formatting.format_metric(clock_value, ".0f")
    video_clock = formatting.format_metric(device["video_clock"], ".0f")
    pcie_rx = formatting.format_megabytes_per_second(device["pcie_rx_kbs"])
    pcie_tx = formatting.format_megabytes_per_second(device["pcie_tx_kbs"])
    fields = [
        f"{records.get_device_label(device)} {util_gpu}",
        f"mem-bandwidth {util_mem}",
        f"{temp}{format_kind_suffix(device['temp_kind'])}",
        f"{power}/{limit} W{format_kind_suffix(device['power_kind'])}",
        f"P{pstate}" if device["pstate"] is not None else f"P {pstate}",
        f"{clock_name} {clock} MHz",
        f"video {video_clock} MHz",
        f"PCIe rx {pcie_rx} MB/s tx {pcie_tx} MB/s",
    ]
    return "  ".join(fields)


def render_header(snapshot: dict) -> list[str]:
    """Header lines: identity, memory, notes and telemetry."""
    lines = [render_identity_line(snapshot), render_memory_line(snapshot)]
    memory_note = records.get_memory_note(snapshot)
    if memory_note:
        lines.append(memory_note)
    if snapshot["counter_status"]:
        lines.append(snapshot["counter_status"])
    if snapshot["device"] is not None:
        lines.append(render_telemetry_line(snapshot["device"]))
    return lines


def format_shared_cell(row: dict, is_drm: bool) -> str:
    """The SHARED cell: empty on NVML and for outside rows, a size or N/A on DRM."""
    if not is_drm or row["outside"]:
        return ""
    gpu = row["gpu"]
    if constants.COUNTER_SHARED not in gpu["available"]:
        return formatting.MISSING_SIZE
    if gpu["shared"] >= constants.MIB:
        return formatting.format_size(gpu["shared"])
    return ""


def format_load_cell(gpu: dict) -> str:
    """The GPU% cell: the busiest engine without decimals, or n/a."""
    load = records.get_engine_columns(gpu)["gpu"]
    if load is None:
        return formatting.MISSING_CELL
    return format(load, ".0f")


def format_table_line(cells: list[str], pid_width: int) -> str:
    """Lay out the eight cells of one table line."""
    pid, name, kind, runtime, vram, shared, load, inside = cells
    line = (
        f"{pid:>{pid_width}}  {name:<{NAME_WIDTH}} {kind:<10} {runtime:<{RUNTIME_WIDTH}} "
        f"{vram:>9} {shared:>8} {load:>5}  {inside}"
    )
    return line.rstrip()


def build_row_cells(row: dict, is_drm: bool) -> list[str]:
    """The eight cells of one process row."""
    runtime_label = records.get_runtime_label(row["detail"])
    return [
        records.get_pid_label(row),
        row["name"][:NAME_WIDTH],
        row["kind"],
        runtime_label[:RUNTIME_WIDTH],
        records.format_vram_label(row["gpu"]),
        format_shared_cell(row, is_drm),
        format_load_cell(row["gpu"]),
        row["inside"],
    ]


def list_reason_notes(shown: list[dict]) -> list[str]:
    """Explanations of the distinct DRM reasons among shown local rows, first seen first."""
    seen = []
    for row in shown:
        reason = row["gpu"]["drm_reason"]
        if row["outside"] or not reason or reason in seen:
            continue
        seen.append(reason)
    return [constants.DRM_REASON_NOTES[reason] for reason in seen
            if reason in constants.DRM_REASON_NOTES]


def render_table_notes(snapshot: dict, shown: list[dict]) -> list[str]:
    """The lines under the process table."""
    lines = []
    marks = {row["gpu"]["inferred_by"] for row in shown}
    for inferred_by in (constants.INFERRED_CONSISTENT, constants.INFERRED_ORDER):
        if inferred_by in marks:
            lines.append(constants.INFERRED_NOTES[inferred_by])
    if snapshot["backend"] == constants.BACKEND_DRM:
        lines.extend(list_reason_notes(shown))
    if snapshot["process_status"]:
        lines.append(f"DRM process scan: {snapshot['process_status']}.")
    hidden = len(snapshot["procs"]) - len(shown)
    if hidden > 0:
        lines.append(f"{hidden} more process(es) are not shown; --json lists all.")
    return lines


def render_process_table(snapshot: dict) -> list[str]:
    """Header, rows and notes of the process table."""
    shown = snapshot["procs"][:constants.REPORT_MAX_ROWS]
    is_drm = snapshot["backend"] == constants.BACKEND_DRM
    label_widths = [len(records.get_pid_label(row)) for row in shown]
    pid_width = max([PID_MIN_WIDTH] + label_widths)
    titles = ["PID", "PROCESS", "KIND", "RUNTIME", "VRAM", "SHARED", "GPU%", "INSIDE"]
    lines = [format_table_line(titles, pid_width)]
    for row in shown:
        lines.append(format_table_line(build_row_cells(row, is_drm), pid_width))
    lines.extend(render_table_notes(snapshot, shown))
    return lines


def render_block_title(row: dict) -> str:
    """The heading of one per-process detail block."""
    vram = records.format_vram_label(row["gpu"])
    return f"== {row['name']} ({records.get_pid_label(row)})  VRAM {vram}"


def render_file_line(model: dict) -> str:
    """One `file:` line of a detail block."""
    size = formatting.format_size(model["size"])
    if model["kind"]:
        size = f"{size}, {model['kind']}"
    evidence = modelfiles.describe_evidence(model["sources"])
    return f"   file: {modelfiles.get_display_name(model)}  ({size})  evidence: {evidence}"


def render_model_line(model: dict) -> str:
    """One parsed model of an LLM view."""
    ctx_train = model["ctx_train"] if model["ctx_train"] is not None else UNKNOWN_MARK
    layers = f"{model['n_layers']} layers, " if model["n_layers"] else ""
    properties = (
        f"{model['arch']}, {model['params_label']} params, {model['quant']}, "
        f"{layers}ctx_train {ctx_train}"
    )
    return (
        f"   {model['kind_label']}: {model['label']}  [{properties}]  "
        f"{model['file_name']}  evidence: {model['evidence']}"
    )


def format_or_unknown(value) -> str:
    """A server figure, or the unknown mark when the server did not report it."""
    if value is None:
        return UNKNOWN_MARK
    return str(value)


def render_server_lines(view: dict) -> list[str]:
    """The server line of an LLM view and the lines that depend on it."""
    server = view["server"]
    if server is None:
        return []
    slots = f"{format_or_unknown(server['slots_busy'])}/{format_or_unknown(server['slots_total'])}"
    line = (
        f"   server: {server['kind']} {server['url']} "
        f"ctx={format_or_unknown(server['n_ctx'])} slots={slots}"
    )
    if server["gen_tok_s"] is not None:
        line += f" gen={server['gen_tok_s']:.1f} tok/s"
    lines = [line]
    if view["server_source"] == constants.SERVER_SOURCE_PARENT:
        lines.append(PARENT_SOURCE_LINE)
    if server["kind"] == constants.SERVER_VLLM:
        for reported in server["models"]:
            lines.append(f"   API-reported model ID: {reported['name']}")
    return lines


def render_llm_block(row: dict) -> list[str]:
    """The detail block of a row that has an LLM view."""
    view = row["llm"]
    lines = ["", render_block_title(row)]
    lines.extend(render_model_line(model) for model in view["models"])
    lines.extend(render_file_line(model) for model in view["other_files"])
    for segment in view["segments"]:
        lines.append(f"   {formatting.format_size(segment['bytes']):>9}  {segment['label']}")
    lines.extend(render_server_lines(view))
    if view["log_source"]:
        lines.append(f"   exact sizes from log: {view['log_source']}")
    lines.extend(f"   - {note}" for note in view["notes"])
    return lines


def render_hint_block(row: dict) -> list[str]:
    """The detail block of a row without an LLM view: its path-argument hints, if any."""
    hinted = [
        model for model in row["detail"]["models"]
        if constants.EVIDENCE_PATH_ARG in model["sources"]
    ]
    if not hinted:
        return []
    lines = ["", render_block_title(row)]
    lines.extend(render_file_line(model) for model in hinted)
    return lines


def render_detail_blocks(snapshot: dict) -> list[str]:
    """Detail blocks of the rows the table shows, in snapshot order."""
    lines = []
    for row in snapshot["procs"][:constants.REPORT_MAX_ROWS]:
        if row["llm"] is not None:
            lines.extend(render_llm_block(row))
        else:
            lines.extend(render_hint_block(row))
    return lines


def render_system_proc_line(proc: dict) -> str:
    """One row of the system process table."""
    cpu = formatting.MISSING_CELL
    if proc["cpu_pct"] is not None:
        cpu = format(proc["cpu_pct"], ".0f")
    name = proc["name"][:SYSTEM_NAME_WIDTH]
    rss = formatting.format_size(proc["rss"])
    return f"{proc['pid']:>7}  {name:<{SYSTEM_NAME_WIDTH}} {rss:>9} {cpu:>5}"


def render_system_section(system_sample: dict) -> list[str]:
    """The system section for a sample that was taken."""
    cpu_line = f"CPU {system_sample['cpu_pct']:.0f}%  {system_sample['logical']} threads"
    if system_sample["physical"]:
        cpu_line += f" / {system_sample['physical']} cores"
    ram_line = (
        f"RAM {formatting.format_size(system_sample['ram_used'])} / "
        f"{formatting.format_size(system_sample['ram_total'])}  "
        f"available {formatting.format_size(system_sample['ram_available'])}   "
        f"swap {formatting.format_size(system_sample['swap_used'])} / "
        f"{formatting.format_size(system_sample['swap_total'])}"
    )
    header = f"{'PID':>7}  {'PROCESS':<{SYSTEM_NAME_WIDTH}} {'RAM':>9} {'CPU%':>5}"
    lines = [SYSTEM_TITLE, cpu_line, ram_line, "", SYSTEM_TOP_TITLE, header]
    top_procs = system.get_top_procs(system_sample, "rss", constants.SYSTEM_TOP_ROWS)
    lines.extend(render_system_proc_line(proc) for proc in top_procs)
    if system_sample["denied"] > 0:
        lines.append(f"({system_sample['denied']} processes not readable without root)")
    return lines


def render_report(
    snapshot: dict,
    system_sample: dict | None = None,
    system_error: str = "",
    include_system: bool = True,
) -> str:
    """The --once text for one snapshot, without a trailing newline.

    With `include_system` true the system section follows after a blank line: the
    sample's table, or one line with `system_error` when there is no sample.
    """
    lines = render_header(snapshot)
    lines.append("")
    lines.extend(render_process_table(snapshot))
    lines.extend(render_detail_blocks(snapshot))
    if include_system and system_sample is not None:
        lines.append("")
        lines.extend(render_system_section(system_sample))
    elif include_system and system_error:
        lines.append("")
        lines.append(f"{SYSTEM_FAILED_PREFIX}{system_error}")
    return "\n".join(line.rstrip() for line in lines)


def main():
    pass


if __name__ == "__main__":
    main()
