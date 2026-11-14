#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspector panel renderer of the gputop TUI: what one process keeps in GPU memory."""

import os

from rich.console import Group
from rich.table import Table
from rich.text import Text

# Local imports
from gputop import cmdline
from gputop import constants
from gputop import formatting
from gputop import modelfiles
from gputop import records
from gputop import tui_draw

# Layout
BAR_WIDTH_MARGIN = 8
BAR_WIDTH_MIN = 20
BAR_WIDTH_MAX = 100
MAX_PORTS = 8
MAX_CMDLINE_CHARS = 400
MAX_SERVER_MODELS = 4
ENGINE_MIN_PCT = 0.5

# Styles
TEXT_STYLE = "grey70"
PORT_STYLE = "grey50"
CMDLINE_STYLE = "grey42"

# Texts
EMPTY_HINT = (
    "Select a process (↑/↓) to see what's inside its GPU memory.  "
    "Press ? to learn what's knowable."
)
MULTI_GPU_CAVEAT = (
    "Model files, launch flags and runtime buffer logs can span GPUs. "
    "The breakdown below is not a measured per-device model placement."
)
PARENT_SERVER_NOTE = " (parent process; API report, not worker residency)"
CANDIDATES_NOTE = (
    "No mapped model file was seen in this process. These models are installed next to "
    "its runtime, but nothing proves that one of them is loaded:"
)
PROTECTED_NOTE = "(protected process - run as root for runtime/file details)"
KIND_PARAGRAPHS = {
    constants.KIND_COMPOSITOR: (
        "A display server or window manager keeps its own back buffers here, plus the "
        "composed frame of every output it drives."
    ),
    constants.KIND_BROWSER: (
        "A browser GPU process holds page tiles, decoded images, video frames and the "
        "buffers of canvas, WebGL and WebGPU content. The browser's own GPU diagnostics "
        "page lists them in more detail."
    ),
    constants.KIND_GAME: (
        "A 3D application fills its memory with textures, meshes and render targets. "
        "Naming single resources takes a graphics debugger attached to the application."
    ),
    constants.KIND_VIDEO: (
        "A video workload keeps decoded and encoded frames and the codec surfaces of the "
        "hardware decoder and encoder (NVDEC / NVENC)."
    ),
    constants.KIND_COMPUTE: (
        "Compute allocations are opaque from outside the process. A profiler of the "
        "runtime in use is needed to see what the buffers are."
    ),
    constants.KIND_DIFFUSION: (
        "The components of an image-generation pipeline are recognised from the launch "
        "arguments. The GPU allocation itself cannot be split per component."
    ),
    constants.KIND_AI: (
        "A machine-learning runtime holds tensors and workspaces here. The memory summary "
        "of the framework inside the process is the tool that goes deeper."
    ),
    constants.KIND_APP: (
        "A desktop application keeps UI surfaces, swap chains and cached bitmaps and "
        "glyphs in GPU memory."
    ),
    constants.KIND_EXTERNAL: (
        "This process lives outside this PID namespace (on the host or in another "
        "container), so no local details can be read for it."
    ),
}


def format_inspector_title(row: dict | None, multi_gpu: bool) -> str:
    """Border title of the inspector for the selected row; empty without a selection."""
    if row is None:
        return ""
    pid_part = f"host PID {row['pid']}" if row["outside"] else str(row["pid"])
    title = f"{row['name']} ({pid_part}) - {row['kind']}"
    if multi_gpu and row["gpu_index"] is not None:
        title += f" - GPU {row['gpu_index']}"
    return title


def format_engines(gpu: dict) -> str:
    """Busy engines of a process, idle when none is busy, n/a without the counter."""
    if constants.COUNTER_ENGINE not in gpu["available"]:
        return formatting.MISSING_CELL
    busy = [f"{name} {load:.0f}%" for name, load in gpu["engines"].items() if load >= ENGINE_MIN_PCT]
    return " ".join(busy) if busy else "idle"


def append_shared_part(line: Text, row: dict, snapshot: dict) -> None:
    """Add the system-RAM part of the header; drawn for rows of a DRM snapshot only."""
    if snapshot["backend"] != constants.BACKEND_DRM:
        return
    line.append("   shared(RAM) ")
    gpu = row["gpu"]
    if constants.COUNTER_SHARED not in gpu["available"]:
        line.append(formatting.MISSING_CELL)
        return
    style = "bold red" if records.is_spill(row, snapshot) else ""
    line.append(formatting.format_size(gpu["shared"]), style=style)


def render_header_line(row: dict, snapshot: dict) -> Text:
    """Line 1: VRAM, shared memory, engines and runtime labels."""
    gpu = row["gpu"]
    line = Text()
    line.append("VRAM ", style="bold")
    line.append(records.format_vram_label(gpu), style="bold")
    append_shared_part(line, row, snapshot)
    line.append("   engines ")
    line.append(format_engines(gpu))
    runtime_labels = row["detail"]["runtimes"]
    if runtime_labels:
        line.append("   runtime ", style=constants.MUTED_STYLE)
        line.append(" ".join(runtime_labels), style="cyan")
    return line


def list_resident_notes(gpu: dict) -> list[str]:
    """The resident-memory line of a DRM row; empty when neither figure is known."""
    if gpu["resident_vram"] is None and gpu["resident_gtt"] is None:
        return []
    resident_vram = formatting.format_size_or_na(gpu["resident_vram"], formatting.MISSING_CELL)
    resident_gtt = formatting.format_size_or_na(gpu["resident_gtt"], formatting.MISSING_CELL)
    return [f"Resident VRAM {resident_vram}, GTT {resident_gtt}. {constants.RESIDENT_NOTE}"]


def list_explanations(row: dict, snapshot: dict) -> list[str]:
    """Explanation lines under the header: inferred owner, DRM reason, resident, KFD."""
    gpu = row["gpu"]
    notes = []
    if not row["outside"] and gpu["inferred_by"]:
        notes.append(constants.INFERRED_NOTES.get(gpu["inferred_by"], ""))
    if snapshot["backend"] == constants.BACKEND_DRM:
        if not row["outside"] and not gpu["attributed"] and gpu["drm_reason"]:
            notes.append(constants.DRM_REASON_NOTES.get(gpu["drm_reason"], ""))
        notes.extend(list_resident_notes(gpu))
        if gpu["kfd"]:
            notes.append(constants.KFD_NOTE)
    return [note for note in notes if note]


def get_bar_width(width: int) -> int:
    """Width of the breakdown bar for a terminal width."""
    return int(tui_draw.clamp_number(width - BAR_WIDTH_MARGIN, BAR_WIDTH_MIN, BAR_WIDTH_MAX))


def get_segment_style(segment: dict) -> str:
    """Colour of one breakdown segment."""
    return constants.SEGMENT_STYLES.get(segment["kind"], constants.KIND_FALLBACK_STYLE)


def render_breakdown_bar(view: dict, gpu: dict, width: int) -> Text:
    """Stacked bar of the segments; an unexplained remainder stays as the empty tail."""
    segments = view["segments"]
    parts = [(segment["bytes"], get_segment_style(segment)) for segment in segments]
    explained = sum(segment["bytes"] for segment in segments)
    total = max(records.get_shown_vram(gpu) or 0, explained, 1)
    return tui_draw.draw_stacked_bar(parts, total, get_bar_width(width))


def render_breakdown_legend(view: dict) -> Text:
    """One legend entry per segment; sizes taken from a log or an API are bold."""
    line = Text()
    for segment in view["segments"]:
        line.append(tui_draw.LEGEND_SQUARE, style=get_segment_style(segment))
        label = f"{segment['label']} {formatting.format_size(segment['bytes'])}"
        line.append(label, style="bold" if segment["exact"] else "")
        line.append("  ")
    if view["log_source"]:
        log_name = os.path.basename(view["log_source"])
        line.append(f"(allocated sizes from {log_name})", style="green")
    return line


def create_model_table() -> Table:
    """The borderless model table with its ten columns."""
    table = Table(box=None, padding=(0, 1), show_edge=False, header_style="bold grey70")
    table.add_column("", style="magenta")
    for header in ("model", "arch", "params", "quant", "layers", "train ctx", "on GPU (est)"):
        table.add_column(header)
    table.add_column("evidence", style="green")
    table.add_column("file", style="grey50", no_wrap=True, overflow="ellipsis")
    return table


def render_model_table(view: dict) -> Table:
    """One table row per parsed model of the view."""
    table = create_model_table()
    for model in view["models"]:
        table.add_row(
            model["kind_label"],
            model["label"],
            model["arch"],
            model["params_label"],
            model["quant"],
            str(model["n_layers"]) if model["n_layers"] else "",
            f"{model['ctx_train']:,}" if model["ctx_train"] else "",
            formatting.format_size(model["gpu_weight_bytes"]),
            model["evidence"],
            model["file_name"],
        )
    return table


def format_file_line(model: dict) -> str:
    """The "file:" line of one model file: name, size, evidence and kind."""
    details = [formatting.format_size(model["size"])]
    evidence = modelfiles.describe_evidence(model["sources"])
    if evidence:
        details.append(evidence)
    if model["kind"]:
        details.append(model["kind"])
    return f"file: {modelfiles.get_display_name(model)}  ({', '.join(details)})"


def format_candidate_line(model: dict) -> str:
    """One installed, unproven model: name, size and kind."""
    details = [formatting.format_size(model["size"])]
    if model["kind"]:
        details.append(model["kind"])
    return f"  {modelfiles.get_display_name(model)}  ({', '.join(details)})"


def render_server_line(view: dict) -> Text:
    """The probed local inference server: kind, address, context, slots and speed."""
    server = view["server"]
    line = Text()
    line.append("server ", style="bold")
    line.append(f"{server['kind']} {server['url']}", style="cyan")
    if view["server_source"] == constants.SERVER_SOURCE_PARENT:
        line.append(PARENT_SERVER_NOTE)
    if server["n_ctx"]:
        line.append(f"  ctx {server['n_ctx']:,}")
    if server["slots_total"] is not None:
        busy = "?" if server["slots_busy"] is None else str(server["slots_busy"])
        line.append(f"  slots busy {busy}/{server['slots_total']}")
    if server["gen_tok_s"] is not None:
        line.append(f"  gen {server['gen_tok_s']:.1f} tok/s", style="bold green")
    if server["prompt_tok_s"] is not None:
        line.append(f"  prompt {server['prompt_tok_s']:.0f} tok/s")
    return line


def format_server_model(server: dict, model: dict) -> str:
    """Sub-line of one server-reported model; empty for server kinds that list none."""
    if server["kind"] == constants.SERVER_LMSTUDIO:
        known = [model["name"], model["type"], model["quant"]]
        described = " ".join(str(value) for value in known if value)
        context = "?" if model["ctx"] is None else str(model["ctx"])
        return f"  loaded: {described} ctx {context}"
    if server["kind"] == constants.SERVER_VLLM:
        return f"  API-reported model ID: {model['name']}"
    return ""


def render_server_lines(view: dict) -> list[Text]:
    """The server line and up to four model sub-lines; empty without a server."""
    server = view["server"]
    if server is None:
        return []
    lines = [render_server_line(view)]
    for model in server["models"][:MAX_SERVER_MODELS]:
        described = format_server_model(server, model)
        if described:
            lines.append(Text(described))
    return lines


def render_note_line(note: str) -> Text:
    """One note of the LLM view; a note starting with "!" is a warning."""
    style = "bold red" if note.startswith("!") else TEXT_STYLE
    return Text(f"- {note}", style=style)


def render_llm_body(row: dict, width: int) -> list:
    """Body of a row with an LLM view: bar, legend, models, files, server, notes."""
    view = row["llm"]
    body = []
    if row["gpu_index"] is not None:
        body.append(Text(MULTI_GPU_CAVEAT, style=TEXT_STYLE))
    body.append(render_breakdown_bar(view, row["gpu"], width))
    body.append(render_breakdown_legend(view))
    if view["models"]:
        body.append(render_model_table(view))
    for model in view["other_files"]:
        body.append(Text(format_file_line(model), style=constants.MUTED_STYLE))
    body.extend(render_server_lines(view))
    body.extend(render_note_line(note) for note in view["notes"])
    return body


def render_plain_body(row: dict) -> list:
    """Body of a row without an LLM view: kind paragraph, model files, candidates."""
    detail = row["detail"]
    body = []
    paragraph = KIND_PARAGRAPHS.get(row["kind"], "")
    if paragraph:
        body.append(Text(paragraph, style=TEXT_STYLE))
    for model in detail["models"]:
        body.append(Text(format_file_line(model), style="magenta"))
    if detail["candidates"]:
        body.append(Text(CANDIDATES_NOTE, style=TEXT_STYLE))
        for model in detail["candidates"]:
            body.append(Text(format_candidate_line(model), style=constants.MUTED_STYLE))
    return body


def render_tail(detail: dict) -> list:
    """Common tail: listening ports, then the redacted command line or the protected note."""
    tail = []
    if detail["ports"]:
        ports = ", ".join(str(port) for port in detail["ports"][:MAX_PORTS])
        tail.append(Text(f"listening: {ports}", style=PORT_STYLE))
    if detail["cmdline"]:
        redacted = cmdline.redact_cmdline(detail["cmdline"])
        joined = " ".join(redacted)[:MAX_CMDLINE_CHARS]
        tail.append(Text(joined, style=CMDLINE_STYLE, overflow="ellipsis", no_wrap=True))
    elif detail["access_denied"]:
        tail.append(Text(PROTECTED_NOTE, style=CMDLINE_STYLE))
    return tail


def render_inspector(row: dict | None, snapshot: dict | None, width: int) -> Text | Group:
    """The inspector body for the selected row, or the hint when nothing is selected."""
    if row is None or snapshot is None:
        return Text(EMPTY_HINT, style=constants.MUTED_STYLE)
    parts = [render_header_line(row, snapshot)]
    for note in list_explanations(row, snapshot):
        parts.append(Text(note, style=constants.MUTED_STYLE))
    if row["llm"] is not None:
        parts.extend(render_llm_body(row, width))
    else:
        parts.extend(render_plain_body(row))
    parts.extend(render_tail(row["detail"]))
    return Group(*parts)


def main():
    pass


if __name__ == "__main__":
    main()
