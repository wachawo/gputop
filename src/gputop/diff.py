#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The `gputop diff` subcommand: compare two --json snapshots."""

import argparse
import codecs
import collections
import json
import math
import sys

# Local imports
from gputop import constants
from gputop import formatting

# Parser texts
DIFF_PROG = "gputop diff"
DIFF_DESCRIPTION = "Compare two --json snapshots, for example before and after loading a model."

# Match kinds
MATCH_PID = "pid"
MATCH_NAME_PORTS = "name+ports"
MATCH_NAME = "name"
MATCH_NEW = "new"
MATCH_GONE = "gone"

# Adapter check results
CHECK_KEY = "key"
CHECK_NAME_ONLY = "name-only"
NAME_ONLY_NOTE = (
    "These snapshots do not both say which adapter they came from (an older gputop, or no "
    "adapter details), so they are matched by GPU name and backend only."
)

# Compared keys
MEASURED_KEYS = ("vram_bytes", "shared_bytes", "vram_resident_bytes", "gtt_resident_bytes")
DEVICE_ROWS = (
    ("dedicated used", "vram_used"),
    ("shared used", "shared_used"),
    ("attributed to processes", "attributed"),
    ("unattributed", "unattributed"),
)
SOURCE_REPORTED = "runtime-reported"
SOURCE_ESTIMATED = "estimated"
PLACEHOLDER_NAMES = (constants.UNKNOWN_NAME, constants.OUTSIDE_NAME)

# Text form
MISSING = "N/A"
NOT_LISTED = "not listed"
HALF_MIB = constants.MIB / 2
INTRO_LINES = (
    "Measured counters are compared directly. Runtime-reported and estimated values are listed",
    "separately and are not added to measured totals.",
)
NO_CHANGE_LINE = "(no process changed by 16 MB or more)"
FILES_LINE = "Model files show what a process opened or named, not what is resident in GPU memory."
NOT_LISTED_LINE = (
    "'not listed' means the process was absent from that snapshot: not running, or using too "
    "little GPU to be listed."
)


def is_number(value) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_known_id(value) -> bool:
    """True for a non-zero integer that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and value != 0


def freeze_value(value):
    """A hashable stand-in for a JSON value used inside a pairing key."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def load_snapshot(path: str) -> dict:
    """Read one --json snapshot file; raises OSError or ValueError."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    document = json.loads(text)
    if not isinstance(document, dict) or not isinstance(document.get("processes"), list):
        raise ValueError(f"{path} is not a gputop --json snapshot")
    return document


def create_triple(before, after) -> dict:
    """A change triple; values that are not numbers read as unknown."""
    before_value = before if is_number(before) else None
    after_value = after if is_number(after) else None
    delta = None
    if before_value is not None and after_value is not None:
        delta = after_value - before_value
    return {"before": before_value, "after": after_value, "delta": delta}


def get_adapter_fact(snapshot: dict, key: str) -> int | None:
    """A well-formed, non-zero adapter identifier of a snapshot, or None."""
    adapter = snapshot.get("adapter")
    if not isinstance(adapter, dict):
        return None
    value = adapter.get(key)
    return value if is_known_id(value) else None


def check_same_gpu(before: dict, after: dict) -> str:
    """Verify both snapshots describe one adapter; returns the adapter check result."""
    names = (before.get("gpu"), after.get("gpu"))
    backends = (before.get("backend"), after.get("backend"))
    if names[0] != names[1] or backends[0] != backends[1]:
        raise ValueError(
            "snapshots are from different GPUs or backends: "
            f"{names[0]} ({backends[0]}) vs {names[1]} ({backends[1]})"
        )
    prefix = f"snapshots are from different adapters, though both are named {names[1]!r}: "
    vendor_before = get_adapter_fact(before, "vendor_id")
    vendor_after = get_adapter_fact(after, "vendor_id")
    if vendor_before and vendor_after and vendor_before != vendor_after:
        raise ValueError(f"{prefix}vendor 0x{vendor_before:04x} vs 0x{vendor_after:04x}")
    key_before = get_adapter_fact(before, "key")
    key_after = get_adapter_fact(after, "key")
    if key_before is None or key_after is None:
        return CHECK_NAME_ONLY
    if key_before != key_after:
        raise ValueError(f"{prefix}adapter key 0x{key_before:x} vs 0x{key_after:x}")
    return CHECK_KEY


def get_start_time(row: dict) -> float | None:
    """The start time of a row when it is a finite positive number."""
    value = row.get("started_at")
    if is_number(value) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def is_recycled(before_row: dict, after_row: dict) -> bool:
    """True when both rows carry one pid but are two different processes."""
    pid = before_row.get("pid")
    if pid is None or pid != after_row.get("pid"):
        return False
    started_before = get_start_time(before_row)
    started_after = get_start_time(after_row)
    if started_before is None or started_after is None:
        return False
    return abs(started_after - started_before) > constants.PID_REUSE_TOLERANCE


def is_named_row(row: dict) -> bool:
    """True when the row may be paired by its name."""
    name = row.get("name")
    if row.get("host") == constants.HOST_OUTSIDE:
        return False
    return isinstance(name, str) and bool(name) and name not in PLACEHOLDER_NAMES


def build_pid_key(row: dict) -> tuple:
    """Pairing key of pass 1."""
    return (freeze_value(row.get("host")), freeze_value(row.get("pid")))


def build_name_ports_key(row: dict) -> tuple | None:
    """Pairing key of pass 2, or None when the row has no ports or no real name."""
    ports = row.get("ports")
    if not is_named_row(row) or not isinstance(ports, list) or not ports:
        return None
    port_labels = tuple(sorted(str(port) for port in ports))
    return (freeze_value(row.get("host")), row["name"], port_labels)


def build_name_key(row: dict) -> tuple | None:
    """Pairing key of pass 3, or None when the row has no real name."""
    if not is_named_row(row):
        return None
    return (freeze_value(row.get("host")), row["name"])


def pair_by_pid(before_rows: list[dict], after_rows: list[dict]) -> tuple:
    """Pass 1: returns (pairs, unpaired before rows, unpaired after rows)."""
    keys = [build_pid_key(row) for row in before_rows]
    counts = collections.Counter(keys)
    candidates = {}
    for index, key in enumerate(keys):
        if counts[key] == 1 and before_rows[index].get("pid") is not None:
            candidates[key] = index
    pairs = []
    paired = set()
    after_left = []
    for after_row in after_rows:
        index = candidates.get(build_pid_key(after_row))
        fits = index is not None and index not in paired
        if fits and before_rows[index].get("name") == after_row.get("name") \
                and not is_recycled(before_rows[index], after_row):
            pairs.append((before_rows[index], after_row, MATCH_PID))
            paired.add(index)
        else:
            after_left.append(after_row)
    before_left = [row for index, row in enumerate(before_rows) if index not in paired]
    return pairs, before_left, after_left


def group_rows(rows: list[dict], build_key) -> dict:
    """Pairing key to the list of row indexes that carry it."""
    groups = {}
    for index, row in enumerate(rows):
        key = build_key(row)
        if key is not None:
            indexes = groups.setdefault(key, [])
            indexes.append(index)
    return groups


def pair_by_key(before_rows: list[dict], after_rows: list[dict], build_key, label: str) -> tuple:
    """Passes 2 and 3: pair keys held by exactly one row on each side."""
    before_groups = group_rows(before_rows, build_key)
    after_groups = group_rows(after_rows, build_key)
    pairs = []
    used_before = set()
    used_after = set()
    for key, after_indexes in after_groups.items():
        before_indexes = before_groups.get(key, [])
        if len(after_indexes) != 1 or len(before_indexes) != 1:
            continue
        before_row = before_rows[before_indexes[0]]
        after_row = after_rows[after_indexes[0]]
        if is_recycled(before_row, after_row):
            continue
        pairs.append((before_row, after_row, label))
        used_before.add(before_indexes[0])
        used_after.add(after_indexes[0])
    before_left = [row for index, row in enumerate(before_rows) if index not in used_before]
    after_left = [row for index, row in enumerate(after_rows) if index not in used_after]
    return pairs, before_left, after_left


def match_processes(before_rows: list[dict], after_rows: list[dict]) -> list[tuple]:
    """Pair the rows of two snapshots; every row appears exactly once."""
    matched, before_left, after_left = pair_by_pid(before_rows, after_rows)
    for build_key, label in ((build_name_ports_key, MATCH_NAME_PORTS),
                             (build_name_key, MATCH_NAME)):
        pairs, before_left, after_left = pair_by_key(before_left, after_left, build_key, label)
        matched.extend(pairs)
    matched.extend((None, after_row, MATCH_NEW) for after_row in after_left)
    matched.extend((before_row, None, MATCH_GONE) for before_row in before_left)
    return matched


def summarize_segments(view) -> dict:
    """Segment kind to {"bytes", "exact"} for one `llm` object; {} when there is none."""
    totals = {}
    segments = view.get("segments") if isinstance(view, dict) else None
    if not isinstance(segments, list):
        return totals
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("kind")
        if not isinstance(kind, str) or not kind:
            kind = constants.SEGMENT_UNKNOWN
        size = segment.get("bytes")
        entry = totals.setdefault(kind, {"bytes": 0, "exact": True})
        entry["bytes"] += size if is_number(size) else 0
        entry["exact"] = entry["exact"] and bool(segment.get("exact"))
    return totals


def compare_segments(before_view, after_view) -> dict:
    """Change triple and source per segment kind present on either side."""
    before_totals = summarize_segments(before_view)
    after_totals = summarize_segments(after_view)
    present = list(before_totals) + list(after_totals)
    extra_kinds = [kind for kind in present if kind not in constants.SEGMENT_ORDER]
    ordered_kinds = list(constants.SEGMENT_ORDER) + list(dict.fromkeys(extra_kinds))
    compared = {}
    for kind in ordered_kinds:
        sides = [totals[kind] for totals in (before_totals, after_totals) if kind in totals]
        if not sides:
            continue
        before_entry = before_totals.get(kind)
        after_entry = after_totals.get(kind)
        entry = create_triple(
            before_entry["bytes"] if before_entry else None,
            after_entry["bytes"] if after_entry else None,
        )
        exact = all(side["exact"] for side in sides)
        entry["source"] = SOURCE_REPORTED if exact else SOURCE_ESTIMATED
        compared[kind] = entry
    return compared


def compare_runtime(before_row: dict, after_row: dict) -> dict | None:
    """The runtime block of one pair; None when neither row has an `llm` object."""
    before_view = before_row.get("llm")
    after_view = after_row.get("llm")
    if not isinstance(before_view, dict) and not isinstance(after_view, dict):
        return None
    before_ctx = before_view.get("n_ctx") if isinstance(before_view, dict) else None
    after_ctx = after_view.get("n_ctx") if isinstance(after_view, dict) else None
    return {
        "n_ctx": create_triple(before_ctx, after_ctx),
        "segments": compare_segments(before_view, after_view),
    }


def list_model_files(row: dict) -> list[dict]:
    """The `{path, sources}` entries of one row, first occurrence of each path."""
    files = row.get("model_files")
    entries = {}
    if not isinstance(files, list):
        return []
    for model in files:
        if not isinstance(model, dict):
            continue
        path = freeze_value(model.get("path"))
        sources = model.get("sources")
        entry = {"path": path, "sources": list(sources) if isinstance(sources, list) else []}
        entries.setdefault(path, entry)
    return list(entries.values())


def compare_model_files(before_row: dict, after_row: dict) -> dict:
    """Model files added and removed between the two rows, compared by path."""
    before_files = list_model_files(before_row)
    after_files = list_model_files(after_row)
    before_paths = {entry["path"] for entry in before_files}
    after_paths = {entry["path"] for entry in after_files}
    return {
        "added": [entry for entry in after_files if entry["path"] not in before_paths],
        "removed": [entry for entry in before_files if entry["path"] not in after_paths],
    }


def build_process_entry(before_row: dict | None, after_row: dict | None, match: str) -> dict:
    """The per-process object of one pair."""
    before_side = before_row if before_row is not None else {}
    after_side = after_row if after_row is not None else {}
    named_side = after_side if after_row is not None else before_side
    measured = {}
    for key in MEASURED_KEYS:
        measured[key] = create_triple(before_side.get(key), after_side.get(key))
    return {
        "name": named_side.get("name"),
        "host": named_side.get("host"),
        "match": match,
        "pid_before": before_side.get("pid"),
        "pid_after": after_side.get("pid"),
        "measured": measured,
        "runtime": compare_runtime(before_side, after_side),
        "model_files": compare_model_files(before_side, after_side),
    }


def get_size_key(entry: dict) -> float:
    """Sort key of a per-process object: the size of its dedicated-memory change."""
    triple = entry["measured"]["vram_bytes"]
    if triple["delta"] is not None:
        return abs(triple["delta"])
    if triple["after"] is not None:
        return triple["after"]
    if triple["before"] is not None:
        return triple["before"]
    return 0


def compute_unattributed(snapshot: dict):
    """Device memory in use that no process row accounts for, or None."""
    used = snapshot.get("vram_used")
    attributed = snapshot.get("attributed")
    if is_number(used) and is_number(attributed):
        return used - attributed
    return None


def compare_device(before: dict, after: dict) -> dict:
    """Change triples of the device-wide counters."""
    compared = {}
    for key in ("vram_used", "shared_used", "attributed"):
        compared[key] = create_triple(before.get(key), after.get(key))
    compared["unattributed"] = create_triple(
        compute_unattributed(before), compute_unattributed(after)
    )
    return compared


def compare_snapshots(before: dict, after: dict) -> dict:
    """The comparison result of two snapshots; raises ValueError for different adapters."""
    adapter_check = check_same_gpu(before, after)
    seconds = None
    if is_number(before.get("taken")) and is_number(after.get("taken")):
        seconds = round(after["taken"] - before["taken"], 1)
    before_rows = [row for row in before["processes"] if isinstance(row, dict)]
    after_rows = [row for row in after["processes"] if isinstance(row, dict)]
    entries = []
    for before_row, after_row, match in match_processes(before_rows, after_rows):
        entries.append(build_process_entry(before_row, after_row, match))
    entries.sort(key=get_size_key, reverse=True)
    return {
        "gpu": after.get("gpu"),
        "backend": after.get("backend"),
        "adapter_check": adapter_check,
        "adapter_note": NAME_ONLY_NOTE if adapter_check == CHECK_NAME_ONLY else None,
        "seconds": seconds,
        "memory_note": after.get("memory_note") or before.get("memory_note") or None,
        "device": compare_device(before, after),
        "processes": entries,
    }


def format_plain_size(value) -> str:
    """A size, or N/A when unknown."""
    return formatting.format_size_or_na(value, MISSING)


def format_signed_size(value) -> str:
    """A signed size: N/A when unknown, `0 MB` below half a mebibyte."""
    if value is None:
        return MISSING
    if abs(value) < HALF_MIB:
        return "0 MB"
    sign = "+" if value > 0 else "-"
    return sign + formatting.format_size(abs(value))


def has_moved(triple: dict) -> bool:
    """True when a value appeared, disappeared, or moved by the noise threshold or more."""
    if triple["before"] == triple["after"]:
        return False
    if triple["delta"] is None:
        return True
    return abs(triple["delta"]) >= constants.DIFF_NOISE_BYTES


def is_changed(entry: dict) -> bool:
    """True when a per-process object is worth a row in the text form."""
    if entry["match"] in (MATCH_NEW, MATCH_GONE):
        return True
    for triple in entry["measured"].values():
        if triple["delta"] is not None and abs(triple["delta"]) >= constants.DIFF_NOISE_BYTES:
            return True
    runtime_block = entry["runtime"]
    if runtime_block is not None:
        context = runtime_block["n_ctx"]
        if context["before"] != context["after"]:
            return True
        if any(has_moved(triple) for triple in runtime_block["segments"].values()):
            return True
    files = entry["model_files"]
    return bool(files["added"] or files["removed"])


def format_pid_label(entry: dict) -> str:
    """The pid cell of one pair, prefixed `H:` for rows of another PID namespace."""
    pid_before = entry["pid_before"]
    pid_after = entry["pid_after"]
    if pid_before is None:
        label = str(pid_after)
    elif pid_after is None or pid_before == pid_after:
        label = str(pid_before)
    else:
        label = f"{pid_before}->{pid_after}"
    if entry["host"] is not None:
        label = f"H:{label}"
    return label


def get_entry_name(entry: dict) -> str:
    """The process name of a pair; a missing name prints as the unknown mark."""
    name = entry["name"]
    return str(name) if name else constants.UNKNOWN_NAME


def render_device_table(device: dict) -> list[str]:
    """Header and four rows of the device table."""
    lines = [f"{'DEVICE (measured)':<26}{'BEFORE':>11}{'AFTER':>11}{'CHANGE':>11}"]
    for label, key in DEVICE_ROWS:
        triple = device[key]
        lines.append(
            f"{label:<26}{format_plain_size(triple['before']):>11}"
            f"{format_plain_size(triple['after']):>11}{format_signed_size(triple['delta']):>11}"
        )
    return lines


def render_process_row(entry: dict) -> str:
    """One row of the process table."""
    vram = entry["measured"]["vram_bytes"]
    shared = entry["measured"]["shared_bytes"]
    before = NOT_LISTED if entry["match"] == MATCH_NEW else format_plain_size(vram["before"])
    after = NOT_LISTED if entry["match"] == MATCH_GONE else format_plain_size(vram["after"])
    return (
        f"{get_entry_name(entry)[:21]:<22}{format_pid_label(entry):>14}  {entry['match']:<11}"
        f"{before:>11}{after:>11}{format_signed_size(vram['delta']):>11}"
        f"{format_signed_size(shared['delta']):>11}"
    )


def render_process_table(entries: list[dict], changed: list[dict]) -> list[str]:
    """Header, changed rows and the two count lines of the process table."""
    lines = [
        f"{'PROCESSES (dedicated)':<22}{'PID':>14}  {'MATCH':<11}"
        f"{'BEFORE':>11}{'AFTER':>11}{'CHANGE':>11}{'SHARED +/-':>11}"
    ]
    lines.extend(render_process_row(entry) for entry in changed)
    if not changed:
        lines.append(NO_CHANGE_LINE)
    hidden = len(entries) - len(changed)
    if hidden:
        lines.append(
            f"{hidden} process(es) changed by less than 16 MB are not shown; --json lists all."
        )
    return lines


def render_runtime_lines(runtime_block: dict | None) -> list[str]:
    """Context-length and segment lines of one detail block."""
    if runtime_block is None:
        return []
    lines = []
    context = runtime_block["n_ctx"]
    if context["before"] != context["after"]:
        before = MISSING if context["before"] is None else context["before"]
        after = MISSING if context["after"] is None else context["after"]
        lines.append(f"  context length  {before} -> {after}")
    for kind, triple in runtime_block["segments"].items():
        if not has_moved(triple):
            continue
        lines.append(
            f"  {kind:<8} {format_plain_size(triple['before']):>9} -> "
            f"{format_plain_size(triple['after']):>9}  "
            f"{format_signed_size(triple['delta']):>9}  {triple['source']}"
        )
    return lines


def render_file_lines(files: dict) -> list[str]:
    """Added and removed model file lines of one detail block."""
    lines = []
    for verb, key in (("added", "added"), ("removed", "removed")):
        for model in files[key]:
            sources = ", ".join(str(source) for source in model["sources"])
            lines.append(f"  model file {verb}: {model['path']} [{sources}]")
    return lines


def render_detail_blocks(changed: list[dict]) -> list[str]:
    """One block per changed pair that has at least one detail line."""
    lines = []
    for entry in changed:
        detail_lines = render_runtime_lines(entry["runtime"])
        detail_lines.extend(render_file_lines(entry["model_files"]))
        if not detail_lines:
            continue
        lines.append("")
        lines.append(f"{get_entry_name(entry)} ({format_pid_label(entry)})")
        lines.extend(detail_lines)
    return lines


def render_footnotes(changed: list[dict]) -> list[str]:
    """The closing explanations that depend on what was shown."""
    lines = []
    has_files = any(
        entry["model_files"]["added"] or entry["model_files"]["removed"] for entry in changed
    )
    if has_files:
        lines.append("")
        lines.append(FILES_LINE)
    if any(entry["match"] in (MATCH_NEW, MATCH_GONE) for entry in changed):
        lines.append(NOT_LISTED_LINE)
    return lines


def render_diff(result: dict) -> str:
    """The text form of a comparison result, without a trailing newline."""
    heading = f"gputop diff: {result['gpu'] or 'unknown GPU'} ({result['backend']})"
    if result["seconds"] is not None:
        heading += f", {result['seconds']:.1f} s apart"
    lines = [heading]
    lines.extend(INTRO_LINES)
    if result["adapter_note"]:
        lines.append(f"Warning: {result['adapter_note']}")
    lines.append("")
    lines.extend(render_device_table(result["device"]))
    if result["memory_note"]:
        lines.append(str(result["memory_note"]))
    lines.append("")
    entries = result["processes"]
    changed = [entry for entry in entries if is_changed(entry)]
    lines.extend(render_process_table(entries, changed))
    lines.extend(render_detail_blocks(changed))
    lines.extend(render_footnotes(changed))
    return "\n".join(lines)


def build_diff_parser() -> argparse.ArgumentParser:
    """The argument parser of `gputop diff`."""
    parser = argparse.ArgumentParser(prog=DIFF_PROG, description=DIFF_DESCRIPTION)
    parser.add_argument("before", help="earlier --json snapshot")
    parser.add_argument("after", help="later --json snapshot")
    parser.add_argument("--json", action="store_true", help="print the comparison as JSON")
    return parser


def run_diff(arguments: list[str]) -> int:
    """Run `gputop diff` with the arguments after the keyword; returns the exit status."""
    parser = build_diff_parser()
    options = parser.parse_args(arguments)
    try:
        before = load_snapshot(options.before)
        after = load_snapshot(options.after)
        result = compare_snapshots(before, after)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if options.json:
        text = json.dumps(result, default=str, indent=2)
    else:
        text = render_diff(result)
    sys.stdout.write(text + "\n")
    return 0


def main():
    pass


if __name__ == "__main__":
    main()
