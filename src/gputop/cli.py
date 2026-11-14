#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line of gputop: argument parsing, logging setup, dispatch, one-shot output."""

import argparse
import logging
import math
import os
import sys
import time

# Local imports
from gputop import adapters
from gputop import collector
from gputop import constants
from gputop import diff
from gputop import export
from gputop import record
from gputop import report
from gputop import system

# Parser texts
PROG = "gputop"
DESCRIPTION = (
    "What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown."
)
EPILOG = (
    "gputop diff BEFORE.json AFTER.json compares two --json snapshots; "
    "gputop record -o FILE appends one --json snapshot per interval to FILE."
)

# Positional keywords that hand the remaining arguments to a subcommand
KEYWORD_DIFF = "diff"
KEYWORD_RECORD = "record"

# Exit status after Ctrl+C in a one-shot run, and after the reader of stdout went away
INTERRUPTED_STATUS = 130
BROKEN_PIPE_STATUS = 1

# Logging
LOG_FILE_VARIABLE = "GPUTOP_LOG_FILE"
LOGGING = {
    "handlers": [logging.NullHandler()],
    "format": "%(asctime)s.%(msecs)03d [%(levelname)s]: (%(name)s) %(message)s",
    "level": logging.INFO,
    "datefmt": "%Y-%m-%d %H:%M:%S",
}

logger = logging.getLogger(__name__)


def create_log_handler() -> logging.Handler:
    """A file handler for the path in GPUTOP_LOG_FILE, or a handler that drops everything."""
    path = os.environ.get(LOG_FILE_VARIABLE, "")
    if not path:
        return logging.NullHandler()
    try:
        return logging.FileHandler(path)
    except OSError:
        return logging.NullHandler()


def configure_logging() -> None:
    """Configure logging once; diagnostics never reach the terminal."""
    LOGGING["handlers"] = [create_log_handler()]
    logging.basicConfig(**LOGGING)


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the main command."""
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION, epilog=EPILOG)
    parser.add_argument(
        "--once", action="store_true", help="print one snapshot and exit (no TUI)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print one JSON snapshot and exit (implies --once)",
    )
    parser.add_argument(
        "--gpu", metavar="N", type=int, default=None,
        help="show only this GPU (TUI default: all; one-shot default: 0)",
    )
    parser.add_argument(
        "--list-gpus", action="store_true", help="list selectable GPU adapters and exit"
    )
    parser.add_argument(
        "--no-system", action="store_true",
        help="don't sample system RAM/CPU: no TUI panel or RAM/CPU%% columns, "
             "no system section in --once/--json",
    )
    parser.add_argument(
        "-i", "--interval", metavar="SECONDS", type=float, default=1.0,
        help="refresh seconds (default 1)",
    )
    return parser


def check_options(parser: argparse.ArgumentParser, options: argparse.Namespace) -> None:
    """The two early value checks; each failure leaves through the parser's usage error."""
    if options.gpu is not None and options.gpu < 0:
        parser.error(constants.GPU_INDEX_MESSAGE)
    if not math.isfinite(options.interval) or options.interval <= 0:
        parser.error(constants.INTERVAL_MESSAGE)


def silence_stdout() -> None:
    """Point stdout at /dev/null after its reader went away.

    The interpreter flushes stdout once more on exit; without this that flush reports the
    broken pipe on the terminal.
    """
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, sys.stdout.fileno())
        os.close(null_fd)
    except (OSError, ValueError):
        pass


def write_stdout(text: str) -> int:
    """Write text to stdout and flush it; returns the exit status of the write."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        silence_stdout()
        return BROKEN_PIPE_STATUS
    return 0


def run_diff(arguments: list[str]) -> int:
    """Run `gputop diff`; a reader that stops early ends it quietly."""
    try:
        status = diff.run_diff(arguments)
        sys.stdout.flush()
    except BrokenPipeError:
        silence_stdout()
        return BROKEN_PIPE_STATUS
    return status


def write_adapter_list() -> int:
    """Print one line per adapter of the ordered list."""
    ordered_adapters = adapters.list_ordered_adapters()
    lines = []
    for index, adapter in enumerate(ordered_adapters):
        lines.append(adapters.format_adapter_line(index, adapter) + "\n")
    return write_stdout("".join(lines))


def render_one_shot(collector_state: dict, as_json: bool, show_system: bool) -> str:
    """Prime, wait for the warm-up, sample once and return the report or JSON text."""
    holder = {"sampler": None}
    system_sample = None
    system_error = ""
    if show_system:
        system.take_system_reading(holder)
    time.sleep(constants.WARMUP_SECONDS)
    snapshot = collector.sample_collector(collector_state)
    if show_system:
        system_sample, system_error = system.take_system_reading(holder)
    if as_json:
        document = export.build_snapshot_dict(
            snapshot, system_sample, system_error, show_system
        )
        return export.dump_json(document)
    return report.render_report(snapshot, system_sample, system_error, show_system)


def run_one_shot(
    parser: argparse.ArgumentParser,
    options: argparse.Namespace,
    ordered_adapters: list[dict] | None,
) -> int:
    """The --once / --json sequence; the collector is closed on every path."""
    gpu_index = options.gpu or 0
    try:
        collector_state = collector.open_collector(
            gpu_index, budget_scale=None, ordered_adapters=ordered_adapters
        )
    except KeyboardInterrupt:
        return INTERRUPTED_STATUS
    except ValueError as exc:
        parser.error(str(exc))
    try:
        text = render_one_shot(collector_state, options.json, not options.no_system)
    except KeyboardInterrupt:
        return INTERRUPTED_STATUS
    finally:
        collector.close_collector(collector_state)
    return write_stdout(text + "\n")


def run_snapshot_mode(parser: argparse.ArgumentParser, options: argparse.Namespace) -> int:
    """Range check of --gpu, then the one-shot sequence; Ctrl+C at any point ends quietly."""
    try:
        ordered_adapters = record.check_gpu_range(parser, options.gpu)
        return run_one_shot(parser, options, ordered_adapters)
    except KeyboardInterrupt:
        return INTERRUPTED_STATUS


def start_tui(options: argparse.Namespace) -> int:
    """Start the full-screen interface; Textual is imported only here."""
    from gputop import tui
    return tui.run_tui(options.gpu, options.interval, not options.no_system)


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, dispatch, and return the exit status."""
    configure_logging()
    arguments = list(sys.argv[1:]) if argv is None else list(argv)
    if arguments and arguments[0] == KEYWORD_DIFF:
        return run_diff(arguments[1:])
    if arguments and arguments[0] == KEYWORD_RECORD:
        return record.run_record(arguments[1:])
    parser = build_parser()
    options = parser.parse_args(arguments)
    check_options(parser, options)
    if not sys.platform.startswith("linux"):
        sys.stderr.write(constants.LINUX_ONLY_MESSAGE + "\n")
        return 1
    if options.list_gpus:
        return write_adapter_list()
    if options.once or options.json:
        return run_snapshot_mode(parser, options)
    record.check_gpu_range(parser, options.gpu)
    return start_tui(options)


if __name__ == "__main__":
    raise SystemExit(main())
