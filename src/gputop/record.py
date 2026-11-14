#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The `gputop record` subcommand: append one --json snapshot per interval to a file."""

import argparse
import functools
import logging
import math
import os
import stat
import sys
import time
import traceback

# Local imports
from gputop import adapters
from gputop import collector
from gputop import constants
from gputop import export
from gputop import formatting
from gputop import system

# Parser texts
RECORD_PROG = "gputop record"
RECORD_DESCRIPTION = (
    "Append one --json snapshot per interval to a JSON Lines file, for example to keep "
    "what a long run did to the GPU."
)
DURATION_MESSAGE = "--duration must be a finite number greater than zero"

logger = logging.getLogger(__name__)


def build_record_parser() -> argparse.ArgumentParser:
    """The argument parser of `gputop record`."""
    parser = argparse.ArgumentParser(prog=RECORD_PROG, description=RECORD_DESCRIPTION)
    parser.add_argument(
        "-o", "--output", metavar="FILE", required=True,
        help="file to append to, one --json snapshot per line",
    )
    parser.add_argument(
        "--gpu", metavar="N", type=int, default=0,
        help="GPU to record (default 0; gputop --list-gpus shows the indexes)",
    )
    parser.add_argument(
        "-i", "--interval", metavar="SECONDS", type=float, default=1.0,
        help="seconds between samples (default 1)",
    )
    parser.add_argument(
        "--duration", metavar="SECONDS", type=float, default=None,
        help="stop after this many seconds (default: until Ctrl+C)",
    )
    parser.add_argument(
        "--no-system", action="store_true",
        help="don't sample system RAM/CPU: no system section in the records",
    )
    return parser


def is_positive_finite(value: float) -> bool:
    """True for a finite number greater than zero."""
    return math.isfinite(value) and value > 0


def check_options(parser: argparse.ArgumentParser, options: argparse.Namespace) -> None:
    """The three value checks; each failure leaves through the parser's usage error."""
    if options.gpu < 0:
        parser.error(constants.GPU_INDEX_MESSAGE)
    if not is_positive_finite(options.interval):
        parser.error(constants.INTERVAL_MESSAGE)
    if options.duration is not None and not is_positive_finite(options.duration):
        parser.error(DURATION_MESSAGE)


def format_seconds(value: float) -> str:
    """A number of seconds in its shortest form: `1`, `0.5`, `300`."""
    if value == int(value):
        return str(int(value))
    return repr(float(value))


def has_unterminated_line(path: str) -> bool:
    """True when the path is a non-empty regular file whose last byte is not a newline.

    Anything else (missing, FIFO, device) is never opened for reading: a FIFO would block
    until a writer appears, and the only writer would be this process.
    """
    try:
        file_mode = os.stat(path).st_mode
    except OSError:
        return False
    if not stat.S_ISREG(file_mode):
        return False
    try:
        with open(path, "rb") as handle:
            handle.seek(-1, 2)
            last_byte = handle.read(1)
    except OSError:
        return False
    return bool(last_byte) and last_byte != b"\n"


def open_output(path: str):
    """Open the recording for appending; closes a line an earlier run left cut short."""
    needs_newline = has_unterminated_line(path)
    stream = open(path, "a", encoding="utf-8", newline="\n")
    if needs_newline:
        try:
            stream.write("\n")
            stream.flush()
        except OSError:
            close_output(stream)
            raise
    return stream


def close_output(stream) -> None:
    """Close the recording; every line was flushed already, so a failure costs nothing."""
    try:
        stream.close()
    except Exception:
        pass


def create_session(collector_state: dict, show_system: bool) -> dict:
    """Sampling state of one recording.

    Keys: `collector` (collector state), `show_system` (bool: records carry a system
    section), `sampler` (system sampler state, or None until it could be created).
    """
    return {"collector": collector_state, "show_system": show_system, "sampler": None}


def take_system_reading(session: dict) -> tuple:
    """One system sample as (sample or None, error text); the sampler is created lazily."""
    if not session["show_system"]:
        return None, ""
    try:
        if session["sampler"] is None:
            session["sampler"] = system.open_system_sampler()
        return system.sample_system(session["sampler"]), ""
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        return None, formatting.format_error(exc)


def take_record_line(session: dict) -> str:
    """One record: the GPU snapshot, then the system sample, as one compact JSON line."""
    snapshot = collector.sample_collector(session["collector"])
    system_sample, system_error = take_system_reading(session)
    document = export.build_snapshot_dict(
        snapshot, system_sample, system_error, session["show_system"]
    )
    return export.dump_json(document, compact=True)


def run_record_loop(
    take_sample, stream, interval: float, duration: float | None, clock, sleep
) -> int:
    """Write one line per interval without drift; returns the number of lines written.

    `take_sample()` returns one JSON line without the newline, `clock()` is monotonic
    seconds, `sleep(seconds)` waits. The first sample is due at once; a sample is taken
    while its deadline lies inside `duration`; a slow sample delays the next one instead
    of causing a burst. Ctrl+C ends the loop quietly.
    """
    count = 0
    start = clock()
    due = start
    try:
        while duration is None or due - start < duration:
            wait = due - clock()
            if wait > 0:
                sleep(wait)
            line = take_sample()
            stream.write(line + "\n")
            stream.flush()
            count += 1
            due = max(due + interval, clock())
    except KeyboardInterrupt:
        pass
    return count


def write_start_line(options: argparse.Namespace) -> None:
    """Announce the recording on stderr."""
    span = ""
    if options.duration is not None:
        span = f" for {format_seconds(options.duration)} s"
    sys.stderr.write(
        f"Recording GPU {options.gpu} every {format_seconds(options.interval)} s{span} "
        f"to {options.output}; press Ctrl+C to stop.\n"
    )


def record_session(session: dict, options: argparse.Namespace, stream) -> int:
    """Prime the samplers, wait for the warm-up, then run the loop; returns the count."""
    try:
        take_system_reading(session)
        time.sleep(constants.WARMUP_SECONDS)
    except KeyboardInterrupt:
        return 0
    write_start_line(options)
    take_sample = functools.partial(take_record_line, session)
    return run_record_loop(
        take_sample, stream, options.interval, options.duration, time.monotonic, time.sleep
    )


def record_to_stream(
    parser: argparse.ArgumentParser,
    options: argparse.Namespace,
    ordered_adapters: list[dict] | None,
    stream,
) -> int:
    """Open the collector, record until the end, close the collector; returns the count."""
    try:
        collector_state = collector.open_collector(
            options.gpu, budget_scale=1.0, ordered_adapters=ordered_adapters
        )
    except KeyboardInterrupt:
        return 0
    except ValueError as exc:
        parser.error(str(exc))
    try:
        session = create_session(collector_state, not options.no_system)
        return record_session(session, options, stream)
    finally:
        collector.close_collector(collector_state)


def write_closing_line(count: int, output: str) -> None:
    """Report on stderr how many samples the recording holds."""
    sys.stderr.write(f"Recorded {count} sample(s) to {output}\n")


def check_gpu_range(parser: argparse.ArgumentParser, gpu_index: int) -> list[dict] | None:
    """Range check of an index above 0; returns the ordered adapter list it enumerated."""
    if gpu_index <= 0:
        return None
    ordered_adapters = adapters.list_ordered_adapters()
    try:
        adapters.select_adapter(ordered_adapters, gpu_index)
    except ValueError as exc:
        parser.error(str(exc))
    return ordered_adapters


def run_record(arguments: list[str]) -> int:
    """Run `gputop record` with the arguments after the keyword; returns the exit status."""
    parser = build_record_parser()
    options = parser.parse_args(arguments)
    check_options(parser, options)
    if not sys.platform.startswith("linux"):
        sys.stderr.write(constants.LINUX_ONLY_MESSAGE + "\n")
        return 1
    try:
        ordered_adapters = check_gpu_range(parser, options.gpu)
    except KeyboardInterrupt:
        write_closing_line(0, options.output)
        return 0
    try:
        stream = open_output(options.output)
    except OSError as exc:
        parser.error(f"cannot write {options.output}: {exc}")
    try:
        count = record_to_stream(parser, options, ordered_adapters, stream)
    except OSError as exc:
        sys.stderr.write(f"gputop record: recording to {options.output} failed: {exc}\n")
        return 1
    finally:
        close_output(stream)
    write_closing_line(count, options.output)
    return 0


def main():
    pass


if __name__ == "__main__":
    main()
