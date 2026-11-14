#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System clipboard through the desktop's own tool: wl-copy, xclip or xsel."""
import logging
import os
import shutil
import subprocess

# Commands in order of preference; each reads the text from stdin.
WAYLAND_COMMANDS = (("wl-copy",),)
X11_COMMANDS = (
    ("xclip", "-selection", "clipboard"),
    ("xsel", "--clipboard", "--input"),
)
COPY_TIMEOUT = 2.0

logger = logging.getLogger(__name__)


def list_copy_commands() -> list[tuple]:
    """Return the installed clipboard commands that fit the running display session."""
    candidates = []
    if os.environ.get("WAYLAND_DISPLAY"):
        candidates.extend(WAYLAND_COMMANDS)
    if os.environ.get("DISPLAY"):
        candidates.extend(X11_COMMANDS)
    return [command for command in candidates if shutil.which(command[0])]


def copy_text(text: str) -> bool:
    """Put text on the system clipboard; False when no tool is installed or all of them fail."""
    for command in list_copy_commands():
        try:
            subprocess.run(
                command,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=COPY_TIMEOUT,
                check=True,
            )
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(f"Clipboard command {command[0]} failed: {type(exc).__name__}: {exc}")
    return False


def main():
    pass


if __name__ == "__main__":
    main()
