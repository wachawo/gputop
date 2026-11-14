#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run gputop straight from a checkout, without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Local imports
from gputop import cli


def main():
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
