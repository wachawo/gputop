#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point of `python -m gputop`."""

# Local imports
from gputop import cli


def main() -> int:
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
