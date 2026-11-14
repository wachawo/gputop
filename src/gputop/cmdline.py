#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument-vector helpers: flag splitting, secret redaction, API key, diffusion test."""

SECRET_FLAGS = ("--api-key", "--api-key-file")
# Account tokens of model hubs; their values are hidden like the API key.
TOKEN_FLAGS = ("--hf-token", "-hft")
API_KEY_FLAG = "--api-key"
# Shortest spelling a parser that accepts unambiguous prefixes still resolves to an API key
# flag is one character longer than this stem.
API_KEY_STEM = "--api-"
LONG_PREFIX = "--"
DIFFUSION_FLAG = "--diffusion-model"
REDACTED_TEXT = "***"


def split_flag(argument: str) -> tuple[str, str | None]:
    """Split one argument at the first "=" into (flag, value); value is None without "="."""
    position = argument.find("=")
    if position < 0:
        return argument, None
    return argument[:position], argument[position + 1:]


def normalize_flag(flag: str) -> str:
    """Return a long flag with its underscores written as hyphens; others unchanged."""
    if not flag.startswith(LONG_PREFIX):
        return flag
    return flag.replace("_", "-")


def is_flag_spelling(flag: str, target: str) -> bool:
    """Tell whether a flag is the target API key flag or an abbreviation of it.

    Runtimes accept underscores for hyphens and any prefix long enough to be unambiguous,
    so `--api_key` and `--api-k` both name `--api-key`.
    """
    normalized = normalize_flag(flag)
    if normalized == target:
        return True
    return len(normalized) > len(API_KEY_STEM) and target.startswith(normalized)


def is_api_key_flag(flag: str) -> bool:
    """Tell whether the flag resolves to the API key flag itself."""
    return is_flag_spelling(flag, API_KEY_FLAG)


def is_secret_flag(flag: str) -> bool:
    """Tell whether the flag part of an argument names a secret."""
    if normalize_flag(flag) in TOKEN_FLAGS:
        return True
    for secret_flag in SECRET_FLAGS:
        if is_flag_spelling(flag, secret_flag):
            return True
    return False


def redact_cmdline(arguments: list[str]) -> list[str]:
    """Return a new argument list with the values of secret flags replaced by "***"."""
    redacted = []
    hide_next = False
    for argument in arguments:
        if hide_next:
            redacted.append(REDACTED_TEXT)
            hide_next = False
            continue
        flag, value = split_flag(argument)
        if not is_secret_flag(flag):
            redacted.append(argument)
        elif value is None:
            redacted.append(argument)
            hide_next = True
        else:
            redacted.append(f"{flag}={REDACTED_TEXT}")
    return redacted


def pick_first_key(value: str) -> str:
    """Return the first non-empty trimmed item of a comma-separated key list."""
    for item in value.split(","):
        trimmed = item.strip()
        if trimmed:
            return trimmed
    return ""


def find_api_key(arguments: list[str]) -> str:
    """Return the first usable item of the --api-key value, or "" when there is none."""
    for position, argument in enumerate(arguments):
        flag, value = split_flag(argument)
        if not is_api_key_flag(flag):
            continue
        if value is None:
            following = arguments[position + 1:position + 2]
            value = following[0] if following else ""
        return pick_first_key(value)
    return ""


def is_diffusion_cmdline(arguments: list[str]) -> bool:
    """Tell whether the argument vector carries the diffusion model flag."""
    prefix = DIFFUSION_FLAG + "="
    for argument in arguments:
        if argument == DIFFUSION_FLAG or argument.startswith(prefix):
            return True
    return False


def main():
    pass


if __name__ == "__main__":
    main()
