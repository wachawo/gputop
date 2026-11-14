#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-file rule, launch-flag models, path-argument hints, installed candidates, evidence text."""
import collections
import os
import stat
import time

# Local imports
from gputop import cmdline
from gputop import constants
from gputop import gguf
from gputop import records
from gputop import stores

# Extensions
MODEL_EXTENSIONS = (
    ".gguf", ".ggml", ".safetensors", ".onnx", ".pt", ".pth", ".bin", ".engine", ".plan",
    ".tflite", ".ckpt",
)
AMBIGUOUS_EXTENSIONS = (".bin", ".pt", ".pth", ".ckpt")
NEVER_MODEL_EXTENSIONS = (
    ".dll", ".exe", ".sys", ".mui", ".ttf", ".ttc", ".otf", ".dat", ".db", ".nls", ".pak", ".mun",
)
GGUF_EXTENSION = ".gguf"

# Size thresholds
AMBIGUOUS_MIN_SIZE = 64 * constants.MIB
MODEL_MIN_SIZE = 4 * constants.MIB
GGUF_CHECK_MIN_SIZE = 64 * constants.MIB

# Pseudo paths
PSEUDO_PREFIXES = ("[", "/dev/", "/proc/", "/sys/", "/memfd:", "/SYSV", "anon_inode:")
DELETED_SUFFIX = " (deleted)"

# Launch flags
MODEL_FLAGS = ("-m", "--model", "--mmproj", "-md", "--model-draft", "--lora")
DIFFUSION_MODEL_FLAGS = ("--diffusion-model", "--vae", "--llm")
POSITIONAL_MARKER = "--"

# Path-argument hints
PATH_ARG_LAST_INDEX = 256
PATH_ARG_MAX_CANDIDATES = 16

# Installed candidates
SYSTEM_EXE_PREFIXES = (
    "/usr/bin/", "/usr/sbin/", "/bin/", "/sbin/", "/usr/lib", "/lib", "/usr/libexec/", "/snap/",
)
INSTALLED_SKIP_DIRECTORIES = (
    "ebwebview", "cache", "code cache", "gpucache", "node_modules", "logs", "crashpad",
    "locales", "__pycache__",
)
INSTALLED_MIN_STEM = 4
INSTALLED_MAX_DEPTH = 4
INSTALLED_ENTRY_BUDGET = 3000
INSTALLED_KEEP = 6
INSTALLED_LIFETIME = 300.0

# Evidence texts
EVIDENCE_SINGLE_TEXT = {
    constants.EVIDENCE_MAPPED: "in use (mapped)",
    constants.EVIDENCE_CMDLINE: "launch arg",
    constants.EVIDENCE_API: "reported (api)",
}
EVIDENCE_PATH_ARG_TEXT = "path-arg (not proven loaded)"
EVIDENCE_INSTALLED_TEXT = "installed, not proven loaded"

# Process-wide cache of the installed-model walk: executable path to {"time", "models"}
INSTALLED_CACHE = {}


def has_model_evidence(model: dict) -> bool:
    """Tell whether a model file record has a source that proves the file is in use."""
    for source in model["sources"]:
        if source in constants.PROVING_EVIDENCE:
            return True
    return False


def describe_evidence(sources: list[str]) -> str:
    """Return the evidence text for a list of sources; "" for an empty list."""
    proving = [source for source in constants.PROVING_EVIDENCE if source in sources]
    if len(proving) >= 2:
        return f"confirmed ({'+'.join(proving)})"
    if len(proving) == 1:
        return EVIDENCE_SINGLE_TEXT.get(proving[0], "")
    if constants.EVIDENCE_PATH_ARG in sources:
        return EVIDENCE_PATH_ARG_TEXT
    if constants.EVIDENCE_INSTALLED in sources:
        return EVIDENCE_INSTALLED_TEXT
    return ""


def get_display_name(model: dict) -> str:
    """Return the store name of a model file record, or its path."""
    return model["store_name"] or model["path"]


def get_short_name(model: dict) -> str:
    """Return the store name of a model file record, or the basename of its path."""
    if model["store_name"]:
        return model["store_name"]
    trimmed = model["path"].rstrip("/")
    return os.path.basename(trimmed) or model["path"]


def get_file_identity(path: str) -> tuple:
    """Return ("inode", device, inode), or ("path", normalised absolute path) as a fallback."""
    try:
        status = os.stat(path)
        if status.st_ino:
            return ("inode", status.st_dev, status.st_ino)
    except (OSError, ValueError):
        pass
    absolute = os.path.abspath(path)
    return ("path", os.path.normpath(absolute))


def get_extension(path: str) -> str:
    """Return the lower-cased extension of the basename of a path, with its dot."""
    basename = os.path.basename(path)
    extension = os.path.splitext(basename)[1]
    return extension.lower()


def is_model_file(extension: str, size: int, gguf: bool, has_store_name: bool) -> bool:
    """Apply the model-file rule: GGUF, model extension, then the size thresholds."""
    if gguf:
        return True
    if extension not in MODEL_EXTENSIONS:
        return False
    if extension in AMBIGUOUS_EXTENSIONS:
        return size >= AMBIGUOUS_MIN_SIZE or has_store_name
    return size >= MODEL_MIN_SIZE


def is_pseudo_path(path: str) -> bool:
    """Tell whether a mapped path is not a real file; decided from the text alone."""
    if path.startswith(PSEUDO_PREFIXES):
        return True
    return path.endswith(DELETED_SUFFIX)


def stat_regular_file(path: str) -> os.stat_result | None:
    """Return the stat result of an existing regular file, or None."""
    try:
        status = os.stat(path)
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return status


def create_mapped_model(path: str, resident: int) -> dict | None:
    """Build the model file record of one mapped path, or None when it is not a model."""
    extension = get_extension(path)
    if extension in NEVER_MODEL_EXTENSIONS:
        return None
    try:
        size = os.stat(path).st_size
    except (OSError, ValueError):
        return None
    wants_check = extension == GGUF_EXTENSION or not extension or size > GGUF_CHECK_MIN_SIZE
    is_verified = gguf.is_gguf(path) if wants_check else False
    if not is_verified and extension not in MODEL_EXTENSIONS:
        return None
    store_name = stores.resolve_store_name(path)
    if not is_model_file(extension, size, is_verified, bool(store_name)):
        return None
    kind = gguf.detect_gguf_kind(path) if is_verified else ""
    sources = [constants.EVIDENCE_MAPPED]
    return records.create_model_file(path, size, resident, is_verified, sources, store_name, kind)


def create_model_from_path(path: str, source: str) -> dict | None:
    """Build a model file record for an existing regular file, or None when it is not one."""
    status = stat_regular_file(path)
    if status is None:
        return None
    is_verified = gguf.is_gguf(path)
    kind = gguf.detect_gguf_kind(path) if is_verified else ""
    store_name = stores.resolve_store_name(path)
    return records.create_model_file(
        path, status.st_size, 0, is_verified, [source], store_name, kind
    )


def list_model_flags(arguments: list[str]) -> tuple:
    """Return the launch flags that name a model for this argument vector."""
    if cmdline.is_diffusion_cmdline(arguments):
        return MODEL_FLAGS + DIFFUSION_MODEL_FLAGS
    return MODEL_FLAGS


def resolve_argument_path(value: str, cwd: str) -> str:
    """Return the path an argument names, joined to cwd when relative; "" when unusable."""
    if not value or "\0" in value:
        return ""
    if os.path.isabs(value):
        return value
    if not cwd or not os.path.isabs(cwd):
        return ""
    return os.path.join(cwd, value)


def list_flag_values(arguments: list[str]) -> list[str]:
    """Return the values of every model flag, in order of appearance."""
    model_flags = list_model_flags(arguments)
    values = []
    skip_secret = False
    for position, argument in enumerate(arguments):
        if skip_secret:
            skip_secret = False
            continue
        flag, value = cmdline.split_flag(argument)
        if cmdline.is_secret_flag(flag):
            skip_secret = value is None
            continue
        if cmdline.normalize_flag(flag) not in model_flags:
            continue
        if value is None:
            following = arguments[position + 1:position + 2]
            value = following[0] if following else ""
        values.append(value)
    return values


def find_cmdline_models(arguments: list[str], cwd: str) -> list[str]:
    """Return the absolute paths of existing regular files named by model flags."""
    paths = []
    for value in list_flag_values(arguments):
        joined = resolve_argument_path(value, cwd)
        if not joined or stat_regular_file(joined) is None:
            continue
        absolute = os.path.abspath(joined)
        if absolute not in paths:
            paths.append(absolute)
    return paths


def pick_path_candidate(argument: str, value: str | None, positional: bool) -> str:
    """Return the text of one argument that may be a path; "" when it is only a switch."""
    if positional or not argument.startswith("-"):
        return argument
    return value or ""


def is_path_candidate(candidate: str) -> bool:
    """Tell whether an argument text is worth checking as a model path."""
    if not candidate or "\0" in candidate or "://" in candidate:
        return False
    if get_extension(candidate) not in MODEL_EXTENSIONS:
        return False
    return not is_pseudo_path(candidate)


def list_path_candidates(arguments: list[str]) -> list[str]:
    """Walk the arguments and return the distinct path-like texts, at most the allowed number."""
    model_flags = list_model_flags(arguments)
    candidates = []
    positional = False
    skip_next = False
    for argument in arguments[1:PATH_ARG_LAST_INDEX + 1]:
        if skip_next:
            skip_next = False
            continue
        flag, value = cmdline.split_flag(argument)
        if cmdline.is_secret_flag(flag):
            skip_next = value is None
            continue
        is_model_flag = flag in model_flags
        skip_next = is_model_flag and value is None
        if not positional and argument == POSITIONAL_MARKER:
            positional = True
            continue
        if not positional and is_model_flag:
            continue
        candidate = pick_path_candidate(argument, value, positional)
        if not is_path_candidate(candidate) or candidate in candidates:
            continue
        candidates.append(candidate)
        if len(candidates) >= PATH_ARG_MAX_CANDIDATES:
            break
    return candidates


def create_path_arg_model(candidate: str, cwd: str) -> dict | None:
    """Check one path-like argument and build its hint record, or None when it is dropped."""
    path = resolve_argument_path(candidate, cwd)
    if not path or is_pseudo_path(path) or is_pseudo_path(os.path.normpath(path)):
        return None
    status = stat_regular_file(path)
    if status is None:
        return None
    extension = get_extension(path)
    is_verified = False
    if extension == GGUF_EXTENSION:
        is_verified = gguf.is_gguf(path)
        if not is_verified:
            return None
    store_name = stores.resolve_store_name(path)
    if not is_model_file(extension, status.st_size, is_verified, bool(store_name)):
        return None
    sources = [constants.EVIDENCE_PATH_ARG]
    return records.create_model_file(
        path, status.st_size, None, is_verified, sources, store_name, ""
    )


def find_path_arg_models(arguments: list[str], cwd: str) -> list[dict]:
    """Return the path-argument hints of an argument vector as model file records."""
    models = []
    for candidate in list_path_candidates(arguments):
        model = create_path_arg_model(candidate, cwd)
        if model is not None:
            models.append(model)
    return models


def get_xdg_directory(variable: str, fallback: str) -> str:
    """Return an XDG base directory from the environment, or its default below the home."""
    configured = os.environ.get(variable, "")
    if configured and os.path.isabs(configured):
        return configured
    return os.path.expanduser(fallback)


def list_named_directories(base: str, stem: str) -> list[str]:
    """Return the directories directly inside base whose lower-cased name contains the stem."""
    found = []
    try:
        with os.scandir(base) as entries:
            for entry in entries:
                try:
                    if stem in entry.name.lower() and entry.is_dir(follow_symlinks=False):
                        found.append(entry.path)
                except OSError:
                    continue
    except OSError:
        return []
    found.sort()
    return found


def list_install_roots(exe: str) -> list[str]:
    """Return the directories searched for models installed with an executable."""
    roots = []
    exe_directory = os.path.dirname(exe)
    if exe_directory and not exe.startswith(SYSTEM_EXE_PREFIXES):
        roots.append(exe_directory)
    basename = os.path.basename(exe)
    stem_text = os.path.splitext(basename)[0]
    stem = stem_text.lower()
    if len(stem) >= INSTALLED_MIN_STEM:
        data_home = get_xdg_directory("XDG_DATA_HOME", "~/.local/share")
        config_home = get_xdg_directory("XDG_CONFIG_HOME", "~/.config")
        for base in (data_home, config_home):
            for directory in list_named_directories(base, stem):
                if directory not in roots:
                    roots.append(directory)
    return roots


def inspect_install_entry(
    entry: os.DirEntry, depth: int, pending: collections.deque, folders: dict
) -> None:
    """Queue a sub-directory for the walk, or record a file that passes the model rule."""
    lowered = entry.name.lower()
    if entry.is_dir(follow_symlinks=False):
        if depth < INSTALLED_MAX_DEPTH and lowered not in INSTALLED_SKIP_DIRECTORIES:
            pending.append((entry.path, depth + 1))
        return
    extension = os.path.splitext(lowered)[1]
    if extension not in MODEL_EXTENSIONS or not entry.is_file():
        return
    size = entry.stat().st_size
    if is_model_file(extension, size, False, False):
        folder = os.path.dirname(entry.path)
        files = folders.setdefault(folder, [])
        files.append((entry.name, entry.path, size))


def walk_install_root(root: str, budget: dict, folders: dict) -> None:
    """Walk one root breadth-first within the shared entry budget, filling folders."""
    pending = collections.deque([(root, 0)])
    while pending and budget["left"] > 0:
        directory, depth = pending.popleft()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if budget["left"] <= 0:
                        break
                    budget["left"] -= 1
                    try:
                        inspect_install_entry(entry, depth, pending, folders)
                    except OSError:
                        continue
        except OSError:
            continue


def create_installed_model(folder: str, files: list[tuple]) -> dict:
    """Build the record of one folder: its single model file, or the folder as a group."""
    ordered = sorted(files)
    first_path = ordered[0][1]
    store_name = stores.resolve_store_name(first_path)
    sources = [constants.EVIDENCE_INSTALLED]
    if len(ordered) > 1:
        total = sum(item[2] for item in ordered)
        return records.create_model_file(folder, total, 0, False, sources, store_name, "")
    is_verified = get_extension(first_path) == GGUF_EXTENSION and gguf.is_gguf(first_path)
    kind = gguf.detect_gguf_kind(first_path) if is_verified else ""
    return records.create_model_file(
        first_path, ordered[0][2], 0, is_verified, sources, store_name, kind
    )


def build_installed_models(exe: str) -> list[dict]:
    """Walk the install roots of an executable and return its largest model candidates."""
    folders = {}
    budget = {"left": INSTALLED_ENTRY_BUDGET}
    for root in list_install_roots(exe):
        walk_install_root(root, budget, folders)
    models = [create_installed_model(folder, files) for folder, files in folders.items()]
    models.sort(key=lambda model: (-model["size"], model["path"]))
    return models[:INSTALLED_KEEP]


def drop_stale_installed(now: float) -> None:
    """Forget cached installed-model walks that are past their lifetime."""
    stale = [exe for exe, cached in INSTALLED_CACHE.items()
             if now - cached["time"] >= INSTALLED_LIFETIME]
    for exe in stale:
        del INSTALLED_CACHE[exe]


def find_installed_models(exe: str) -> list[dict]:
    """Return the installed, unproven model candidates of an executable (cached per path)."""
    if not exe:
        return []
    now = time.monotonic()
    cached = INSTALLED_CACHE.get(exe)
    if cached is None or now - cached["time"] >= INSTALLED_LIFETIME:
        drop_stale_installed(now)
        cached = {"time": now, "models": build_installed_models(exe)}
        INSTALLED_CACHE[exe] = cached
    return [records.copy_model_file(model) for model in cached["models"]]


def main():
    pass


if __name__ == "__main__":
    main()
