#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Friendly model names from Ollama, Hugging Face and LM Studio stores."""
import json
import logging
import os
import re
import time

# Local imports
from gputop import constants

OLLAMA_BLOB_PATTERN = re.compile(r"^(.+)/blobs/sha256-([0-9a-f]{64})$", re.IGNORECASE)
OLLAMA_DEFAULT_REGISTRY = "registry.ollama.ai"
OLLAMA_DEFAULT_NAMESPACE = "library"
OLLAMA_DIGEST_PREFIX = "sha256:"
OLLAMA_INDEX_LIFETIME = 60.0
OLLAMA_MANIFEST_MAX_BYTES = 4 * constants.MIB

HUGGINGFACE_PATTERN = re.compile(r"/models--([^/]+?)--([^/]+)/snapshots/", re.IGNORECASE)

LMSTUDIO_SETTINGS_FILES = ("~/.lmstudio/settings.json", "~/.cache/lm-studio/settings.json")
LMSTUDIO_MODEL_DIRECTORIES = ("~/.lmstudio/models", "~/.cache/lm-studio/models")
LMSTUDIO_DOWNLOADS_KEY = "downloadsFolder"
LMSTUDIO_LIFETIME = 60.0
LMSTUDIO_SETTINGS_MAX_BYTES = 4 * constants.MIB
LMSTUDIO_MIN_COMPONENTS = 3

# Process-wide caches: manifest index per models root, and the LM Studio directory list
OLLAMA_INDEX_CACHE = {}
LMSTUDIO_CACHE = {}

logger = logging.getLogger(__name__)


def read_json_file(path: str, max_bytes: int) -> object:
    """Parse a small UTF-8 JSON file; None when it is unreadable, too large or invalid."""
    try:
        with open(path, "rb") as handle:
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            return None
        return json.loads(payload.decode("utf-8"))
    except (OSError, ValueError):
        return None


def list_directory(path: str, want_directories: bool) -> list[str]:
    """Return the sorted names of the sub-directories (or of the other entries) of a directory."""
    names = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    is_directory = entry.is_dir()
                except OSError:
                    continue
                if is_directory == want_directories:
                    names.append(entry.name)
    except OSError:
        return []
    names.sort()
    return names


def read_manifest_digests(path: str) -> list[str]:
    """Return the lower-cased layer digests of one Ollama manifest file."""
    document = read_json_file(path, OLLAMA_MANIFEST_MAX_BYTES)
    if not isinstance(document, dict):
        return []
    layers = document.get("layers")
    if not isinstance(layers, list):
        return []
    digests = []
    for layer in layers:
        digest = layer.get("digest") if isinstance(layer, dict) else None
        if not isinstance(digest, str):
            continue
        if digest.startswith(OLLAMA_DIGEST_PREFIX):
            digest = digest[len(OLLAMA_DIGEST_PREFIX):]
        digests.append(digest.lower())
    return digests


def build_ollama_name(registry: str, namespace: str, model: str, tag: str) -> str:
    """Compose the short Ollama model name for one manifest location."""
    prefix = "" if registry == OLLAMA_DEFAULT_REGISTRY else registry + "/"
    if namespace != OLLAMA_DEFAULT_NAMESPACE:
        prefix = prefix + namespace + "/"
    return f"{prefix}{model}:{tag}"


def add_model_tags(
    index: dict, model_directory: str, registry: str, namespace: str, model: str
) -> None:
    """Add every tag file of one model directory to the digest index."""
    for tag in list_directory(model_directory, False):
        name = build_ollama_name(registry, namespace, model, tag)
        manifest_path = os.path.join(model_directory, tag)
        for digest in read_manifest_digests(manifest_path):
            names = index.setdefault(digest, set())
            names.add(name)


def build_ollama_index(root: str) -> dict:
    """Walk <root>/manifests three levels deep and map each layer digest to its model names."""
    index = {}
    manifests = os.path.join(root, "manifests")
    for registry in list_directory(manifests, True):
        registry_directory = os.path.join(manifests, registry)
        for namespace in list_directory(registry_directory, True):
            namespace_directory = os.path.join(registry_directory, namespace)
            for model in list_directory(namespace_directory, True):
                model_directory = os.path.join(namespace_directory, model)
                add_model_tags(index, model_directory, registry, namespace, model)
    return index


def get_ollama_index(root: str) -> dict:
    """Return the manifest index of a models root, rebuilt at most once per lifetime."""
    now = time.monotonic()
    cached = OLLAMA_INDEX_CACHE.get(root)
    if cached is not None and now - cached["time"] < OLLAMA_INDEX_LIFETIME:
        return cached["index"]
    index = build_ollama_index(root)
    OLLAMA_INDEX_CACHE[root] = {"time": now, "index": index}
    return index


def resolve_ollama_name(path: str) -> str | None:
    """Name an Ollama blob by the manifests that name its digest."""
    found = OLLAMA_BLOB_PATTERN.match(path)
    if found is None:
        return None
    root = found.group(1)
    digest_text = found.group(2)
    digest = digest_text.lower()
    index = get_ollama_index(root)
    names = index.get(digest)
    if not names:
        return None
    return ", ".join(sorted(names))


def resolve_huggingface_name(path: str) -> str | None:
    """Name a Hugging Face cache snapshot file as <org>/<repo>."""
    found = HUGGINGFACE_PATTERN.search(path)
    if found is None:
        return None
    return f"{found.group(1)}/{found.group(2)}"


def read_downloads_folder(settings_path: str) -> str:
    """Return the downloadsFolder of one LM Studio settings file, or "" when there is none."""
    document = read_json_file(os.path.expanduser(settings_path), LMSTUDIO_SETTINGS_MAX_BYTES)
    if not isinstance(document, dict):
        return ""
    folder = document.get(LMSTUDIO_DOWNLOADS_KEY)
    if not isinstance(folder, str) or not folder:
        return ""
    return folder


def build_lmstudio_directories() -> list[str]:
    """List the LM Studio model directories in priority order, absolute and de-duplicated."""
    configured = []
    for settings_path in LMSTUDIO_SETTINGS_FILES:
        folder = read_downloads_folder(settings_path)
        if folder:
            configured.insert(0, folder)
    directories = []
    for folder in configured + list(LMSTUDIO_MODEL_DIRECTORIES):
        expanded = os.path.expanduser(folder)
        absolute = os.path.abspath(expanded)
        if absolute not in directories:
            directories.append(absolute)
    return directories


def get_lmstudio_directories() -> list[str]:
    """Return the LM Studio model directories, re-read at most once per lifetime."""
    now = time.monotonic()
    if LMSTUDIO_CACHE and now - LMSTUDIO_CACHE["time"] < LMSTUDIO_LIFETIME:
        return LMSTUDIO_CACHE["directories"]
    directories = build_lmstudio_directories()
    LMSTUDIO_CACHE["time"] = now
    LMSTUDIO_CACHE["directories"] = directories
    return directories


def resolve_lmstudio_name(path: str) -> str | None:
    """Name a file below an LM Studio model directory as <publisher>/<repo>."""
    absolute = os.path.abspath(path)
    for directory in get_lmstudio_directories():
        prefix = directory.rstrip(os.sep) + os.sep
        if not absolute.startswith(prefix):
            continue
        remainder = absolute[len(prefix):]
        components = [part for part in remainder.split(os.sep) if part]
        if len(components) >= LMSTUDIO_MIN_COMPONENTS:
            return f"{components[0]}/{components[1]}"
    return None


def resolve_store_name(path: str) -> str | None:
    """Return the friendly store name of a model path: Ollama, Hugging Face, then LM Studio."""
    if not path:
        return None
    try:
        for resolver in (resolve_ollama_name, resolve_huggingface_name, resolve_lmstudio_name):
            name = resolver(path)
            if name:
                return name
    except Exception as exc:
        logger.debug(f"Store name lookup failed: {type(exc).__name__}: {str(exc)}")
    return None


def main():
    pass


if __name__ == "__main__":
    main()
