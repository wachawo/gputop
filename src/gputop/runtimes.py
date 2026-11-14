#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library-name normalisation and the runtime label table."""
import fnmatch
import re

# Local imports
from gputop import constants

LIBRARY_PATTERN = re.compile(r"^(.*\.so)(\.[0-9][0-9a-z.]*)?$")
NODE_SUFFIX = ".node"

# Rows of (label, library patterns, is an AI runtime), most specific label first
RUNTIME_TABLE = (
    (constants.RUNTIME_LLAMA_CPP, ("libllama*.so", "llm_engine*"), True),
    ("whisper.cpp", ("libwhisper*.so",), True),
    ("ggml", ("libggml*.so",), True),
    ("PyTorch", ("libtorch_cuda*.so", "libtorch_cpu.so"), True),
    ("TensorRT", ("libnvinfer*.so",), True),
    ("ONNX RT", ("libonnxruntime*.so",), True),
    (constants.RUNTIME_HIP, ("libamdhip64.so",), False),
    ("cuDNN", ("libcudnn*.so",), True),
    (constants.RUNTIME_CUDA, ("libcuda.so", "libcudart.so", "libcublas.so"), False),
    (constants.RUNTIME_VULKAN, ("libvulkan.so",), False),
    (constants.RUNTIME_D3D12, ("libd3d12core.so",), False),
    ("OpenGL", ("libgl.so", "libglx*.so", "libegl*.so"), False),
    (constants.RUNTIME_NVENC, ("libnvidia-encode.so",), False),
    ("NVDEC", ("libnvcuvid.so",), False),
)
AI_RUNTIMES = tuple(row[0] for row in RUNTIME_TABLE if row[2])


def normalize_library_name(basename: str) -> str | None:
    """Return the lower-cased library name without its version suffix, or None for a non-library."""
    lowered = basename.lower()
    if lowered.endswith(NODE_SUFFIX):
        return lowered
    found = LIBRARY_PATTERN.match(lowered)
    if found is None:
        return None
    return found.group(1)


def has_library_match(library_names: set[str], patterns: tuple) -> bool:
    """Tell whether any library name matches any of the glob patterns as a whole name."""
    for pattern in patterns:
        for library_name in library_names:
            if fnmatch.fnmatchcase(library_name, pattern):
                return True
    return False


def match_runtimes(library_names: set[str]) -> list[str]:
    """Return the runtime labels triggered by the library names, in table order."""
    labels = []
    for label, patterns, unused_is_ai in RUNTIME_TABLE:
        if has_library_match(library_names, patterns):
            labels.append(label)
    return labels


def main():
    pass


if __name__ == "__main__":
    main()
