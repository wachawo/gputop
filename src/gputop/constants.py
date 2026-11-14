#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants of gputop: units, names, style tables, note texts and budgets."""

# Units
MIB = 1 << 20
GIB = 1 << 30

# Vendors
VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002
VENDOR_INTEL = 0x8086
VENDOR_NAMES = {
    VENDOR_NVIDIA: "NVIDIA",
    VENDOR_AMD: "AMD",
    VENDOR_INTEL: "Intel",
}

# Backends, telemetry sources, kernel drivers
BACKEND_NVML = "nvml"
BACKEND_DRM = "drm"
SOURCE_NVML = "NVML"
SOURCE_AMD = "AMD sysfs"
SOURCE_INTEL = "Intel sysfs"
DRIVER_AMDGPU = "amdgpu"
DRIVER_I915 = "i915"
DRIVER_XE = "xe"
DRM_DRIVERS = (DRIVER_AMDGPU, DRIVER_I915, DRIVER_XE)

# Command line messages
LINUX_ONLY_MESSAGE = "gputop supports Linux only (NVIDIA through NVML, AMD and Intel through DRM)."
GPU_INDEX_MESSAGE = "--gpu must be a non-negative index"
INTERVAL_MESSAGE = "--interval must be a finite number greater than zero"

# Per-process counter names (members of proc_gpu["available"])
COUNTER_DEDICATED = "dedicated"
COUNTER_SHARED = "shared"
COUNTER_ENGINE = "engine"

# Engine names (keys of proc_gpu["engines"])
ENGINE_3D = "3d"
ENGINE_COMPUTE = "compute"
ENGINE_COPY = "copy"
ENGINE_VIDEO = "video"
ENGINE_ENCODE = "encode"
ENGINE_DECODE = "decode"
ENGINE_OTHER = "other"

# Model file evidence sources
EVIDENCE_API = "api"
EVIDENCE_MAPPED = "mapped"
EVIDENCE_CMDLINE = "cmdline"
EVIDENCE_PATH_ARG = "path-arg"
EVIDENCE_INSTALLED = "installed"
PROVING_EVIDENCE = (EVIDENCE_API, EVIDENCE_MAPPED, EVIDENCE_CMDLINE)

# Owner inference inside a container with its own PID namespace
INFERRED_CONSISTENT = "consistent"
INFERRED_ORDER = "order"
INFERRED_MARKS = {
    INFERRED_CONSISTENT: "*",
    INFERRED_ORDER: "?",
}
INFERRED_NOTES = {
    INFERRED_CONSISTENT: (
        "* matched: the size is exact (NVML's figure for a host PID seen from inside a "
        "container); which local process owns it is inferred, and only one explanation fits."
    ),
    INFERRED_ORDER: (
        "? order guess: the size is exact but several owners fit, so newer local processes "
        "are paired with newer host PIDs and outside ones are assumed oldest. Becomes * with "
        "evidence, such as a process starting while gputop runs."
    ),
}

# Reasons why a DRM row has no measured memory
DRM_REASON_LEGACY = "legacy"
DRM_REASON_SHARED_CLIENT = "shared_client"
DRM_REASON_SHARED_BUFFERS = "shared_buffers"
DRM_REASON_PRIMARY = "primary"
DRM_REASON_INCOMPLETE = "incomplete"
DRM_REASON_PARTIAL_SCAN = "partial_scan"
DRM_REASON_NOTES = {
    DRM_REASON_LEGACY: (
        "VRAM n/a: this driver's fdinfo has missing or unparseable drm-shared-* stats (older "
        "amdgpu kernels, Intel integrated GPUs, xe), so resident memory may include buffers "
        "imported from other processes and would be counted once per importer."
    ),
    DRM_REASON_SHARED_CLIENT: (
        "VRAM and GPU% n/a: this process holds a DRM client other processes hold too (fd "
        "passing, fork). fdinfo counts the client once and doesn't say which of them uses it."
    ),
    DRM_REASON_SHARED_BUFFERS: (
        "VRAM n/a: this process's DRM client shares buffers with other clients (dma-buf, e.g. "
        "windows and video frames). fdinfo counts shared buffers by where they prefer to live, "
        "not where they are, so its private part can't be measured."
    ),
    DRM_REASON_PRIMARY: (
        "VRAM n/a: this process opened the GPU through /dev/dri/card*, which root-owned "
        "services such as logind may also hold where a normal user can't see them; run as root "
        "to check."
    ),
    DRM_REASON_INCOMPLETE: (
        "VRAM and GPU% n/a: some of this process's file descriptors or fdinfo couldn't be read."
    ),
    DRM_REASON_PARTIAL_SCAN: (
        "VRAM and GPU% n/a this refresh: the scan budget ran out before every process's fds "
        "were listed, so another holder of this DRM client can't be ruled out."
    ),
}

# Fixed note texts
KFD_NOTE = (
    "Engines n/a: this process uses ROCm/HIP (/dev/kfd), whose compute queues the kernel "
    "doesn't time per process. The device's AMD gfx load includes its work."
)
RESIDENT_NOTE = (
    "Resident: every buffer the process can reach, shared ones included. The same buffer can "
    "appear under several processes, so these are not added up."
)
INTEL_MEMORY_NOTE = (
    "Intel VRAM used is the sum of memory resident in the processes gputop can inspect (the "
    "driver has no device-wide counter): local memory on a discrete card, system memory on an "
    "integrated GPU. Buffers shared between processes can be counted twice, and kernel-owned "
    "memory and processes it can't inspect are missing. The total is the size of the card's "
    "local-memory PCI window, an estimate that can differ from its memory (0 on an integrated "
    "GPU, whose total is unknown), so free memory is not shown."
)

# Placeholders and small vocabularies
OUTSIDE_NAME = "(outside this container)"
UNKNOWN_NAME = "?"
NO_READING = "…"
HOST_OUTSIDE = "outside"
SERVER_SOURCE_PROCESS = "process"
SERVER_SOURCE_PARENT = "parent"
VLLM_ROLE_WORKER = "worker"
VLLM_ROLE_SERVER = "server"
PIDNS_INITIAL = "initial"
PIDNS_CHILD = "child"
PIDNS_UNKNOWN = "unknown"

# Runtime labels the classifier tests
RUNTIME_LLAMA_CPP = "llama.cpp"
RUNTIME_CUDA = "CUDA"
RUNTIME_HIP = "HIP"
RUNTIME_VULKAN = "Vulkan"
RUNTIME_D3D12 = "D3D12"
RUNTIME_NVENC = "NVENC"

# Process kinds
KIND_LLM = "LLM"
KIND_DECISION = "decision"
KIND_EMBEDDING = "embedding"
KIND_SPEECH = "speech"
KIND_DIFFUSION = "diffusion"
KIND_AI = "AI"
KIND_COMPUTE = "compute"
KIND_GAME = "3D/game"
KIND_VIDEO = "video"
KIND_BROWSER = "browser"
KIND_APP = "app"
KIND_COMPOSITOR = "compositor"
KIND_EXTERNAL = "external"
AI_KINDS = (KIND_LLM, KIND_DECISION, KIND_EMBEDDING, KIND_SPEECH, KIND_DIFFUSION, KIND_AI)
KIND_STYLES = {
    KIND_LLM: "bold magenta",
    KIND_DECISION: "bold orchid",
    KIND_EMBEDDING: "plum2",
    KIND_SPEECH: "light_sky_blue1",
    KIND_DIFFUSION: "orchid",
    KIND_AI: "medium_purple1",
    KIND_COMPUTE: "orange3",
    KIND_GAME: "green3",
    KIND_VIDEO: "yellow",
    KIND_BROWSER: "dodger_blue2",
    KIND_APP: "cyan",
    KIND_COMPOSITOR: "grey50",
    KIND_EXTERNAL: "grey50",
}
KIND_FALLBACK_STYLE = "white"
UNATTRIBUTED_STYLE = "grey35"
FREE_STYLE = "grey27"
KIND_INSIDE_TEXT = {
    KIND_COMPOSITOR: "desktop back buffers + composed frames",
    KIND_EXTERNAL: "a process in another PID namespace (host or another container)",
    KIND_BROWSER: "page textures, video, canvas/WebGL",
    KIND_VIDEO: "video frames (decode/encode)",
    KIND_GAME: "textures, meshes, render targets",
}

# GGUF model kinds
GGUF_KIND_LLM = "LLM"
GGUF_KIND_DECISION = "decision model"
GGUF_KIND_EMBEDDING = "embedding model"
GGUF_KIND_RERANKER = "reranker"
GGUF_KIND_SPEECH = "speech model"
GGUF_KIND_PROJECTOR = "vision projector"
GGUF_KIND_LORA = "LoRA adapter"
GGUF_KIND_MODEL = "model"
GGUF_SIDE_KINDS = (GGUF_KIND_PROJECTOR, GGUF_KIND_LORA)

# Segments of the LLM memory breakdown
SEGMENT_WEIGHTS = "weights"
SEGMENT_KV = "kv"
SEGMENT_COMPUTE = "compute"
SEGMENT_RUNTIME = "runtime"
SEGMENT_UNKNOWN = "unknown"
SEGMENT_ORDER = (SEGMENT_WEIGHTS, SEGMENT_KV, SEGMENT_COMPUTE, SEGMENT_RUNTIME, SEGMENT_UNKNOWN)
SEGMENT_STYLES = {
    SEGMENT_WEIGHTS: "magenta",
    SEGMENT_KV: "dark_orange",
    SEGMENT_COMPUTE: "yellow",
    SEGMENT_RUNTIME: "grey58",
    SEGMENT_UNKNOWN: "grey42",
}

# Local server kinds
SERVER_LLAMA = "llama-server"
SERVER_OLLAMA = "ollama"
SERVER_LMSTUDIO = "lmstudio"
SERVER_VLLM = "vllm"

# Thresholds and budgets
DEEP_SCAN_BUDGET = 0.15
DRM_SCAN_BUDGET = 0.05
DEEP_SCAN_TOP_ROWS = 25
DEEP_SCAN_MIN_GPU_PCT = 2.0
LIVE_MIN_DEDICATED = MIB
LIVE_MIN_GPU_PCT = 0.5
LIVE_MIN_SHARED = 16 * MIB
REPORT_MAX_ROWS = 30
DIFF_NOISE_BYTES = 16 * MIB
PID_REUSE_TOLERANCE = 2.0
HISTORY_LENGTH = 60
WARMUP_SECONDS = 1.0
TUI_WARMUP_SECONDS = 0.5
SPILL_WARN_BYTES = 256 * MIB
SPILL_KINDS = (KIND_LLM, KIND_AI)

# Rich style of secondary text in the TUI
MUTED_STYLE = "grey62"
SYSTEM_TOP_ROWS = 10


def main():
    pass


if __name__ == "__main__":
    main()
