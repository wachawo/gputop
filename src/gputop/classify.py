#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process kind, vLLM signature, Ollama runner test and the "what's inside" summary."""
import re

# Local imports
from gputop import cmdline
from gputop import constants
from gputop import formatting
from gputop import modelfiles
from gputop import runtimes

# Name tables (compared with the lower-cased process name)
COMPOSITOR_NAMES = (
    "xorg", "xwayland", "gnome-shell", "kwin_wayland", "kwin_x11", "sway", "hyprland", "mutter",
    "weston", "wayfire", "labwc", "niri", "cosmic-comp", "picom", "compiz",
)
BROWSER_NAMES = (
    "chrome", "chromium", "chromium-browser", "firefox", "firefox-bin", "brave", "msedge",
    "opera", "vivaldi-bin", "msedge.exe", "chrome.exe", "firefox.exe", "msedgewebview2.exe",
    "brave.exe", "opera.exe", "arc.exe", "vivaldi.exe",
)
LLM_NAME_PREFIXES = (
    "llama-", "llama.", "ollama", "lm studio", "lmstudio", "llmster", "koboldcpp", "jan-",
    "gpt4all", "llamafile",
)
LLM_EXACT_NAMES = ("jan",)

# GGUF kind of a proven model to process kind, in priority order
MODEL_KIND_PRIORITY = (
    constants.KIND_LLM, constants.KIND_DECISION, constants.KIND_EMBEDDING, constants.KIND_SPEECH,
)
GGUF_KIND_TO_KIND = {
    constants.GGUF_KIND_LLM: constants.KIND_LLM,
    constants.GGUF_KIND_DECISION: constants.KIND_DECISION,
    constants.GGUF_KIND_EMBEDDING: constants.KIND_EMBEDDING,
    constants.GGUF_KIND_RERANKER: constants.KIND_EMBEDDING,
    constants.GGUF_KIND_SPEECH: constants.KIND_SPEECH,
}

# Engine thresholds, percent
VIDEO_MIN_PCT = 1.0
GAME_MIN_3D_PCT = 10.0

# vLLM signature
VLLM_WORKER_PATTERN = re.compile(r"^VLLM::(EngineCore|Worker|Worker_TP[0-9]+)$")
PYTHON_PATTERN = re.compile(r"^python([0-9]+(\.[0-9]+)*)?$")
PYTHON_SWITCHES = ("-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO")
VLLM_PROGRAM = "vllm"
VLLM_SERVE = "serve"
VLLM_MODULE = "vllm.entrypoints.openai.api_server"
VLLM_WATCHED_FLAGS = (
    "--config", "--headless", "--node-rank", "--data-parallel-size", "--api-server-count",
    "--nnodes", "--distributed-executor-backend", "--pipeline-parallel-size",
)
VLLM_FORBIDDEN_FLAGS = ("--config", "--headless", "--node-rank")
VLLM_SINGLE_NODE_VALUES = {
    "--data-parallel-size": "1",
    "-dp": "1",
    "--api-server-count": "1",
    "--nnodes": "1",
    "-n": "1",
    "--distributed-executor-backend": "mp",
    "--pipeline-parallel-size": "1",
    "-pp": "1",
}
VLLM_SHORT_FLAGS = ("-dp", "-pp", "-n")
VLLM_DATA_PARALLEL_PREFIX = "--data-parallel"
VLLM_DATA_PARALLEL_FLAG = "--data-parallel-size"
VLLM_RANK_PREFIX = "-r"

# Ollama runner
OLLAMA_PATH_PART = "/ollama/"
OLLAMA_NAME_PREFIX = "ollama"
OLLAMA_RUNNER_ARGUMENT = "runner"

# Summary
INSIDE_MAX_ITEMS = 2
PATH_ARG_PREFIX = "path-arg (unproven): "
INSTALLED_PREFIX = "installed (unproven): "


def has_detail_evidence(detail: dict) -> bool:
    """Tell whether any model of the detail has model evidence."""
    for model in detail["models"]:
        if modelfiles.has_model_evidence(model):
            return True
    return False


def is_ai_detail(detail: dict) -> bool:
    """Tell whether the process has model evidence or an AI runtime loaded."""
    if has_detail_evidence(detail):
        return True
    for label in detail["runtimes"]:
        if label in runtimes.AI_RUNTIMES:
            return True
    return False


def get_basename(text: str) -> str:
    """Return the text after the last slash."""
    return text.rsplit("/", 1)[-1]


def is_vllm_server_cmdline(arguments: list[str]) -> bool:
    """Tell whether the argument vector starts a vLLM API server."""
    if len(arguments) < 2:
        return False
    program = get_basename(arguments[0])
    if program == VLLM_PROGRAM:
        return arguments[1] == VLLM_SERVE
    if PYTHON_PATTERN.match(program) is None:
        return False
    if len(arguments) >= 3 and get_basename(arguments[1]) == VLLM_PROGRAM:
        if arguments[2] == VLLM_SERVE:
            return True
    position = 1
    while position < len(arguments) and arguments[position] in PYTHON_SWITCHES:
        position += 1
    return arguments[position:position + 2] == ["-m", VLLM_MODULE]


def get_vllm_role(detail: dict) -> str:
    """Return the vLLM role of a process by its signature: worker, server or ""."""
    arguments = detail["cmdline"]
    first_argument = arguments[0] if arguments else ""
    for title in (detail["name"], first_argument):
        if VLLM_WORKER_PATTERN.match(title) is not None:
            return constants.VLLM_ROLE_WORKER
    if is_vllm_server_cmdline(arguments):
        return constants.VLLM_ROLE_SERVER
    return ""


def is_watched_prefix(flag: str) -> bool:
    """Tell whether a long flag is a strict prefix of a watched vLLM flag."""
    if not flag.startswith("--") or len(flag) <= 2:
        return False
    for watched in VLLM_WATCHED_FLAGS:
        if watched != flag and watched.startswith(flag):
            return True
    return False


def is_ambiguous_flag(flag: str) -> bool:
    """Tell whether a flag alone marks a multi-process or file-configured vLLM layout."""
    if is_watched_prefix(flag) or flag in VLLM_FORBIDDEN_FLAGS:
        return True
    if flag.startswith(VLLM_DATA_PARALLEL_PREFIX) and flag != VLLM_DATA_PARALLEL_FLAG:
        return True
    for short_flag in VLLM_SHORT_FLAGS:
        if flag.startswith(short_flag) and flag != short_flag:
            return True
    return flag.startswith(VLLM_RANK_PREFIX)


def is_vllm_ambiguous(arguments: list[str]) -> bool:
    """Tell whether the vLLM launch flags describe a layout whose API port cannot be attributed."""
    for position, argument in enumerate(arguments):
        flag, value = cmdline.split_flag(argument)
        if flag.startswith("--"):
            flag = flag.replace("_", "-")
        if is_ambiguous_flag(flag):
            return True
        allowed = VLLM_SINGLE_NODE_VALUES.get(flag)
        if allowed is None:
            continue
        if value is None:
            following = arguments[position + 1:position + 2]
            value = following[0] if following else ""
        if value != allowed:
            return True
    return False


def find_model_kind(detail: dict) -> str:
    """Return the kind implied by the GGUF kinds of the proven models, or ""."""
    found = set()
    for model in detail["models"]:
        if not modelfiles.has_model_evidence(model):
            continue
        kind = GGUF_KIND_TO_KIND.get(model["kind"])
        if kind is not None:
            found.add(kind)
    for kind in MODEL_KIND_PRIORITY:
        if kind in found:
            return kind
    return ""


def has_llm_name(lowered_name: str) -> bool:
    """Tell whether a lower-cased process name belongs to a known LLM runner."""
    if lowered_name in LLM_EXACT_NAMES:
        return True
    return lowered_name.startswith(LLM_NAME_PREFIXES)


def is_doing_video(engines: dict[str, float]) -> bool:
    """Tell whether the decode, encode and video engines together are busy."""
    decode = engines.get(constants.ENGINE_DECODE, 0.0)
    encode = engines.get(constants.ENGINE_ENCODE, 0.0)
    video = engines.get(constants.ENGINE_VIDEO, 0.0)
    return decode + encode + video > VIDEO_MIN_PCT


def classify_ai(detail: dict, lowered_name: str, doing_video: bool) -> str:
    """Apply the model, LLM and AI rules; "" when none of them matches."""
    if cmdline.is_diffusion_cmdline(detail["cmdline"]):
        return constants.KIND_DIFFUSION
    model_kind = find_model_kind(detail)
    if model_kind:
        return model_kind
    loaded = detail["runtimes"]
    if get_vllm_role(detail) or constants.RUNTIME_LLAMA_CPP in loaded or has_llm_name(lowered_name):
        return constants.KIND_LLM
    if is_ai_detail(detail) and not doing_video:
        return constants.KIND_AI
    return ""


def classify_graphics(detail: dict, lowered_name: str, engines: dict, doing_video: bool) -> str:
    """Apply the browser, video, compute and game rules; the fallback is the app kind."""
    loaded = detail["runtimes"]
    if lowered_name in BROWSER_NAMES:
        return constants.KIND_BROWSER
    if doing_video or constants.RUNTIME_NVENC in loaded:
        return constants.KIND_VIDEO
    if constants.RUNTIME_CUDA in loaded or constants.RUNTIME_HIP in loaded:
        return constants.KIND_COMPUTE
    has_3d_runtime = constants.RUNTIME_D3D12 in loaded or constants.RUNTIME_VULKAN in loaded
    if has_3d_runtime and engines.get(constants.ENGINE_3D, 0.0) > GAME_MIN_3D_PCT:
        return constants.KIND_GAME
    return constants.KIND_APP


def classify_process(detail: dict, engines: dict[str, float]) -> str:
    """Return the kind of a process from its detail and its per-engine load; first rule wins."""
    lowered_name = detail["name"].lower()
    if lowered_name in COMPOSITOR_NAMES:
        return constants.KIND_COMPOSITOR
    doing_video = is_doing_video(engines)
    ai_kind = classify_ai(detail, lowered_name, doing_video)
    if ai_kind:
        return ai_kind
    return classify_graphics(detail, lowered_name, engines, doing_video)


def is_ollama_runner(detail: dict, kind: str) -> bool:
    """Tell whether an AI row is an Ollama model runner process."""
    if kind not in constants.AI_KINDS:
        return False
    lowered_exe = detail["exe"].lower()
    if OLLAMA_PATH_PART in lowered_exe:
        return True
    lowered_name = detail["name"].lower()
    if not lowered_name.startswith(OLLAMA_NAME_PREFIX):
        return False
    return OLLAMA_RUNNER_ARGUMENT in detail["cmdline"]


def describe_models(models: list[dict]) -> str:
    """Summarise model files: proven ones first, at most two, hints marked as unproven."""
    proven = [model for model in models if modelfiles.has_model_evidence(model)]
    unproven = [model for model in models if not modelfiles.has_model_evidence(model)]
    parts = []
    for model in (proven + unproven)[:INSIDE_MAX_ITEMS]:
        prefix = "" if modelfiles.has_model_evidence(model) else PATH_ARG_PREFIX
        parts.append(prefix + modelfiles.get_short_name(model))
    return ", ".join(parts)


def describe_candidates(candidates: list[dict]) -> str:
    """Summarise installed candidates: at most two, each with its size."""
    parts = []
    for model in candidates[:INSIDE_MAX_ITEMS]:
        size_text = formatting.format_size(model["size"])
        parts.append(f"{modelfiles.get_short_name(model)} {size_text}")
    return INSTALLED_PREFIX + ", ".join(parts)


def describe_inside(detail: dict, kind: str, view: dict | None) -> str:
    """Return the one-line summary of what a process holds on the GPU; first match wins."""
    if view is not None and view["summary"]:
        return view["summary"]
    if detail["models"]:
        return describe_models(detail["models"])
    if detail["candidates"]:
        return describe_candidates(detail["candidates"])
    return constants.KIND_INSIDE_TEXT.get(kind, "")


def main():
    pass


if __name__ == "__main__":
    main()
