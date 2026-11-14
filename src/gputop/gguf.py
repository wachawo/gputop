#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GGUF header reader with an LRU cache and the model properties derived from a header."""

import os
import re
import struct

# Local imports
from gputop import constants
from gputop import formatting

# File layout
GGUF_MAGIC = b"GGUF"
READ_BUFFER_BYTES = constants.MIB
DEFAULT_ALIGNMENT = 32
TYPE_STRING = 8
TYPE_ARRAY = 9
SCALAR_CODES = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

# Defensive caps: the file is untrusted input
MAX_KEY_BYTES = 64 * 1024
MAX_STRING_VALUE_BYTES = constants.MIB
MAX_KEPT_ARRAY_ITEMS = 4096
MAX_TENSOR_COUNT = 1_000_000
MAX_KV_COUNT = 100_000
MAX_DIMENSIONS = 8
MAX_TENSOR_ELEMENTS = 1 << 48
MAX_ARRAY_DEPTH = 8
MAX_NESTED_ITEMS = 1_000_000
TRUNCATED_MESSAGE = "truncated GGUF header"

# Cache of parsed headers, keyed by (path, file size, mtime); oldest entry first
CACHE_LIMIT = 64
GGUF_CACHE = {}

# Derived properties
BLOCK_NAME_PATTERN = re.compile(r"^blk\.(\d+)\.")
RERANKER_POOLING_TYPE = 4
FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
}


def create_gguf_info(path: str, file_size: int) -> dict:
    """Create an empty header info record.

    Keys: path (str), file_size (int, bytes), version (int, 0 until read), metadata (dict, key to
    value; long arrays and long strings are placeholder texts), n_params (int, element count of
    all tensors), tensor_bytes (int), layer_bytes (dict, block index to bytes), token_embd_bytes
    (int), output_bytes (int), other_bytes (int), error (str or None, None = parsed completely).
    """
    return {
        "path": path,
        "file_size": file_size,
        "version": 0,
        "metadata": {},
        "n_params": 0,
        "tensor_bytes": 0,
        "layer_bytes": {},
        "token_embd_bytes": 0,
        "output_bytes": 0,
        "other_bytes": 0,
        "error": None,
    }


def create_reader(handle, file_size: int) -> dict:
    """Create the reader state: handle, size (bytes), pos (bytes read), version, nested_left."""
    return {
        "handle": handle,
        "size": file_size,
        "pos": 0,
        "version": 0,
        "nested_left": MAX_NESTED_ITEMS,
    }


def read_exact(reader: dict, count: int) -> bytes:
    """Read exactly count bytes or fail with the truncation error."""
    if count < 0 or reader["pos"] + count > reader["size"]:
        raise ValueError(TRUNCATED_MESSAGE)
    chunk = reader["handle"].read(count)
    if len(chunk) != count:
        raise ValueError(TRUNCATED_MESSAGE)
    reader["pos"] += count
    return chunk


def skip_bytes(reader: dict, count: int) -> None:
    """Seek past count bytes, never beyond the end of the file."""
    if count < 0 or reader["pos"] + count > reader["size"]:
        raise ValueError(TRUNCATED_MESSAGE)
    reader["handle"].seek(count, os.SEEK_CUR)
    reader["pos"] += count


def read_scalar(reader: dict, value_type: int):
    """Read one fixed-size little-endian value of a scalar metadata type."""
    layout = "<" + SCALAR_CODES[value_type]
    value_bytes = read_exact(reader, struct.calcsize(layout))
    return struct.unpack(layout, value_bytes)[0]


def read_uint32(reader: dict) -> int:
    """Read one uint32."""
    return read_scalar(reader, 4)


def read_length(reader: dict) -> int:
    """Read a length integer: uint64 from version 2 on, uint32 in version 1."""
    if reader["version"] >= 2:
        return read_scalar(reader, 10)
    return read_scalar(reader, 4)


def read_name(reader: dict, what: str) -> str:
    """Read a short string (metadata key or tensor name), rejecting oversized ones."""
    length = read_length(reader)
    if length > MAX_KEY_BYTES:
        raise ValueError(f"{what} longer than {MAX_KEY_BYTES} bytes")
    name_bytes = read_exact(reader, length)
    return name_bytes.decode("utf-8", errors="replace")


def read_string_value(reader: dict) -> str:
    """Read a string value; an oversized one is skipped and replaced by a placeholder."""
    length = read_length(reader)
    if length > MAX_STRING_VALUE_BYTES:
        skip_bytes(reader, length)
        return f"<text[{length}]>"
    text_bytes = read_exact(reader, length)
    return text_bytes.decode("utf-8", errors="replace")


def read_scalar_array(reader: dict, element_type: int, count: int):
    """Keep a short scalar array as a list, skip a long one."""
    code = SCALAR_CODES[element_type]
    item_size = struct.calcsize("<" + code)
    if count > MAX_KEPT_ARRAY_ITEMS:
        skip_bytes(reader, count * item_size)
        return f"<array[{count}]>"
    array_bytes = read_exact(reader, count * item_size)
    values = struct.unpack(f"<{count}{code}", array_bytes)
    return list(values)


def skip_string_array(reader: dict, count: int) -> str:
    """Skip every string of an array so that vocabularies are never held in memory."""
    for index in range(count):
        length = read_length(reader)
        skip_bytes(reader, length)
    return f"<string[{count}]>"


def skip_nested_array(reader: dict, count: int, depth: int) -> str:
    """Parse and discard the elements of an array of arrays."""
    if depth >= MAX_ARRAY_DEPTH:
        raise ValueError("GGUF arrays are nested too deeply")
    for index in range(count):
        reader["nested_left"] -= 1
        if reader["nested_left"] < 0:
            raise ValueError("too many nested GGUF array elements")
        read_array(reader, depth + 1)
    return f"<array[{count}]>"


def read_array(reader: dict, depth: int):
    """Read one array value: a list of numbers or a placeholder text."""
    element_type = read_uint32(reader)
    count = read_length(reader)
    if element_type in SCALAR_CODES:
        return read_scalar_array(reader, element_type, count)
    if element_type == TYPE_STRING:
        return skip_string_array(reader, count)
    if element_type == TYPE_ARRAY:
        return skip_nested_array(reader, count, depth)
    raise ValueError(f"unknown GGUF value type {element_type}")


def read_value(reader: dict, value_type: int):
    """Read one metadata value of the given type."""
    if value_type in SCALAR_CODES:
        return read_scalar(reader, value_type)
    if value_type == TYPE_STRING:
        return read_string_value(reader)
    if value_type == TYPE_ARRAY:
        return read_array(reader, 0)
    raise ValueError(f"unknown GGUF value type {value_type}")


def read_metadata(reader: dict, kv_count: int, metadata: dict) -> None:
    """Fill metadata entry by entry, so a later failure keeps what was read."""
    for index in range(kv_count):
        key = read_name(reader, "metadata key")
        value_type = read_uint32(reader)
        metadata[key] = read_value(reader, value_type)


def read_tensor_table(reader: dict, tensor_count: int) -> list[tuple]:
    """Read the tensor infos as (data offset, name, element count) tuples."""
    tensors = []
    for index in range(tensor_count):
        name = read_name(reader, "tensor name")
        dimension_count = read_uint32(reader)
        if dimension_count > MAX_DIMENSIONS:
            raise ValueError(f"tensor with more than {MAX_DIMENSIONS} dimensions")
        element_count = 1
        for axis in range(dimension_count):
            element_count *= read_length(reader)
            if element_count > MAX_TENSOR_ELEMENTS:
                raise ValueError(f"tensor with more than {MAX_TENSOR_ELEMENTS} elements")
        read_uint32(reader)
        offset = read_scalar(reader, 10)
        tensors.append((offset, name, element_count))
    return tensors


def get_alignment(metadata: dict) -> int:
    """Alignment of the tensor-data region."""
    value = metadata.get("general.alignment")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return DEFAULT_ALIGNMENT


def add_tensor_bytes(info: dict, name: str, size: int) -> None:
    """Add the byte size of one tensor to the bucket its name selects."""
    info["tensor_bytes"] += size
    block_match = BLOCK_NAME_PATTERN.match(name)
    if block_match:
        block_index = int(block_match.group(1))
        info["layer_bytes"][block_index] = info["layer_bytes"].get(block_index, 0) + size
    elif name.startswith("token_embd."):
        info["token_embd_bytes"] += size
    elif name.startswith("output"):
        info["output_bytes"] += size
    else:
        info["other_bytes"] += size


def check_tensor_offsets(tensors: list[tuple], data_size: int) -> None:
    """Reject a tensor table that points past the end of the file."""
    if data_size < 0:
        raise ValueError(TRUNCATED_MESSAGE)
    for tensor in tensors:
        if tensor[0] > data_size:
            raise ValueError(f"tensor {tensor[1]!r} starts past the end of the file")


def accumulate_tensors(info: dict, tensors: list[tuple], data_size: int) -> None:
    """Derive tensor sizes from the offset table and fill the byte totals."""
    check_tensor_offsets(tensors, data_size)
    ordered = sorted(tensors, key=lambda tensor: tensor[0])
    has_output_weight = False
    for position, tensor in enumerate(ordered):
        offset, name, element_count = tensor
        if position + 1 < len(ordered):
            next_offset = ordered[position + 1][0]
        else:
            next_offset = data_size
        info["n_params"] += element_count
        add_tensor_bytes(info, name, max(next_offset - offset, 0))
        if name == "output.weight":
            has_output_weight = True
    if not has_output_weight and info["layer_bytes"]:
        info["output_bytes"] += info["token_embd_bytes"]


def parse_header(reader: dict, info: dict) -> None:
    """Parse magic, counts, metadata and tensor table into info; raises on any defect."""
    if reader["size"] < len(GGUF_MAGIC) or read_exact(reader, len(GGUF_MAGIC)) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    reader["version"] = read_uint32(reader)
    info["version"] = reader["version"]
    tensor_count = read_length(reader)
    kv_count = read_length(reader)
    if tensor_count > MAX_TENSOR_COUNT:
        raise ValueError(f"tensor count {tensor_count} is above {MAX_TENSOR_COUNT}")
    if kv_count > MAX_KV_COUNT:
        raise ValueError(f"metadata count {kv_count} is above {MAX_KV_COUNT}")
    read_metadata(reader, kv_count, info["metadata"])
    tensors = read_tensor_table(reader, tensor_count)
    alignment = get_alignment(info["metadata"])
    padded = reader["pos"] + alignment - 1
    data_start = padded - padded % alignment
    accumulate_tensors(info, tensors, reader["size"] - data_start)


def parse_file(path: str, file_size: int) -> dict:
    """Read the header of one file; a failure is stored in info["error"]."""
    info = create_gguf_info(path, file_size)
    try:
        with open(path, "rb", buffering=READ_BUFFER_BYTES) as handle:
            reader = create_reader(handle, file_size)
            parse_header(reader, info)
    except (OSError, ValueError, struct.error) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def is_gguf(path: str) -> bool:
    """True when the first four bytes of the file are the GGUF magic."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(GGUF_MAGIC)) == GGUF_MAGIC
    except OSError:
        return False


def read_gguf(path: str) -> dict | None:
    """Header info of the file, cached until its size or mtime changes; None when stat fails.

    The returned dict is shared with the cache and must not be modified by the caller.
    """
    try:
        status = os.stat(path)
    except OSError:
        return None
    cache_key = (path, status.st_size, status.st_mtime)
    cached = GGUF_CACHE.pop(cache_key, None)
    if cached is None:
        cached = parse_file(path, status.st_size)
    GGUF_CACHE[cache_key] = cached
    while len(GGUF_CACHE) > CACHE_LIMIT:
        oldest_key = next(iter(GGUF_CACHE))
        del GGUF_CACHE[oldest_key]
    return cached


def detect_gguf_kind(path: str) -> str:
    """GGUF kind of the file, or an empty string when its header is not usable."""
    info = read_gguf(path)
    if info is None or info["error"]:
        return ""
    return get_gguf_kind(info)


def get_arch_value(info: dict, suffix: str):
    """Value of the metadata key <architecture>.<suffix>, None when missing."""
    arch = info["metadata"].get("general.architecture")
    if not isinstance(arch, str):
        return None
    return info["metadata"].get(f"{arch}.{suffix}")


def get_gguf_name(info: dict) -> str:
    """general.name, else the basename of the file."""
    name = info["metadata"].get("general.name")
    if isinstance(name, str) and name:
        return name
    return os.path.basename(info["path"])


def get_gguf_arch(info: dict) -> str:
    """general.architecture, else a question mark."""
    arch = info["metadata"].get("general.architecture")
    if isinstance(arch, str) and arch:
        return arch
    return "?"


def get_gguf_quant(info: dict) -> str:
    """Quantisation label from general.file_type."""
    file_type = formatting.convert_to_int(info["metadata"].get("general.file_type"))
    if file_type is None:
        return "?"
    return FILE_TYPE_NAMES.get(file_type, f"type{file_type}")


def get_gguf_layers(info: dict) -> int:
    """Block count from the metadata, 0 when missing."""
    layers = formatting.convert_to_int(get_arch_value(info, "block_count"))
    if layers is None or layers < 0:
        return 0
    return layers


def get_gguf_ctx_train(info: dict) -> int | None:
    """Training context length, None when missing."""
    return formatting.convert_to_int(get_arch_value(info, "context_length"))


def get_gguf_sliding_window(info: dict) -> int | None:
    """Sliding attention window in tokens, None when missing."""
    return formatting.convert_to_int(get_arch_value(info, "attention.sliding_window"))


def is_gguf_causal(info: dict) -> bool:
    """False only when the header says the attention is not causal."""
    return get_arch_value(info, "attention.causal") is not False


def get_gguf_decision_type(info: dict) -> str | None:
    """Decision type of a decision model, None when missing."""
    value = get_arch_value(info, "decision.type")
    if value is None or value == "":
        return None
    return str(value)


def has_speech_keys(metadata: dict) -> bool:
    """True when any metadata key belongs to the speech-to-text group."""
    for key in metadata:
        if key.startswith("stt."):
            return True
    return False


def get_gguf_kind(info: dict) -> str:
    """Kind of file, first matching rule wins."""
    metadata = info["metadata"]
    general_type = metadata.get("general.type", "model")
    if get_gguf_arch(info) == "clip" or general_type == "mmproj":
        return constants.GGUF_KIND_PROJECTOR
    if general_type == "adapter":
        return constants.GGUF_KIND_LORA
    if get_gguf_decision_type(info) is not None:
        return constants.GGUF_KIND_DECISION
    pooling_type = get_arch_value(info, "pooling_type")
    if pooling_type is not None:
        if pooling_type == RERANKER_POOLING_TYPE:
            return constants.GGUF_KIND_RERANKER
        return constants.GGUF_KIND_EMBEDDING
    if has_speech_keys(metadata):
        return constants.GGUF_KIND_SPEECH
    has_context = get_gguf_ctx_train(info) is not None
    if get_gguf_layers(info) > 0 and has_context and is_gguf_causal(info):
        return constants.GGUF_KIND_LLM
    return constants.GGUF_KIND_MODEL


def format_params(n_params: int) -> str:
    """Parameter count as 7.6B, 137M or a plain integer."""
    if n_params >= 1e9:
        return f"{n_params / 1e9:.1f}B"
    if n_params >= 1e6:
        return f"{n_params / 1e6:.0f}M"
    return str(int(n_params))


def estimate_gpu_weight_bytes(info: dict, ngl: int | None) -> int:
    """Bytes of weights placed on the GPU when the last ngl blocks are offloaded.

    ngl None or negative means everything, output head included. Token embeddings and
    unclassified tensors stay on the CPU. A file without block tensors goes to the GPU whole.
    """
    layer_bytes = info["layer_bytes"]
    if not layer_bytes:
        return info["tensor_bytes"]
    layers = get_gguf_layers(info) or len(layer_bytes)
    offloaded = layers + 1 if ngl is None or ngl < 0 else ngl
    first_block = max(layers - offloaded, 0)
    total = 0
    for block_index, block_bytes in layer_bytes.items():
        if block_index >= first_block:
            total += block_bytes
    if offloaded > layers:
        total += info["output_bytes"]
    return total


def list_kv_heads(value, layers: int) -> list[int] | None:
    """Per-layer KV head counts from a scalar or a per-layer list; None when unusable."""
    if isinstance(value, list):
        heads = []
        for item in value[:layers]:
            count = formatting.convert_to_int(item)
            if count is None:
                return None
            heads.append(count)
        return heads
    count = formatting.convert_to_int(value)
    if count is None:
        return None
    return [count] * layers


def get_head_dim(info: dict, head_count) -> int | None:
    """Embedding length divided by the (largest) head count, None when either is unknown."""
    if isinstance(head_count, list):
        counts = [formatting.convert_to_int(item) for item in head_count]
        known = [count for count in counts if count is not None]
        head_count = max(known) if known else None
    heads = formatting.convert_to_int(head_count)
    embedding = formatting.convert_to_int(get_arch_value(info, "embedding_length"))
    if not heads or heads < 0 or embedding is None:
        return None
    return embedding // heads


def estimate_kv_elements_per_token(info: dict) -> int | None:
    """KV cache elements one context token costs over all layers; None when not derivable."""
    layers = get_gguf_layers(info)
    head_count = get_arch_value(info, "attention.head_count")
    kv_head_count = get_arch_value(info, "attention.head_count_kv")
    if kv_head_count is None:
        kv_head_count = head_count
    if layers <= 0 or kv_head_count is None:
        return None
    head_dim = get_head_dim(info, head_count)
    key_length = formatting.convert_to_int(get_arch_value(info, "attention.key_length")) or head_dim
    value_length = formatting.convert_to_int(get_arch_value(info, "attention.value_length")) or head_dim
    kv_heads = list_kv_heads(kv_head_count, layers)
    if not key_length or not value_length or kv_heads is None:
        return None
    elements = 0
    for heads in kv_heads:
        elements += heads * (key_length + value_length)
    return elements if elements > 0 else None


def main():
    pass


if __name__ == "__main__":
    main()
