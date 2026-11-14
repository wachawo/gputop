#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Help modal Markdown text of the gputop TUI."""

HELP_MARKDOWN = """\
# What can be known about GPU memory from outside a process?

gputop reads what the driver and the kernel publish and adds what can be proven from the
process itself. Some layers are measured, some are derived, and some stay closed. This page
says which is which.

## Slim and Full

`m` switches between **Full diagnostics** and **Slim monitoring**. Slim hides the driver
strings and the bus address on the name line, keeps only temperature, power draw and the
main clock on the sensors line, and reduces the System panel to its RAM and CPU bars. It is
a display choice only: sampling, charts, the table and the inspector stay the same.

## Layers

| Layer | Source | Works for |
|---|---|---|
| Device telemetry (load, memory, temperature, power, clocks) | NVML on NVIDIA cards, sysfs and hwmon on AMD and Intel cards | every supported card |
| Per-process VRAM and engine load | NVML process lists and process utilisation, DRM fdinfo on AMD and Intel | processes the driver accounts for and this user may read |
| Which runtime | shared libraries mapped into the process, from `/proc/<pid>/maps` | CUDA, HIP, Vulkan, llama.cpp, ONNX Runtime, PyTorch and similar |
| Which model file | memory-mapped files plus the command line | runtimes that map their weights or name them in a launch flag |
| Model name and kind | GGUF header, local model stores, local server APIs | GGUF files, Ollama, Hugging Face and LM Studio stores, llama.cpp server, Ollama, LM Studio, vLLM |
| Weights, KV cache and compute split | GGUF header arithmetic, runtime load logs, server APIs | llama.cpp style runtimes |
| Names of single textures or buffers | not available without a debugger attached to the process | nothing from outside |

## Evidence tags

Each tag is shown in the inspector under the label given here.

- `api`, shown as `reported (api)`: a local inference server reported the model as loaded.
- `mapped`, shown as `in use (mapped)`: the file is memory-mapped into the process.
- `cmdline`, shown as `launch arg`: a launch flag of the process names the file.
- `path-arg`, shown as `path-arg (not proven loaded)`: a path among the arguments points at
  the file. This is a hint, not a proof.
- `installed`, shown as `installed, not proven loaded`: the model is installed next to the
  runtime. Nothing shows that it is loaded.
- `confirmed (tag+tag)`, for example `confirmed (api+mapped)`: two or more of the proving
  tags `api`, `mapped` and `cmdline` agree on the file.

A model counts as proven when it carries `api`, `mapped` or `cmdline`. Entries with only
`path-arg` or `installed` are listed as unproven.

## Scope

gputop looks at local workloads only: processes on this machine and servers that listen on
localhost. Mapped libraries identify a runtime, not a model, so a process can be labelled
`AI` without any model name.

## Linux specifics

- On NVIDIA cards the per-process numbers come from NVML. On AMD and Intel cards they come
  from the DRM fdinfo files of each process.
- A cell shows `n/a` when the number cannot be read or cannot be charged to one process:
  processes of other users without root, or memory shared between several clients that has
  no single owner.
- An `H:` row is a process NVML reports from another PID namespace, seen when gputop runs
  inside a container. It has no local details.
- Inside a container the size of such a process is exact, but its local owner is inferred.
  `*` after a size means only one owner fits. `?` means several owners fit and newer
  processes were paired with newer GPU PIDs, which is a guess.
- Running as root reveals more: other users' processes, their mapped files and their
  command lines.

## What's inside

The last table column and the inspector are inferred from runtimes and files. Sizes taken
from a load log are what the runtime allocated at load time. They are not proof that the
memory is still resident.

## GPU%

The `GPU%` of a process is its busiest engine, not a sum. The device `GPU` load is the share
of time the card was executing work. `Mem activity` is the share of time device memory was
being read or written. It is not memory occupancy.

## System panel, RAM and CPU%

The System panel and the `RAM` and `CPU%` columns are reported by the operating system
through psutil. `RAM` is resident memory including shared pages, so the rows can add up to
more than the machine uses. `CPU%` is a share of the whole machine. `n/a` means the system
would not say. `…` means the process needs one more sample before it can be rated. These
figures are system memory and CPU, never GPU memory, and they are in no GPU total. Start
gputop with `--no-system` to hide them.

## Keys

`q` quit - `s` sort - `c` chart - `a` all/top - `m` slim/full - `p` pause - `?` this page -
arrows, PageUp and PageDown move the selection

## Charts

Eight charts show the last 60 samples of each GPU: load, main clock, VRAM, memory activity,
temperature, power, encoder and decoder. The grid uses three columns in a wide terminal,
two in a medium one and one in a narrow one. Pick a single chart from the dropdown, or
press `c` to cycle through them. A blank cell means no reading and is not a zero. Percent
charts run from 0 to 100 %, temperature from 0 to 120 °C. Power and clock scale to the
device limit or to the highest value seen, and the note under the grid states both tops.
"""


def main():
    pass


if __name__ == "__main__":
    main()
