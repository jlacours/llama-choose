# llama-choose

`llama-choose` is a terminal picker and launcher for local llama.cpp models. It reads the same `llama-models.ini` preset file as llama.cpp, discovers additional GGUFs on disk, tracks usage and throughput, and runs small chat/code benchmarks.

## Install

On Arch Linux:

```bash
yay -S --needed rust curl tailscale
git clone https://github.com/jlacours/llama-choose.git
cd llama-choose
./install.sh
```

The installer builds with Cargo and copies `llama-choose` to `~/.local/bin`. A local llama.cpp build at `~/repos/llama.cpp/build/bin` is preferred; otherwise `llama-server` and `llama-cli` are resolved from `PATH`.

## Configuration

Models are read from `~/.local/share/llama-models.ini`. The optional llama.cpp web UI configuration is read from `~/.config/llama.cpp/ui-config.json`.

```bash
llama-choose                        # interactive picker
llama-choose launch ALIAS tools     # launch one model with built-in tools
llama-choose launch ALIAS server    # launch one model without built-in tools
llama-choose check [ALIAS]          # validate model headers and split files
llama-choose stats                  # usage and throughput
llama-choose bench ALIAS chat       # correctness benchmark
llama-choose stop                   # stop active servers
```

Run `llama-choose --help` for the complete command summary.
