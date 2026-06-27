#!/usr/bin/env bash

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="${XDG_BIN_HOME:-$HOME/.local/bin}"

command -v cargo >/dev/null 2>&1 || {
  printf 'error: cargo is required (install with: yay -S rust)\n' >&2
  exit 1
}

cargo build --release --locked --manifest-path "$root/Cargo.toml"
if [[ -L "$bin_dir/llama-choose" ]]; then
  unlink "$bin_dir/llama-choose"
fi
install -Dm755 "$root/target/release/llama-choose" "$bin_dir/llama-choose"
printf 'installed %s\n' "$bin_dir/llama-choose"
