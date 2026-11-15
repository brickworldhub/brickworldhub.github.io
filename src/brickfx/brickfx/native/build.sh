#!/bin/sh
# Cross-compiles bfx for the TrimUI Brick (aarch64 Linux) and Linux x86_64 with zig (pip install ziglang).
set -e
cd "$(dirname "$0")"
ZIG="python3 -m ziglang"
FLAGS="-O3 -fno-math-errno -ffp-contract=fast -fvisibility=hidden -shared -fPIC"
$ZIG cc -target aarch64-linux-gnu.2.17 $FLAGS -o libbfx_aarch64.so bfx.c -lm -lpthread -ldl
$ZIG cc -target x86_64-linux-gnu.2.17 $FLAGS -o libbfx_x86_64.so bfx.c -lm -lpthread -ldl
ls -la *.so
