#!/bin/sh
# Cross-compiles bsdsp for the TrimUI Brick (aarch64 Linux) and Linux x86_64 (desktop testing) with zig (pip install ziglang).
set -e
cd "$(dirname "$0")"
ZIG="python3 -m ziglang"
FLAGS="-O3 -fno-math-errno -ffp-contract=fast -fvisibility=hidden -shared -fPIC"
$ZIG cc -target aarch64-linux-gnu.2.17 $FLAGS -o libbsdsp_aarch64.so bsdsp.c -lm -lpthread -ldl
$ZIG cc -target x86_64-linux-gnu.2.17 $FLAGS -o libbsdsp_x86_64.so bsdsp.c -lm -lpthread -ldl

rm -f *.lib *.pdb
ls -la *.so
