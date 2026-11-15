#!/bin/sh
# Cross-compiles btdsp (BrickTape engine) for the TrimUI Brick (aarch64 Linux), Windows and Linux x86_64 with zig (pip install ziglang).
set -e
cd "$(dirname "$0")"
ZIG="python3 -m ziglang"
FLAGS="-O3 -fno-math-errno -ffp-contract=fast -fvisibility=hidden -shared -fPIC"
$ZIG cc -target aarch64-linux-gnu.2.17 $FLAGS -o libbtdsp_aarch64.so btdsp.c -lm -lpthread -ldl
$ZIG cc -target x86_64-linux-gnu.2.17 $FLAGS -o libbtdsp_x86_64.so btdsp.c -lm -lpthread -ldl
$ZIG cc -target x86_64-windows-gnu -O3 -fno-math-errno -ffp-contract=fast -shared -o btdsp_x64.dll btdsp.c
rm -f *.lib *.pdb
ls -la *.so *.dll
