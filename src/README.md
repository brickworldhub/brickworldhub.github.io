# BrickWorld source code

The source of the BrickWorld apps, GNU GPL v3 (see [LICENSE](../LICENSE)). The ready-to-use zips are in [`downloads/`](../downloads/).

- [`bricktape/`](bricktape/) – BrickTape 1.0.0. `bricktape/bricktape.py` is the app (Python, Pygame, NumPy), `bricktape/native/btdsp.c` the audio engine; `bricktape/native/build.sh` builds `libbtdsp_aarch64.so` with zig. The zip also has the built library, NumPy (BSD) and the DejaVu fonts.
- [`bricklooper/`](bricklooper/) – BrickLooper 1.5.0. `bricklooper/bricklooper.py` is the app (Python, Pygame, NumPy), `bricklooper/native/bsdsp.c` the audio engine; `bricklooper/native/build.sh` builds `libbsdsp_aarch64.so` with zig. The zip also has the built library, NumPy (BSD) and the DejaVu fonts.
- [`brickambientfx/`](brickambientfx/) – BrickAmbientFX (Sarta Edition) 1.0.0. `brickambientfx/brickambientfx.py` is the app (Python, Pygame, NumPy), `brickambientfx/native/bafx.c` the audio engine; `brickambientfx/native/build.sh` builds `libbafx_aarch64.so` with zig. The zip also has the built library and NumPy (BSD). Changes: [`brickambientfx/CHANGELOG.md`](brickambientfx/CHANGELOG.md).
