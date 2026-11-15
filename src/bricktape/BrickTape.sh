#!/bin/sh
# BrickTape launcher for KNULLI (TrimUI Brick).
# Lives in /userdata/roms/ports/ next to the bricktape/ folder.

DIR="$(dirname "$(readlink -f "$0")")/bricktape"
cd "$DIR" || exit 1
LOG="$DIR/log.txt"
[ -f "$LOG" ] && mv -f "$LOG" "$DIR/log_previous.txt"

# Write the log to the SD card every 2 s, so it survives a forced power-off.
( while sleep 2; do sync; done ) &
SYNCER=$!

{
  echo "=== BrickTape $(date)"
  uname -a
  echo "python: $(command -v python3)"
  python3 --version
  echo "aplay: $(command -v aplay)  arecord: $(command -v arecord)"
  echo "sound devices:"; ls /dev/snd 2>/dev/null
  aplay -l 2>&1 | head -20
  arecord -l 2>&1 | head -20
} > "$LOG" 2>&1

# First run: unpack the bundled NumPy (KNULLI doesn't ship it).
if [ ! -f "$DIR/vendor/numpy/__init__.py" ] && [ -f "$DIR/vendor_numpy.zip" ]; then
  echo "unpacking NumPy..." >> "$LOG"
  rm -rf "$DIR/vendor"
  python3 -m zipfile -e "$DIR/vendor_numpy.zip" "$DIR/vendor" >> "$LOG" 2>&1 && echo "NumPy unpacked" >> "$LOG"
  sync
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 bricktape.py "$@" >> "$LOG" 2>&1
echo "=== exited with code $? at $(date)" >> "$LOG"
kill $SYNCER 2>/dev/null
sync
