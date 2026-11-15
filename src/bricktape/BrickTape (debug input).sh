#!/bin/bash
# Same as BrickTape.sh, but shows and logs the number of every button you press.
# Use it once if a button does nothing, then edit bricktape/controls.json.
exec "$(dirname "$(readlink -f "$0")")/BrickTape.sh" --debug-input
