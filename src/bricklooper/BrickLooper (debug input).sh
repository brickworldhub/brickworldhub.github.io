#!/bin/bash
# Same as BrickLooper.sh, but shows and logs the number of every button you press.
# Use it once if a button does nothing, then edit bricklooper/controls.json.
exec "$(dirname "$(readlink -f "$0")")/BrickLooper.sh" --debug-input
