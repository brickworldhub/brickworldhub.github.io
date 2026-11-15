#!/bin/bash
# Same as BrickFX.sh, but shows and logs the number of every button you press.
# Use it once if a button does nothing, then edit brickfx/controls.json.
exec "$(dirname "$(readlink -f "$0")")/BrickFX.sh" --debug-input
