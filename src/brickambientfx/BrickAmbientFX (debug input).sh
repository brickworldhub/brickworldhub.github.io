#!/bin/bash
# Same as BrickAmbientFX.sh, but shows and logs the number of every button you press.
# Use it once if a button does nothing, then edit brickambientfx/controls.json.
exec "$(dirname "$(readlink -f "$0")")/BrickAmbientFX.sh" --debug-input
