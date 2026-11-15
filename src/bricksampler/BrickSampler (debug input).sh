#!/bin/bash
# Same as BrickSampler.sh, but shows and logs the number of every button you press.
# Use it once if a button does nothing, then edit bricksampler/controls.json.
exec "$(dirname "$(readlink -f "$0")")/BrickSampler.sh" --debug-input
