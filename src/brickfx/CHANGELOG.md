# BrickFX changes

## 1.0.2 (2026-11-18)
- USB inputs that gave silence now work: when a USB device is opened, any capture switch that is off is switched on, and a capture level left at 0 is set to 0 dB (unity). On the Brick nothing did this, so some adapters (e.g. a line in) only sent zeros. The device's mixer is written to log.txt.
- Each USB input opens in its own best format (24-bit when it has it, falling back to 16-bit) and at a rate it really supports (48 kHz if it can, otherwise its first rate, converted to 48 kHz).
- Each USB device remembers its own USB INPUT: switch to another card and back, and it is on the input you chose before.
- log.txt shows the inputs of every card at the start, format/channels/rate/period/buffer of the opened input, and the real level of each channel 2 s after it opens (or "silent (all 0)").
- Status shows "USB n IN m".

## 1.0.1 (2026-11-18)
- I/O page: new USB INPUT row for USB devices with more than one input (for example mic in and line in): Input 1, Input 2... Only the inputs the device really has are listed; with one input it stays on Input 1. If the saved input is gone, the first one is used.
- Switching input fades out and back in, with no click; delay and reverb tails carry on.

## 1.0.0 (2026-11-15)
- First stable release: DRIVE > MOD > DELAY > REVERB > EQ, smooth switching, tails, oscilloscope, 8 presets, MIDI CC learn, clock and program change.
