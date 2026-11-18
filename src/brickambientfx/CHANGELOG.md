# BrickAmbientFX (Sarta Edition) - changes

## 1.0.2 - 2026-11-18
- USB cards with the record input switched off (for example a line input that starts [off] at 0% on the Brick):
  when a card is opened, its capture switches are turned on and capture volumes left at 0% are set to 0 dB.
  The full mixer and every change are written to log.txt. Other controls are not touched.
- The input opens in the device's own best format (S24_3LE, S32_LE, S24_LE, then S16_LE) and rate (48000 if it
  has it, otherwise its own, converted in the engine).
- Every USB card remembers its own input (Input 1, 2...).
- log.txt: record devices of every card at start; format, channels, rate, period and buffer at each open; the real
  level of each channel 2 s later (or "silent (all 0)").
- Status shows "USB n IN m".

## 1.0.1 - 2026-11-18
- New setting I/O > USB input: choose the input of the USB card (Input 1, 2...) when an adapter has more than one
  (for example mic in and line in). Only the inputs the card has are listed. Saved with the card; if it is gone at
  the next start, back to Input 1 (written to log.txt). The change fades out and in, no restart.
- STATUS > DEVICE shows "USB n · IN m" (no device names on screen).
- log.txt: card, device opened, format, channels, rate, period and buffer (app side and hardware).

## 1.0.0 - 2026-11-15 (first stable release)
- Input named "USB" everywhere: any class-compliant USB audio device works (a USB synth or an audio interface).
  The real device name is shown on the I/O page, STATUS > CARD.
- Everything from the 0.1 test versions below.

## 0.1.3
- No clicks when you change a type, turn a section on or off, change the cabinet, the delay mode or the routing.
- Delay and reverb switched off ring out on their own (TAIL on the FX page).

## 0.1.2
- Glitch: no more clicks going in and out of an event.

## 0.1.1
- Long reverbs no longer distort; all reverb types at about the same level.

## 0.1
- First version: COMP > AMP + CAB > MOD > DELAY > REVERB, Serial or Split routing, stereo width,
  FREEZE, 16 presets, MIDI learn, MIDI clock, program change.
