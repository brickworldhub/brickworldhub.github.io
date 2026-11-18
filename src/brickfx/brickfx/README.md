# BrickFX 1.0.2

A stereo multi-effect for the TrimUI Brick (KNULLI): five effects in a chain, **DRIVE > MOD > DELAY > REVERB > EQ**, one type at a time in each, like the Hologram Chroma Console. Same look and controls as BrickSampler, but a separate app.

```
USB audio device (synth, audio interface...) --USB--> Brick --headphone jack--> audio interface
```

## Audio path

All audio runs in C (`native/bfx.c`), in two real-time threads. Python never touches the sound, so drawing the screen can't cause a dropout.

- **Capture thread** reads the USB card (`plughw:CARD=<id>,DEV=<n>`, the input chosen on the I/O page, stereo; mono devices are doubled to both sides), in the device's own best format (S24_3LE > S32_LE > S24_LE > S16_LE, decoded in C; S16_LE if refused) and at a rate it supports (48 kHz, otherwise its first rate, converted to 48 kHz). Before opening, the card's mixer is checked once: capture switches that are off are switched on and capture levels at 0% go to 0 dB (on the Brick nothing else does it, and some line inputs then send only zeros). Changing input fades the input out and back in (20 ms), so there's no click, and delay and reverb tails carry on.
- **Writer thread** plays on the Brick's own codec (`plughw:0,0`), 128-frame blocks.
- **Drift resampler** between the two. The USB device and the Brick each have their own clock, and they are never exactly the same (typically 50-300 ppm apart). Without correction the buffer between them slowly empties or overflows, and you get a click every few minutes. BrickFX reads the input with a tiny fractional speed change (Hermite interpolation) steered by a PI controller that holds the buffer on target. Tested offline with ±300 ppm of drift and jittery USB packets: no dropouts in 2 minutes of simulated time, spurious tones below -100 dB.

Latency, input to output: about 6 ms of input buffer plus the output buffer (Low 8 ms, Mid 16 ms, Safe 32 ms), plus a few ms in the converters. The I/O page shows the live figure.

## The chain

Each section has a **Type**; Off skips it. Switching a section on or off, or changing its type (or the delay mode), crossfades: the old effect fades to the dry signal and the new one fades in (10-80 ms), so there are no clicks. Switching DELAY or REVERB **off** only closes its input: the echoes and the tail ring out on their own while the dry signal comes back at full level (CHAIN shows TAIL); switch it on again during the tail and it carries on from there. X bypasses the whole chain (dry signal). With **Trails** on (I/O page) the delay and reverb ring out after bypass.

- **DRIVE**: Drive (warm, slightly asymmetric tube-style), Fuzz (hard clip, tight lows), Crush (bit depth 16 to 4 bit and sample rate; Tone = rate). Amount, Tone, Level. Loudness stays about the same as Amount goes up.
- **MOD**: Chorus, Phaser (6 stages), Tremolo, Auto-pan, Vibrato. Rate, Depth, Mix. Left and right LFOs are 90° apart (auto-pan 180°), so chorus and phaser are wide.
- **DELAY**: Digital, Tape (flutter, softer top, more saturation in the repeats), Analog (dark BBD-style repeats, warm clip).
  - Time 20 ms to 2 s, or Sync to a note value at the Tempo, a tapped tempo or MIDI clock. Time changes glide like tape.
  - Feedback 0-100% (soft-limited), Mix (at 50% dry and echoes are both full), Mode Stereo / Ping-pong / Mono, Width, Hi cut / Lo cut in the feedback loop, Mod depth / rate (wow).
- **REVERB**: Room, Plate, Hall (Dattorro plate topology, true stereo out, modulated tank). Decay, Pre-delay (0-250 ms), Damping, Mod, Mix. The picture shows the pre-delay gap and the decay. The level into the reverb follows type and decay, so long tails stay about 2 dB under the dry signal (at mix 100%) and never clip.
- **EQ**: 3 bands at the end of the chain: Low shelf (30-500 Hz), Mid peak (150 Hz-8 kHz, Q 0.3-8), High shelf (1.5-16 kHz), each +-15 dB. The picture shows the response curve.

## Oscilloscope

Tap SELECT (keyboard O, or a short Tab) for a full-screen scope; tap again to go back. X, Y and the presets still work.

- A: input or output. B: L / R lanes, overlay, or XY (mid up, side across: a goniometer for stereo width).
- Left / right: time across the screen (1.3 ms to 85 ms). Up / down: gain x1 to x16.
- Schmitt trigger at the zero crossing, between samples, and it keeps the crossing that matches the last picture, so a held note stands still (also FM sounds with several crossings per cycle).

## Pages

- **I/O** – Source USB (the first USB audio device found: a synth, an audio interface...; left/right picks a specific one), USB input (for cards with more than one recording input, e.g. an adapter with a mic in and a line in: Input 1, Input 2...; fixed on Input 1 when there's only one), channels, input gain, output level, trails, latency; live status: card, drift in ppm, latency, dropouts, CPU.
- **CHAIN** – the four sections at a glance (lit = on) and their types, plus the tempo.
- **DRV / MOD / DLY / REV / EQ** – one page per section: a picture (drive curve, LFOs, echoes, reverb decay) and its settings.
- **SET** – preset, MIDI clock follow, CC map, display colour, engine info.

## Controls

| Action | Brick | Keyboard |
|---|---|---|
| Select | D-pad up / down | Up / Down |
| Change value | D-pad left / right | Left / Right |
| Fine adjust | hold Y | hold Shift |
| Tap tempo | tap Y | T |
| On / bypass | X | Space |
| Step a setting (sync, mode...) | A | Z / Enter |
| MIDI learn | B on a delay value | X / Esc |
| Page | L1 / R1 | `[` `]` |
| Preset | L2 / R2 | `,` `.` |
| Save | Start | Ctrl+S |
| Quit | Select + Start | Ctrl+Q |
| Oscilloscope | tap Select | O |
| Output volume | volume rocker | – |

Tap tempo: free time takes the interval between your taps; a synced delay takes the tapped tempo.

## Presets

Eight presets, saved in `presets.json`. Edits save themselves after 2 seconds; Start saves at once. SET > Reset puts the current preset back to factory.

Factory: 1 ping-pong + plate, 2 chorus + tape slap + room, 3 dotted 1/8 + hall, 4 drive + dark analog echo, 5 ambient (chorus, long delay, big hall), 6 phaser + plate, 7 lo-fi (crush, tremolo, tape), 8 wash (fuzz, runaway ping-pong, hall).

Presets from v0.1 (delay only) are replaced by the new factory set on the first start.

## MIDI

Any class-compliant USB MIDI device (a synth's MIDI usually comes over the same USB cable as its audio).

- **CC learn**: B on any FX value (types too), then move a knob. A `*` next to a value means a CC moves it. SET > CC map lists them, Clear map removes them all.
- **MIDI clock**: with SET > MIDI clock on Follow and Sync not Off, the delay follows the clock tempo. The header shows CLK.
- **Program change** 0-7 picks preset 1-8.

## Desktop test

```
pip install pygame numpy
python3 brickfx.py
```

On a Linux PC a built-in test phrase plays through the delay (no USB input needed). `controls.json` `"audio_backend": "alsa"` uses real ALSA devices on a PC as well.

## Files

- `settings.json` – input (card, USB input, and the USB input of each card), output, display, CC map (app-wide)
- `presets.json` – the eight presets
- `log.txt` – what happened at the last start (`log_previous.txt` the one before): the cards' inputs, the mixer, the opened format/rate and the input level after 2 s
- `native/build.sh` – rebuilds the C engine with zig

## Licence

BrickFX is free software under the GNU General Public License version 3 (see `LICENSE`). The full source is in this folder: `brickfx.py` (screen, buttons, MIDI, presets) and `native/bfx.c` (the audio engine); `native/build.sh` rebuilds the engine. Source-only downloads and updates: https://brickworldhub.com

`vendor_numpy.zip` is NumPy 1.26.4 for the Brick (BSD licence, its licence file is inside).

Made by Alessandro Pietrosanti · info@brickworldhub.com · Instagram @brickworld_hub

TrimUI and Brick are trademarks of their owners. BrickWorld is not affiliated with TrimUI.
