# BrickAmbientFX v1.0.2 - Sarta Edition

An ambient stereo multi-effect for the TrimUI Brick (KNULLI). It is built on BrickFX 0.2.2 and takes its idea
from the Zoom Neurig: compressor, amp and cabinet, modulation, delay and reverb, serial or split routing, stereo out
and presets. For ambient music the delay, reverb and modulation go much further than in BrickFX.

    USB audio device (synth, audio interface...) --USB--> Brick --headphone jack--> audio interface

Same screen, same buttons and same audio engine (USB input, drift resampler, low latency output) as BrickFX.

## The chain

    COMP > AMP + CAB > MOD > DELAY > REVERB                 Routing: Serial
    COMP > AMP + CAB > MOD -+-> DELAY  -+-> out              Routing: Split
                            +-> REVERB -+

Every section has a Type. Off means it does nothing and costs no CPU.

| Section | Types | Settings |
|---|---|---|
| COMP | Studio, Opto, Limiter, Swell | Threshold, Ratio, Attack, Release, Makeup, Mix (parallel compression). Swell: each new note fades in over the Swell time, Reset = how long the signal must stay quiet before the next swell |
| AMP | Clean, Crunch, Lead, Fuzz, Lo-fi | Gain (Lo-fi: Crush), Bass, Middle, Treble (+-12 dB), Cabinet (Off, 1x12, 2x12, 4x12, Radio), Level |
| MOD | Chorus, Ensemble, Flanger, Phaser, Tremolo, Auto-pan, Vibrato, Glitch | Rate or Sync (note value at the tempo), Depth, Color, Mix |
| DELAY | Digital, Tape, Analog, Dual, Reverse, Shimmer | Time or Sync, Feedback, Mix, Mode (Stereo, Ping-pong, Mono), Width, Hi cut, Lo cut, Mod depth and rate, Color. Up to 4 s |
| REVERB | Room, Plate, Hall, Cloud, Shimmer, Bloom | Decay, Pre-delay, Mix, Damping, Lo cut, Mod, Color |

**Color** changes with the type:

- MOD: Chorus and Ensemble = stereo Spread, Flanger and Phaser = Feedback, Tremolo and Auto-pan = Shape (sine to square),
  Vibrato = Drift (random warble, like worn tape), Glitch = Variety.
- DELAY: Digital and Reverse = Diffuse (each repeat gets softer, into a wash), Tape and Analog = Age (more wow, flutter,
  saturation, darker), Dual = R ratio (right side time: 7/8, 3/4, 2/3, 1/2, 3/8, 1/3, 1/4), Shimmer = Octave (how much
  of each repeat goes an octave up).
- REVERB: Room, Plate, Hall and Cloud = Diffuse, Shimmer = Shimmer (octave up inside the tail), Bloom = Bloom (how slowly
  the reverb builds).

**Glitch** cuts the sound into slices (Rate = slices per second, or a note value with Sync). At each slice a dice roll
(Chance) can replace it with the slice before: repeated, reversed, an octave up, slowed to a stop or dropped. Variety
opens up more kinds of event. The MOD page shows each event as it happens.

## Switching without clicks

Changing a type, turning a section on or off, changing the cabinet, the delay mode or the routing: the section fades
out, changes over while silent and fades back in (COMP and AMP 10 ms, MOD 25 ms, DELAY 30 ms, REVERB 80 ms, routing
30 ms). Turning DELAY or REVERB off only closes its input: the echoes and the tail ring out on their own (TAIL on the
FX page) and the dry sound is back at full level at once. Turn it on again during the tail and it carries on.

## USB input (I/O page)

- **Source**: Auto takes the first USB audio device; with two devices choose USB 1 or USB 2.
- **USB input**: some USB adapters have more than one input on the same card (for example a mic input and a line
  input). Choose Input 1, 2, 3... (only the inputs the card really has are listed; with one input it stays on 1).
  The change fades out and back in, no restart needed. Every card remembers its own input; if the saved one is
  gone, the app takes the first and says so in log.txt.
- When a card is opened, the app switches on its record controls if they are off (some cards start with the line
  input off and at 0% on the Brick, so they send only silence). Other mixer settings are not touched.
- The input opens in the device's own best format (24 or 32 bit if it has it) and rate (48 kHz if it has it,
  otherwise its own rate, converted in the engine).
- STATUS > DEVICE shows which one is open, as "USB n IN m". log.txt has the full details and, 2 s after opening,
  the real input level of each channel.

## FX page (routing)

- **Routing**: Serial (delay into reverb) or Split (delay and reverb side by side, both fed by the same signal).
- **Balance** (Split): more delay or more reverb.
- **Spread** (Split): delay to the left, reverb to the right. At 100% they are fully apart.
- **Width**: stereo width of the whole output, 0% mono, 100% as is, 200% extra wide.
- **Freeze**: delay and reverb hold what they have, for ever, and new sound goes over them dry.
- **Tempo**: for synced delay and modulation (or follows MIDI clock).

## Controls

| Button | Action |
|---|---|
| Up / Down | choose a setting |
| Left / Right | change it (hold Y for fine steps) |
| A | step through a setting with options |
| X | on / bypass (with Trails the tails keep ringing) |
| tap SELECT | FREEZE on / off |
| hold SELECT | show the controls |
| tap Y | tap tempo |
| B | MIDI learn on the selected setting |
| L1 / R1 | page |
| L2 / R2 | preset |
| START | save now |
| SELECT + START | quit |
| volume buttons | output volume |

Keyboard (desktop test): arrows, Z/Enter = A, X/Esc = B, Space = bypass, F = freeze, T = tap, [ ] = page, , . = preset,
Tab = help, Ctrl+S save, Ctrl+Q quit.

## Presets

16 presets, they save themselves (2 s after a change). Reset on the SET page brings one back to factory.
MIDI program change 0-15 picks the preset. A MIDI CC can move any setting, plus On/bypass, Tap and Freeze.

1 Sarta Cloud, 2 Shimmer Sky, 3 Tape Dreams, 4 Swell Hall, 5 Reverse Rain, 6 Bloom Pad, 7 Glitch Garden,
8 Dual Echoes, 9 Slow Jet, 10 Tremolo Waves, 11 Lo-fi Radio, 12 Crunch Space, 13 Octave Climb, 14 Phase Drift,
15 Split World, 16 Clean Room.

## Engine

Native C (`native/bafx.c`), 48 kHz, 128-frame blocks, its own real-time threads. Python (`brickambientfx.py`) only draws
the screen and handles buttons, MIDI and presets. Build the libraries with `native/build.sh` (zig).

On a PC (Linux) the app plays a built-in test sound through the chain: `python3 brickambientfx.py`.

Licence: GPL-3.0-or-later (see LICENSE). Made by Alessandro Pietrosanti, part of BrickWorld (brickworldhub.com).
Bundled NumPy (vendor_numpy.zip) is under its own BSD licence (inside the zip).

TrimUI and Brick are trademarks of their owners. BrickWorld is not affiliated with TrimUI.
