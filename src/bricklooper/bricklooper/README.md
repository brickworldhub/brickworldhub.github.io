# BrickLooper

A 5-track loop station for the TrimUI Brick (KNULLI), with the BrickSampler FX on the output. Python, Pygame and NumPy, with the audio engine in C (`native/bsdsp.c`).

Three pages: **LOOP**, **FX**, **SET**.

## Run on desktop

```
pip install pygame numpy
python3 bricklooper.py               # --scale 0.8 for a smaller window, --debug-input to see pad button numbers
```

## LOOP

Five stereo tracks, always in sync.

- The **first take** sets the loop length (from A to A). Under 0.2 s it is ignored.
- **Later takes** line up with the loop and are rounded to whole loops (1x, 2x, 3x...). Stop a little early and the track keeps recording to the end of the loop; stop a little late and the extra bit is folded onto the start.
- **Overdub** adds a layer while the track plays. With **Dub mode = Replace** the new take replaces the old one where you play (the old part is muted while you record). **Undo / redo** removes or brings back the last layer (or the whole take, if there was no overdub yet).
- To redo a track from scratch: **Y + X** clears it, then **A** records a new take.
- **Start / stop**: B starts or stops everything (the box top left shows PLAY or STOP). After a full stop, loops start again from the top.
- Each track has its own volume and level meter. Up to 60 s per track.

| Action | Brick | Keyboard |
|---|---|---|
| Record / overdub / play (current track) | A | Z or Enter |
| Stop / play track | X | Space |
| Start / stop all | B | X or Esc |
| Undo / redo | Y + A | Shift + Z |
| Clear track | Y + X | Shift + Space |
| Clear all (asks first) | Y + B | Shift + X |
| Choose track | Up / Down, or L2 / R2 | Up / Down, `,` `.` |
| Track volume | Left / Right on a track row | Left / Right |

A, X and B always act on the current track (the highlighted number), even when the cursor is on a setting.

Settings on the LOOP page:

- **Source**: Internal mic, or USB: any USB interface, USB mic or synth with USB audio (USB 1, USB 2 when more than one is plugged in). It is opened as soon as it is plugged in.
- **Rec end**: what a new take does when you close it: Play, or go straight into overdub
- **Dub mode**: Add (layers on top) or Replace (records over the old part)
- **Metronome**: Click on/off, BPM (40-240), Time (2/4, 3/4, 4/4, 5/4, 6/8, 7/8) and Count-in. With the click on, the first loop starts on a bar line and is a whole number of bars. Once there is a loop the click follows it (the BPM shown is the loop's tempo); clear all to set a new tempo. With **Count-in** (on by default) the first take starts after one bar of click: press A, count along, play on the next beat 1. A again during the count-in cancels it. The click is not recorded.

On the SET page: Click volume, and for the input Monitor (hear it live), Channel (1, 2, or 1+2 in stereo) and Offset (fine timing: if what you record plays back late, raise it; if early, lower it).

**Built-in mic**: the Brick can't play and record from its mic at the same time. While a take records from the mic the sound stops; the loop keeps time, so the take still lands in the right place. Good for sketching ideas; for playing along, use a USB input.

Loops are saved with the project, as `projects/<NAME>_LOOPS/T1.wav` ... `T5.wav` (32-bit float stereo). Only tracks that changed are written, in the background. They load stopped: press B to start.

## FX

Drive, bitcrush, filter, delay and reverb on everything the Brick plays (loops and input monitor). Recording is dry. Switching a section on or off, or changing the filter type, fades over 10-20 ms (no clicks). Delay and reverb switched off only close their input: echoes and tail ring out. Press B on an FX value to learn a MIDI CC for it.

## SET

Display colour, latency, MIDI CC map, projects (name, load, save as, new) and engine status.

## Files

- `projects/<NAME>.json` – saved on Start/Ctrl+S and on quit, reloaded at launch
- `projects/<NAME>_LOOPS/` – the project's loops
- `settings.json` – app-wide settings (colour, latency, input, offset, CC map)
- `log.txt` – what happened on the last run (`log_previous.txt` for the one before)

## Engine notes

- The looper runs in the C audio thread. The USB input is read by its own C thread (ALSA, `plughw:CARD=<id>`). The USB device and the Brick have separate clocks: the input is resampled continuously (PI control on the ring fill, up to +-12000 ppm), so no frames are ever skipped. log.txt shows the drift in ppm.
- Built-in mic: the writer thread closes the speaker and reads `hw:0,0`; the mic's clock drives the engine until the take is done.
- MIDI: only CCs are used (FX). Notes are ignored.
- Rebuild the library with `native/build.sh` (zig).

## Licence

BrickLooper is free software under the GNU General Public License v3 (see `LICENSE`). Copyright (C) 2026 Alessandro Pietrosanti / BrickWorld. All the source is in this folder: `bricklooper.py` (app), `native/bsdsp.c` (audio engine, `native/build.sh` builds the libraries).

Included third-party parts keep their own licences: NumPy (`vendor_numpy.zip`, BSD), DejaVu fonts (`fonts/LICENSE.txt`).

Inspired by GrainBrick by ETRNLIAM. More free apps for the Brick: [brickworldhub.com](https://brickworldhub.com). TrimUI and Brick are trademarks of their owners. BrickWorld is not affiliated with TrimUI.
