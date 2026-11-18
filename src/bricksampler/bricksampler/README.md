# BrickSampler 2.0.2

A sampler for the TrimUI Brick running KNULLI. Play it from a USB MIDI keyboard, record from the built-in mic or any USB audio device, shape the sound with FX and build Wavestation-style wave sequences from your own samples.

Free software, GNU GPL v3 (see `LICENSE`). Made by Alessandro Pietrosanti – https://brickworldhub.com

## Install

Copy `BrickSampler.sh`, the `.keys` files and the `bricksampler` folder to `roms/ports` on the SD card, then start BrickSampler from Ports. The first start unpacks NumPy (a few seconds). Put your WAV files anywhere under `bricksampler/samples/` (8/16/24/32-bit or float, any sample rate, mono or stereo).

Updating from an older version: keep your `samples/`, `projects/` and `settings.json`.

## Modes

- **Chromatic** – one sample played across the keyboard, 8 voices, sustain pedal and pitch bend (±2 st).
- **Kit** – 16 pads on MIDI notes 36–51 (GM drum layout: kick 36, snare 38, closed hat 42). Pads are one-shot by default and cut themselves when played again. Playing a pad selects it for editing.

Each mode keeps its own samples, so you can switch freely (SELECT).

## Pages

- **SAMPLE** – everything about the sample on one page: file, root note, tune, fine, gain, pan, reverse, gate/one-shot; region start/end, loop on/off, loop start/end and crossfade, with the loop seam drawn under them. TOOLS (press A): chop onto pads, time-stretch, zero snap the loop, SAVE REGION AS NEW (writes only the part between start and end to a new WAV, named on the keyboard, and uses it) and OVERWRITE ORIGINAL (asks first; other pads or steps that play the same file are reloaded and reset). When the slot's wave sequence is on, this page shows and edits the selected step's sample (region start/end, loop); L2/R2 move between steps.
- **WSEQ** – wave sequence, after the Korg Wavestation. Switch it on (WAVE SEQ = On) and the slot or pad plays a list of steps instead of one sample. Each step: sample, duration (ms, or a note value when TEMPO is on, or GATE = until you let go of the key), crossfade into the next step (Cut = hard switch), level, tune, fine, loop. The sequence: start step, loop Off / Forward / Back-fwd between two steps, repeats (or endless), tempo, key sync (each note starts its own sequence) or free-running, and MOD: velocity moves the start step, or the mod wheel scans through the steps by hand. L2/R2 move between steps; A on a step opens the step menu (add, duplicate, move, delete); A on SAMPLE picks the step's sample.
- **AMP** – attack, decay, sustain, release, velocity sensitivity.
- **LFO** – one LFO for the whole project: wave (sine, triangle, saw, square, S&H), rate, delay (fade-in), mod wheel = depth; depth to pitch, volume, filter cutoff. Restarts on the first note after silence.
- **FX** – drive, bitcrush, downsample, filter (LP/HP/BP with resonance), stereo delay (tone, ping-pong), reverb, master. A switches a section on/off. Switching makes no click: drive and filter fade over 10 ms, delay and reverb only close their input so the echoes and the tail ring out.
- **REC** – records into the current slot or pad; takes are saved to `samples/recordings/`. SOURCE = Internal mic or USB (any USB audio device KNULLI sees: audio interface, USB mic, synth with USB audio; USB 1 / USB 2 when there are several); USB INPUT = which input of the USB device (Input 1, Input 2...: some adapters have more than one, e.g. mic in and line in; fixed on Input 1 when there is only one); CHANNEL = left, right or both. With a USB source the speaker keeps playing while you record; with the internal mic the sound pauses while recording (the Brick can't do both).
- **SET** – record rate/bits, display colour, latency (8/16/32 ms), MIDI CC map, projects, engine status.

## Controls

| Action | Brick | Keyboard (desktop) |
|---|---|---|
| Select parameter | D-pad up / down | Up / Down |
| Change value | D-pad left / right | Left / Right |
| Fine adjust | hold Y | hold Shift |
| Open / confirm / record | A | Z or Enter |
| Back / MIDI learn menu | B | X or Esc |
| Play current slot | X | Space |
| Previous / next page | L1 / R1 | `[` `]` |
| Previous / next pad or step | L2 / R2 | `,` `.` |
| Chromatic / kit | Select | Tab |
| Help screen | hold Select | – |
| Save project | Start | Ctrl+S |
| Quit | Select + Start (or Menu + Start) | Ctrl+Q |
| Output volume | volume rocker | – |
| Piano | – | A W S E D F T G Y H U J K O L P, octave `-` `=` |

## MIDI

Any class-compliant USB MIDI device, all channels.

- Notes, velocity, pitch bend, sustain (CC64)
- CC1 mod wheel (LFO depth, or wave sequence scan), CC120/123 all notes off
- Any CC can move any FX or LFO value, or switch an FX section: press **B** on the value, choose LEARN, move a knob. Defaults: CC74 cutoff, CC71 resonance, CC7 master. The map is in SET > MIDI CC (A removes an entry) and is kept for all projects.

On the Brick it reads the raw ALSA MIDI devices directly, nothing extra to install. Devices are picked up when plugged in. The MIDI dot in the header lights on incoming notes.

## Files

- `projects/NAME.json` – projects (SET > PROJECT: rename, load, save as, new). Saved on Start and on quit.
- `settings.json` – app-wide settings: colour, latency, MIDI CC map, record source, volume.
- `samples/recordings/` – your takes.
- `log.txt` – what happened at the last start (the one before is in `log_previous.txt`). Look here if something goes wrong.

## Source

- `bricksampler.py` – the app (Python, Pygame, NumPy).
- `native/bsdsp.c` – the audio engine: voices, envelopes, loops, LFO, wave sequences, FX and the real-time ALSA output thread. Python only sends notes and settings, so drawing the screen can't delay the sound.
- `native/build.sh` – builds `libbsdsp_aarch64.so` (Brick) and `libbsdsp_x86_64.so` (Linux desktop) with zig (`pip install ziglang`). Without the library the app falls back to a slower engine in Python.

Run on a desktop for testing:

```
pip install pygame numpy
python3 bricksampler.py        # --scale 0.8 for a smaller window, --debug-input to see button numbers
```

## Thanks

Inspired by GrainBrick by ETRNLIAM, which showed what the Brick can do with Python, Pygame and NumPy.

TrimUI and Brick are trademarks of their owners. BrickWorld is not affiliated with TrimUI.
