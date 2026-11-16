# BrickTape 1.0.1

A 4-track tape recorder for the TrimUI Brick (KNULLI), in the spirit of the 80s cassette 4-track recorders. Built on BrickLooper 1.4.0: same USB input (with clock drift correction), latency compensation, FX and display. Python, Pygame and NumPy, with the audio engine in C (`native/btdsp.c`).

Four pages: **TAPE**, **MIX**, **FX**, **SET**.

## TAPE

Four mono tracks on one tape, 10 minutes long (the whole project, all in RAM). The screen shows the transport, a cassette whose reels turn with the tape (still in pause and stop, fast in rewind and fast forward), the level of each track, where each track has sound, and the four tracks.

- **SELECT** steps the current track through SAFE, REC (armed: X records on it, its meter shows the input) and MUTE.
- **X** starts recording on the armed tracks. From play it punches in where you are. **A** while recording punches out and keeps playing. Punch in and out crossfade over 5 ms (no clicks).
- **Y** pauses (PLAY or REC). Y again goes on from the same point.
- **B** stops. B again at stop goes back to zero.
- **L1 / R1** rewind / fast forward (30x), and you hear the tracks wind past (SET > Wind sound to switch it off). Press again or B to stop.
- **Y + X** undoes the last take (all the tracks it recorded), and again redoes it.
- **Y + B** clears the current track (asks first).
- **Up / Down** choose the track, **Left / Right** its volume. **L2 / R2** change page.

| Action | Brick | Keyboard |
|---|---|---|
| Play / punch out | A | Z or Enter |
| Record / punch in | X | Space |
| Stop / back to zero | B | X or Esc |
| Pause | Y | Shift |
| Rewind / fast forward | L1 / R1 | [ / ] |
| Safe / rec / mute | SELECT | Tab |
| Undo / redo last take | Y + X | Shift + Space |
| Clear track | Y + B | Shift + X |
| Track / volume | D-pad | arrows |
| Page | L2 / R2 | , / . |

## MIX

For each track: Input (L, R or L+R of the USB input), Volume, Pan, Mute. The USB input is always opened in stereo, so with a two-channel interface two tracks can record different inputs at once.

## FX

Drive, bitcrush, filter, delay and reverb on what the Brick plays. Recording is dry.

## SET

Display colour, latency, click volume; input (source, gain, monitor, offset); metronome (click, BPM, time; the click counts from the start of the tape and is not recorded); projects; MIDI CC map.

## Files

- `projects/<NAME>.json` – mix, metronome and where the tape was left
- `projects/<NAME>_TAPE/T1.wav` ... `T4.wav` – the tracks (16-bit mono), written in the background when you save (START) and on quit
- `settings.json`, `log.txt` as in BrickLooper

## Not in this version yet

EQ, effect send per track, bounce (ping-pong), stereo mixdown export, pitch / varispeed, tape sound, MIDI control of the transport.

Rebuild the library with `native/build.sh` (zig).

## Version history

- **1.0.1** – no more "AUDIO ERROR" message for a few seconds at every start (the sound card opens in the background while the app starts; the message now only shows if a problem lasts).
- **1.0.0** – first stable version. USB audio interface and built-in mic tested on the Brick. SELECT steps SAFE / REC / MUTE, wind sound in rewind and fast forward, 10-minute tape for the whole project with reels that follow it.
- 0.1.x – betas.

## Licence

BrickTape is free software under the GNU GPL v3 (see `LICENSE`). The full source is in this folder: `bricktape.py` (the app) and `native/btdsp.c` (the audio engine; `native/build.sh` builds it). Also at https://github.com/brickworldhub/brickworldhub.github.io in `src/bricktape/`.

Included third-party parts: NumPy (`vendor_numpy.zip`, BSD licence) and the DejaVu fonts (`fonts/`, see `fonts/LICENSE.txt`).

TrimUI and Brick are trademarks of their owners. BrickWorld is not affiliated with TrimUI.
Made by Alessandro Pietrosanti – brickworldhub.com – info@brickworldhub.com
