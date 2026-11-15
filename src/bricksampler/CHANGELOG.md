# BrickSampler changes

## 2.0.0 (2026-11-15)
- Wave sequencing (WSEQ page), after the Korg Wavestation: up to 64 steps per slot or pad, each with its own sample, region, duration or GATE, crossfade, level, tune and loop; sequence loop forward or back-and-forth, repeats, tempo, key sync or free-running, velocity or mod wheel control.
- SAMPLE and EDIT are now one SAMPLE page, with a TOOLS menu. With the wave sequence on, it edits the selected step's sample.
- SAVE REGION AS NEW and OVERWRITE ORIGINAL: keep only the part between start and end.
- No clicks when you switch FX sections on or off; delay echoes and reverb tail ring out.
- USB recording sources are shown as "USB", whatever the device.

## 1.4.x
- USB audio input (audio interface, USB mic, synth with USB audio), channel 1 / 2 / both.
- Volume rocker sets the output volume. X and Y buttons fixed.
- Loop points with crossfade, CHOP onto pads, TIME-STRETCH, LFO page, MIDI learn.

## 1.2.x
- Audio engine in C with its own real-time thread: low MIDI latency (8 / 16 / 32 ms).
- Named projects, sample rename/delete, takes named before saving, FX on/off, display colours.
