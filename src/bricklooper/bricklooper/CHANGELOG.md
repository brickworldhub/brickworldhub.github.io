# BrickLooper changelog

## 1.5.2 (18 Nov 2026)
- USB line inputs that gave only silence now record: when a USB device opens, its mixer is written in log.txt, any capture switch that is off is switched on, and a capture volume left at 0% is set to a neutral 0 dB. Nothing else is touched. (A computer does this by itself; on the Brick nothing did.)
- The USB input opens in the device's own best format (24-bit when it has it) and at a rate it really has (48 kHz if it has it). A device without 48 kHz is converted in the app.
- log.txt: the recording devices of every USB device, and 2 seconds after opening the real level of each channel (dBFS, or "silent (all 0)").
- Each USB device keeps its own input when you switch between devices. Built-in mic: USB INPUT shows Input 1.
- SET page: the input shows as USB n IN m. The input fades over 20 ms when it changes.

## 1.5.1 (18 Nov 2026)
- New USB input setting on the LOOP page, under Source: choose Input 1, Input 2... when a USB device has more than one recording input (for example an adapter with a mic in and a line in). Only the inputs the device really has are shown. Saved for each device; if the input is gone next time, BrickLooper goes back to Input 1 and says so in log.txt.
- Changing source, USB input or channel fades the input out and in (10 ms), no click.
- log.txt: card, device, formats, channels, rates, and what the card really runs at (format, channels, rate, period, buffer).
- SET page, ENGINE: the input shows as USB n · IN m.

## 1.5.0 (15 Nov 2026) – first stable release
- Tested on the TrimUI Brick (KNULLI) with a USB synth and a USB audio interface.
- No device names on screen: the input is "Internal mic" or "USB" (USB 1 / USB 2 with two devices).
- GPL-3.0 licence file and notices in the source.

## 1.4.0 (15 Nov 2026)
- No more click every few seconds with a USB input: the input follows the USB device's clock (continuous resampling) instead of skipping frames.
- FX switch on and off without clicks; delay and reverb switched off let the echoes and tail ring out.
- X and Y buttons in the right place.

## 1.3.0 (11 Nov 2026)
- Count-in: one bar of click before the first take.

## 1.2.0 (11 Nov 2026)
- Metronome (click on/off, BPM, time signature, click volume); first loop in whole bars.
- Dub mode Add / Replace.

## 1.1.0 (11 Nov 2026)
- Looper only: pages LOOP, FX, SET. Input from USB or the built-in mic. FX on the output.

## 1.0.0 (10 Nov 2026)
- 5 stereo tracks in sync, overdub, undo / redo, start / stop all, loops saved with the project.
