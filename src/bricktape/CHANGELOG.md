# BrickTape changelog

- **1.1.3** (19 Nov 2026) – Built-in mic records in full-duplex: the mic opens first, then the output, and the output stays on, so you hear the other tracks while you overdub. SET > Mic monitor (Off / On, default Off; On warns to use headphones). Falls back to the old way (output off while recording) if full-duplex can't start, with the reason in log.txt. Log: mic device, channels, latency, input and output period/buffer, full-duplex yes/no. Fades, no clicks.
- **1.1.2** (18 Nov 2026) – USB inputs that sent silence work now: capture switches on and capture volumes off 0% on the card's mixer (once per card, logged), the device's best format (S24_3LE > S32_LE > S24_LE > S16_LE) and a rate it supports (48 kHz or its first rate, converted in the app). Log: recording inputs of each card, what is open, channel levels after 2 s. About 20 ms fade when the input changes.
- **1.1.1** (18 Nov 2026) – SET > USB input: choose Input 1, 2... on adapters with more than one recording input. Saved per card, falls back to Input 1. No restart needed. TAPE screen shows USB n IN m.
- **1.1.0** (16 Nov 2026) – L1 / R1 change page everywhere; hold L2 / R2 to rewind / fast forward, it stops where you let go.
- **1.0.1** (16 Nov 2026) – No more "AUDIO ERROR" flash at every start.
- **1.0.0** (15 Nov 2026) – First stable version.
- 0.1.x (15 Nov 2026) – Betas.
