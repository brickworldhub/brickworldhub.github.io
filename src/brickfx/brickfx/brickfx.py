#!/usr/bin/env python3
"""
BrickFX - a stereo multi-effect for the TrimUI Brick (KNULLI). First effect: delay.

  USB audio device (synth, audio interface...) --USB--> Brick --headphone jack--> audio interface

The audio path is native C (native/bfx.c): a capture thread reads the USB card, a writer thread plays on
the Brick's jack, and an adaptive resampler between them absorbs the clock difference of the two cards.
This file is only the screen, buttons, MIDI and presets. Same look and controls as BrickSampler.

  pip install pygame numpy
  python3 brickfx.py            # desktop test: a built-in test sound plays through the delay

Copyright (C) 2026 Alessandro Pietrosanti - BrickWorld (https://brickworldhub.com)
This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License version 3 (see LICENSE). It comes with ABSOLUTELY NO WARRANTY.
"""
import os, sys, time, json, glob, math, argparse, select, threading, re, faulthandler

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _fatal(msg):
    """Show a startup error on screen (instead of a black screen), log it, and quit."""
    print("FATAL:", msg, flush=True)
    try:
        import pygame as pg
        pg.display.init()
        pg.font.init()
        scr = pg.display.set_mode((1024, 768), pg.FULLSCREEN if os.path.isdir("/userdata/roms") else 0)
        scr.fill((30, 10, 2))
        f = pg.font.Font(None, 40)
        for i, line in enumerate(["BrickFX can't start", ""] + [msg[j:j + 60] for j in range(0, len(msg), 60)]
                                 + ["", "Details in brickfx/log.txt. Closing in 10 s."]):
            scr.blit(f.render(line, True, (255, 138, 36)), (60, 120 + i * 44))
        pg.display.flip()
        time.sleep(10)
    except Exception:
        pass
    os._exit(1)


try:
    import numpy as np
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    try:
        import numpy as np
    except Exception as e:
        _fatal(f"NumPy could not be loaded: {e}")

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

APP_NAME = "BRICKFX"
VERSION = "1.0.0"
SR = 48000
BLOCK = 128                     # native processing block: 2.7 ms
SCREEN_W, SCREEN_H = 1024, 768
SWAP_AB = False
ON_DEVICE = os.path.isdir("/userdata/roms")
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "controls.json")
SETTINGS_FILE = os.path.join(HERE, "settings.json")
PRESETS_FILE = os.path.join(HERE, "presets.json")
NUM_PRESETS = 8
ASOUND_CARDS = "/proc/asound/cards"


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ----------------------------------------------------------------------------- parameters

def f_pct(v): return f"{v * 100:.0f}%"
def f_time(v): return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"
def f_gain(v): return "-inf dB" if v <= 0.0001 else f"{20 * math.log10(v):+.1f} dB"
def f_bpm(v): return f"{v:.0f} BPM"
def f_hicut(v): return "Off" if v >= 19000 else (f"{v:.0f} Hz" if v < 1000 else f"{v / 1000:.1f} kHz")
def f_locut(v): return "Off" if v <= 21 else f"{v:.0f} Hz"
def f_rate(v): return f"{v:.2f} Hz"


class Param:
    def __init__(self, key, label, kind, lo=0.0, hi=1.0, step=0.01, fmt=f_pct, opts=None, scope="fx"):
        self.key, self.label, self.kind, self.lo, self.hi = key, label, kind, lo, hi
        self.step, self.fmt, self.opts, self.scope = step, fmt, opts, scope


OFFON = ["Off", "On"]
# note values: (label, length in quarter notes)
SYNC = [("Off", 0), ("1/1", 4), ("1/2", 2), ("1/2.", 3), ("1/2T", 4 / 3), ("1/4", 1), ("1/4.", 1.5),
        ("1/4T", 2 / 3), ("1/8", 0.5), ("1/8.", 0.75), ("1/8T", 1 / 3), ("1/16", 0.25), ("1/16.", 0.375),
        ("1/16T", 1 / 6)]
MODES = ["Stereo", "Ping-pong", "Mono"]
MAX_TIME = 2.5

P_TIME = Param("time", "Time", "exp", 0.02, 2.0, 0.03, f_time)
P_TEMPO = Param("tempo", "Tempo", "lin", 40, 300, 1, f_bpm)
P_SYNC = Param("sync", "Sync", "enum", opts=[s for s, _ in SYNC])
P_FB = Param("fb", "Feedback", "lin", 0, 1, 0.02)
P_MIX = Param("mix", "Mix", "lin", 0, 1, 0.02)
P_MODE = Param("mode", "Mode", "enum", opts=MODES)
P_WIDTH = Param("width", "Width", "lin", 0, 1, 0.05)
P_HICUT = Param("hicut", "Hi cut", "exp", 1000, 20000, 0.05, f_hicut)
P_LOCUT = Param("locut", "Lo cut", "exp", 20, 1000, 0.06, f_locut)
P_MODD = Param("moddepth", "Mod depth", "lin", 0, 1, 0.05)
P_MODR = Param("modrate", "Mod rate", "exp", 0.1, 5.0, 0.06, f_rate)
DLY_TYPES = ["Off", "Digital", "Tape", "Analog"]
P_DLY_TYPE = Param("dly_type", "Type", "enum", opts=DLY_TYPES)
DELAY_PARAMS = [P_DLY_TYPE, P_TIME, P_TEMPO, P_SYNC, P_FB, P_MIX, P_MODE, P_WIDTH, P_HICUT, P_LOCUT, P_MODD, P_MODR]

DRV_TYPES = ["Off", "Drive", "Fuzz", "Crush"]
P_DRV_TYPE = Param("drv_type", "Type", "enum", opts=DRV_TYPES)
P_DRV_AMT = Param("drv_amt", "Amount", "lin", 0, 1, 0.02)
P_DRV_TONE = Param("drv_tone", "Tone", "lin", 0, 1, 0.02)
P_DRV_LEVEL = Param("drv_level", "Level", "lin", 0, 1.5, 0.02, f_gain)
DRIVE_PARAMS = [P_DRV_TYPE, P_DRV_AMT, P_DRV_TONE, P_DRV_LEVEL]

MOD_TYPES = ["Off", "Chorus", "Phaser", "Tremolo", "Auto-pan", "Vibrato"]
P_MOD_TYPE = Param("mod_type", "Type", "enum", opts=MOD_TYPES)
P_MOD_RATE = Param("mod_rate", "Rate", "exp", 0.05, 12.0, 0.05, f_rate)
P_MOD_DEPTH = Param("mod_depth", "Depth", "lin", 0, 1, 0.02)
P_MOD_MIX = Param("mod_mix", "Mix", "lin", 0, 1, 0.02)
MOD_PARAMS = [P_MOD_TYPE, P_MOD_RATE, P_MOD_DEPTH, P_MOD_MIX]

REV_TYPES = ["Off", "Room", "Plate", "Hall"]
REV_SIZE = [1.0, 0.55, 1.0, 1.7]
P_REV_TYPE = Param("rev_type", "Type", "enum", opts=REV_TYPES)
P_REV_DECAY = Param("rev_decay", "Decay", "lin", 0, 1, 0.02)
P_REV_PRE = Param("rev_pre", "Pre-delay", "lin", 0, 0.25, 0.005, f_time)
P_REV_DAMP = Param("rev_damp", "Damping", "lin", 0, 1, 0.02)
P_REV_MOD = Param("rev_mod", "Mod", "lin", 0, 1, 0.05)
P_REV_MIX = Param("rev_mix", "Mix", "lin", 0, 1, 0.02)
REVERB_PARAMS = [P_REV_TYPE, P_REV_DECAY, P_REV_PRE, P_REV_DAMP, P_REV_MOD, P_REV_MIX]

def f_db(v): return f"{v:+.1f} dB"
def f_hz(v): return f"{v:.0f} Hz" if v < 1000 else f"{v / 1000:.1f} kHz"
def f_q(v): return f"{v:.2f}"


EQ_TYPES = ["Off", "On"]
P_EQ_TYPE = Param("eq_type", "Type", "enum", opts=EQ_TYPES)
P_EQ_LOW = Param("eq_low", "Low", "lin", -15, 15, 0.5, f_db)
P_EQ_LOWF = Param("eq_lowf", "Low freq", "exp", 30, 500, 0.05, f_hz)
P_EQ_MID = Param("eq_mid", "Mid", "lin", -15, 15, 0.5, f_db)
P_EQ_MIDF = Param("eq_midf", "Mid freq", "exp", 150, 8000, 0.05, f_hz)
P_EQ_Q = Param("eq_q", "Mid Q", "exp", 0.3, 8.0, 0.08, f_q)
P_EQ_HIGH = Param("eq_high", "High", "lin", -15, 15, 0.5, f_db)
P_EQ_HIGHF = Param("eq_highf", "High freq", "exp", 1500, 16000, 0.05, f_hz)
EQ_PARAMS = [P_EQ_TYPE, P_EQ_LOW, P_EQ_LOWF, P_EQ_MID, P_EQ_MIDF, P_EQ_Q, P_EQ_HIGH, P_EQ_HIGHF]

# the sections, in signal order: (name, type key, type names)
SECTIONS = [("DRIVE", "drv_type", DRV_TYPES), ("MOD", "mod_type", MOD_TYPES), ("DELAY", "dly_type", DLY_TYPES),
            ("REVERB", "rev_type", REV_TYPES), ("EQ", "eq_type", EQ_TYPES)]
CHAIN_PARAMS = [Param(k, n.capitalize(), "enum", opts=o) for n, k, o in SECTIONS]

PRESETS_VERSION = 2
PRESET_DEFAULTS = dict(on=1, tempo=120.0,
                       drv_type=0, drv_amt=0.3, drv_tone=0.6, drv_level=1.0,
                       mod_type=0, mod_rate=0.8, mod_depth=0.5, mod_mix=0.5,
                       dly_type=1, time=0.375, sync=0, fb=0.4, mix=0.3, mode=1, width=1.0,
                       hicut=7000.0, locut=100.0, moddepth=0.15, modrate=0.6,
                       rev_type=0, rev_decay=0.5, rev_pre=0.02, rev_damp=0.4, rev_mod=0.3, rev_mix=0.25,
                       eq_type=0, eq_low=0.0, eq_lowf=120.0, eq_mid=0.0, eq_midf=1000.0, eq_q=0.8, eq_high=0.0,
                       eq_highf=6000.0)


def _pr(**kw):
    return dict(PRESET_DEFAULTS, **kw)


FACTORY = [
    # 1 ping-pong delay into a plate
    _pr(rev_type=2, rev_decay=0.55, rev_mix=0.2),
    # 2 chorus, tape slapback, small room
    _pr(mod_type=1, mod_rate=0.6, mod_depth=0.5, mod_mix=0.5, dly_type=2, time=0.12, fb=0.15, mix=0.3, mode=0,
        hicut=5000.0, rev_type=1, rev_decay=0.4, rev_mix=0.2),
    # 3 dotted 1/8 ping-pong, hall
    _pr(sync=9, fb=0.45, mix=0.35, rev_type=3, rev_decay=0.55, rev_mix=0.25),
    # 4 dark analog echo with drive
    _pr(drv_type=1, drv_amt=0.35, drv_tone=0.45, dly_type=3, time=0.45, fb=0.6, mix=0.4, mode=0, hicut=6000.0,
        locut=150.0, moddepth=0.35, modrate=0.5),
    # 5 ambient: long hall + wide delay, slow chorus
    _pr(mod_type=1, mod_rate=0.25, mod_depth=0.7, mod_mix=0.4, time=0.7, fb=0.55, mix=0.35, hicut=5000.0,
        rev_type=3, rev_decay=0.85, rev_pre=0.04, rev_damp=0.5, rev_mod=0.5, rev_mix=0.45),
    # 6 phaser + plate
    _pr(mod_type=2, mod_rate=0.35, mod_depth=0.8, mod_mix=0.5, dly_type=0, rev_type=2, rev_decay=0.5, rev_mix=0.2),
    # 7 lo-fi: crush, tremolo, tape echo
    _pr(drv_type=3, drv_amt=0.55, drv_tone=0.55, drv_level=0.9, mod_type=3, mod_rate=4.0, mod_depth=0.4,
        mod_mix=1.0, dly_type=2, sync=5, fb=0.35, mix=0.3, mode=2, hicut=4000.0),
    # 8 shimmer-ish wash: fuzz into runaway ping-pong and hall
    _pr(drv_type=2, drv_amt=0.3, drv_tone=0.5, drv_level=0.8, time=0.25, fb=0.85, mix=0.45, hicut=4000.0,
        locut=300.0, moddepth=0.3, rev_type=3, rev_decay=0.8, rev_mix=0.35),
]
PRESET_NAMES = ["PING-PONG PLATE", "CHORUS SLAP", "DOTTED HALL", "DARK ECHO", "AMBIENT", "PHASE PLATE", "LO-FI",
                "WASH"]

ROUTES = ["1+2 Stereo", "1 to both", "2 to both", "Mono sum"]
LATENCY_NAMES = ["Low", "Mid", "Safe"]
LATENCY_FRAMES = [384, 768, 1536]          # output buffer: 8 / 16 / 32 ms
P_SOURCE = Param("source", "Source", "source", scope="ui")
P_ROUTE = Param("route", "Channels", "enum", opts=ROUTES, scope="ui")
P_INGAIN = Param("ingain", "Input gain", "lin", 0, 4, 0.05, f_gain, scope="ui")
P_LEVEL = Param("outgain", "Level", "lin", 0, 1.5, 0.02, f_gain, scope="ui")
P_TRAILS = Param("trails", "Trails", "enum", opts=OFFON, scope="ui")
P_LATENCY = Param("latency", "Latency", "enum", opts=LATENCY_NAMES, scope="ui")
P_PRESET = Param("preset", "Preset", "preset", scope="ui")
P_RESET = Param("reset", "Reset", "action", scope="ui")
P_CLOCK = Param("clock", "MIDI clock", "enum", opts=["Ignore", "Follow"], scope="ui")
P_CCMAP = Param("ccmap", "CC map", "action", scope="ui")
P_CCRESET = Param("ccreset", "Clear map", "action", scope="ui")

# What a MIDI CC can move. "on" switches the effect, "tap" is a tap-tempo footswitch.
CC_TARGETS = {p.key: p for p in DELAY_PARAMS + DRIVE_PARAMS + MOD_PARAMS + REVERB_PARAMS + EQ_PARAMS}
CC_TARGETS["on"] = Param("on", "On/bypass", "switch")
CC_TARGETS["tap"] = Param("tap", "Tap", "tap")
FIXED_CCS = {120, 121, 123}


def cc_label(key):
    p = CC_TARGETS.get(key)
    if p is None:
        return key.upper()
    sec = {"drv": "DRIVE ", "mod": "MOD ", "rev": "REVERB ", "eq": "EQ "}.get(key.split("_")[0], "" if key in ("on", "tap") else "DELAY ")
    return sec + p.label.upper()


# Order of P_* in bfx.c
ENGINE_KEYS = ["on", "time", "fb", "mix", "mode", "width", "hicut", "locut", "moddepth", "modrate", "trails",
               "ingain", "outgain", "route", "drv_type", "drv_amt", "drv_tone", "drv_level", "mod_type", "mod_rate",
               "mod_depth", "mod_mix", "dly_type", "rev_type", "rev_decay", "rev_pre", "rev_damp", "rev_mod", "rev_mix",
               "eq_type", "eq_low", "eq_lowf", "eq_mid", "eq_midf", "eq_q", "eq_high", "eq_highf"]
STATUS_KEYS = ["in_l", "in_r", "out_l", "out_r", "load", "underruns", "cap_xruns", "rs_under", "fill", "ppm",
               "cap_open", "play_open", "rt", "cap_period", "play_buffer", "cur_time", "blocks", "cap_rate",
               "target", "priming", "cap_ch", "rs_over", "play_period", "rt_cap", "cap_frames", "cap_reads", "cap_raw",
               "cap_buffer", "cap_opens", "wpos", "nan_resets", "dly_tail", "rev_tail"]


def delay_seconds(p, bpm=None):
    """Delay time actually used: free time, or the note value at the tempo (preset tempo or MIDI clock)."""
    s = int(p["sync"])
    if s == 0:
        return float(p["time"])
    t = 60.0 / (bpm or p["tempo"]) * SYNC[s][1]
    while t > MAX_TIME:                 # too long for the line: drop an octave rather than clip
        t /= 2
    return t


# ----------------------------------------------------------------------------- native engine

def load_lib():
    import ctypes as C
    import platform
    m = platform.machine().lower()
    name = {"aarch64": "libbfx_aarch64.so", "arm64": "libbfx_aarch64.so", "x86_64": "libbfx_x86_64.so",
            "amd64": "libbfx_x86_64.so"}.get(m)
    if not name or not sys.platform.startswith("linux"):
        _fatal(f"no native engine for {sys.platform}/{m} (BrickFX needs Linux: the Brick or a Linux PC)")
    lib = C.CDLL(os.path.join(HERE, "native", name))
    vp, ci, cs = C.c_void_p, C.c_int, C.c_char_p
    lib.eng_create.restype, lib.eng_create.argtypes = vp, [ci, ci]
    for fn, args in (("eng_set_params", [vp, vp]), ("eng_process", [vp, vp, vp, ci]), ("eng_clear", [vp]),
                     ("eng_set_capture", [vp, cs]), ("eng_set_buffer", [vp, ci])):
        f = getattr(lib, fn)
        f.restype, f.argtypes = None, args
    lib.eng_status.restype, lib.eng_status.argtypes = ci, [vp, vp, ci]
    lib.eng_scope.restype, lib.eng_scope.argtypes = ci, [vp, ci, vp, ci]
    lib.eng_start.restype, lib.eng_start.argtypes = ci, [vp, cs, cs, ci]
    for fn in ("eng_device", "eng_cap_device", "eng_error", "eng_cap_error"):
        getattr(lib, fn).restype, getattr(lib, fn).argtypes = cs, [vp]
    print(f"dsp: native {name} (version {lib.bfx_version()})", flush=True)
    return lib


class Engine:
    def __init__(self, lib):
        self.lib = lib
        self.h = lib.eng_create(SR, BLOCK)
        self.pbuf = np.zeros(len(ENGINE_KEYS), np.float32)
        self.sent = None
        self.st = np.zeros(len(STATUS_KEYS), np.float32)
        self.s = dict.fromkeys(STATUS_KEYS, 0.0)
        self.threaded = False
        self.lock = threading.Lock()

    def push(self, values):
        t = tuple(float(values[k]) for k in ENGINE_KEYS)
        with self.lock:
            if t != self.sent:
                self.pbuf[:] = t
                self.lib.eng_set_params(self.h, self.pbuf.ctypes.data)
                self.sent = t

    def refresh(self):
        self.lib.eng_status(self.h, self.st.ctypes.data, len(self.st))
        self.s = dict(zip(STATUS_KEYS, self.st.tolist()))
        return self.s

    def start_alsa(self, cap_devs, frames):
        play = ["plughw:0,0", "default"] if ALSA_DEVICE in ("default", "auto") else [ALSA_DEVICE, "plughw:0,0", "default"]
        ok = self.lib.eng_start(self.h, "\n".join(play).encode(), "\n".join(cap_devs).encode(), frames)
        self.threaded = bool(ok)
        return ok

    def set_capture(self, cap_devs):
        self.lib.eng_set_capture(self.h, "\n".join(cap_devs).encode())

    def text(self, fn):
        return getattr(self.lib, fn)(self.h).decode(errors="replace")


def capture_cards():
    """[(card id, name)] of the USB sound cards that can record. Card 0 is the Brick's own codec."""
    out = []
    try:
        with open(ASOUND_CARDS) as f:
            lines = f.read().splitlines()
    except OSError:
        return out
    base = os.path.dirname(ASOUND_CARDS)
    for ln in lines:
        m = re.match(r"\s*(\d+)\s+\[(\S+)\s*\]:\s*(.*)", ln)
        if not m:
            continue
        n, cid, desc = int(m.group(1)), m.group(2), m.group(3)
        if n == 0 or not glob.glob(os.path.join(base, f"card{n}", "pcm*c")):
            continue
        name = desc.split(" - ", 1)[1] if " - " in desc else desc
        out.append((cid, " ".join(name.split()) or cid))
    return out


def card_number(cid):
    try:
        with open(ASOUND_CARDS) as f:
            for ln in f:
                m = re.match(r"\s*(\d+)\s+\[(\S+)\s*\]", ln)
                if m and m.group(2) == cid:
                    return int(m.group(1))
    except OSError:
        pass
    return None


class DesktopAudio:
    """Desktop test: a built-in stereo test phrase goes through the delay and out of SDL."""

    def __init__(self, engine):
        self.engine, self.ok, self.error = engine, False, None
        self.t = 0
        self.inbuf = np.zeros((BLOCK, 2), np.float32)
        self.outbuf = np.zeros((BLOCK, 2), np.float32)
        self.fifo = np.zeros((0, 2), np.float32)
        self.test = self._phrase()
        self.load = 0.0
        try:
            pygame.mixer.pre_init(SR, 32, 2, 512, allowedchanges=0)
            pygame.mixer.init()
            from pygame._sdl2.mixer import set_post_mix
            set_post_mix(self._cb)
            self.ok = True
            print("audio: SDL", pygame.mixer.get_init(), flush=True)
        except Exception as e:
            self.error = str(e)
            print("audio: desktop output unavailable:", e, flush=True)

    @staticmethod
    def _phrase():
        """Two bars of short plucks, left and right, so the echoes are easy to hear."""
        n = int(SR * 4.0)
        x = np.zeros((n, 2), np.float32)
        notes = [(0.0, 220, 0.5), (0.5, 330, 0.5), (1.5, 262, 0.5), (2.0, 196, 0.5)]
        for t0, f, pan in notes:
            i0, ln = int(t0 * SR), int(0.25 * SR)
            tt = np.arange(ln) / SR
            s = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 18) * 0.4
            x[i0:i0 + ln, 0] += s * (1 - pan * 0.6)
            x[i0:i0 + ln, 1] += s * (0.4 + pan * 0.6)
        return x

    def _block(self):
        n = len(self.test)
        idx = (self.t + np.arange(BLOCK)) % n
        self.inbuf[:] = self.test[idx]
        self.t = (self.t + BLOCK) % n
        self.engine.lib.eng_process(self.engine.h, self.inbuf.ctypes.data, self.outbuf.ctypes.data, BLOCK)
        return self.outbuf.copy()

    def _cb(self, _pm, mv):
        a = np.asarray(mv).view(np.float32)
        frames = len(a) // 2
        parts, have = [self.fifo], len(self.fifo)
        while have < frames:
            b = self._block()
            parts.append(b)
            have += len(b)
        buf = np.concatenate(parts)
        a[:] = buf[:frames].reshape(-1)
        self.fifo = buf[frames:]


# ----------------------------------------------------------------------------- MIDI in

class Midi:
    """Raw ALSA MIDI devices (/dev/snd/midiC*D*): CCs (learn, mapped parameters), program change (presets),
    MIDI clock (tempo). Rescans every 2 s, so a USB device can be plugged in at any time."""

    def __init__(self, app):
        self.app = app
        self.devices = []
        self.activity = 0.0
        self.ticks = []                 # times of the last MIDI clock ticks
        self.clock_bpm = None
        self.last_tick = 0.0
        self.events = []                # ("cc", num, val) / ("pc", prog), handled by the UI loop
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def clock_running(self):
        return self.clock_bpm is not None and time.time() - self.last_tick < 0.5

    def _tick(self):
        now = time.time()
        if now - self.last_tick > 0.5:
            self.ticks = []
        self.last_tick = now
        self.ticks.append(now)
        if len(self.ticks) > 49:
            self.ticks = self.ticks[-49:]
        if len(self.ticks) >= 25:
            span = (self.ticks[-1] - self.ticks[0]) / (len(self.ticks) - 1) * 24
            bpm = 60.0 / span
            # round to 0.1 BPM and ignore jitter, so the delay time doesn't wobble
            if self.clock_bpm is None or abs(bpm - self.clock_bpm) > 0.4:
                self.clock_bpm = round(bpm, 1)

    def _feed(self, st, data):
        for b in data:
            if b >= 0xF8:
                if b == 0xF8:
                    self._tick()
                continue
            if b >= 0x80:
                st["sysex"] = b == 0xF0
                st["status"] = b if b < 0xF0 else 0
                st["data"] = []
                continue
            if st["sysex"] or not st["status"]:
                continue
            st["data"].append(b)
            hi = st["status"] & 0xF0
            need = 1 if hi in (0xC0, 0xD0) else 2
            if len(st["data"]) == need:
                d = st["data"]
                st["data"] = []
                self.activity = time.time()
                with self.lock:
                    if hi == 0xB0:
                        self.events.append(("cc", d[0], d[1]))
                    elif hi == 0xC0:
                        self.events.append(("pc", d[0], 0))

    def take(self):
        with self.lock:
            ev, self.events = self.events, []
        return ev

    def _run(self):
        if not (sys.platform.startswith("linux") and os.path.isdir("/dev/snd")):
            return
        fds, states, last_scan = {}, {}, 0.0
        while True:
            now = time.time()
            if now - last_scan > 2:
                last_scan = now
                for path in glob.glob("/dev/snd/midiC*D*"):
                    if path not in fds:
                        try:
                            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                            fds[path] = fd
                            states[fd] = dict(status=0, data=[], sysex=False)
                        except OSError:
                            pass
                self.devices = [os.path.basename(p) for p in fds]
            if not fds:
                time.sleep(0.5)
                continue
            try:
                r, _, _ = select.select(list(fds.values()), [], [], 0.5)
            except (OSError, ValueError):
                r = []
            for fd in r:
                try:
                    data = os.read(fd, 512)
                    if data:
                        self._feed(states[fd], data)
                except BlockingIOError:
                    pass
                except OSError:
                    for p, f in list(fds.items()):
                        if f == fd:
                            del fds[p]
                    states.pop(fd, None)
                    try:
                        os.close(fd)
                    except OSError:
                        pass


# ----------------------------------------------------------------------------- dot-matrix LCD (from BrickSampler)
# Dot-matrix LCD look: one orange on a very dark orange glass, 256x192 dots shown 4 px per dot.
# Everything is drawn into a small boolean grid; compose() turns it into dots with a few C-speed blits.
LCD_W, LCD_H, DOT = 256, 192, 4
LCD_BG = (30, 10, 2)          # glass
LCD_GHOST = (52, 20, 5)       # unlit dots, just visible
LCD_LIT = (255, 138, 36)      # lit dots
LCD_BAR = (235, 127, 33)      # solid selection bars (text inside is punched out)
# Display colours: (glass, unlit dot, lit dot, selection bar)
PALETTES = {
    "Orange": ((30, 10, 2), (52, 20, 5), (255, 138, 36), (235, 127, 33)),
    "Blue": ((4, 12, 34), (12, 26, 58), (150, 205, 255), (130, 190, 250)),
    "Red": ((28, 3, 4), (50, 8, 9), (255, 52, 46), (235, 44, 40)),
    "Green": ((4, 22, 8), (10, 40, 16), (110, 255, 120), (96, 236, 108)),
}
PALETTE_NAMES = list(PALETTES)

_FONT_SRC = """
A .###. #...# #...# ##### #...# #...# #...#
B ####. #...# #...# ####. #...# #...# ####.
C .###. #...# #.... #.... #.... #...# .###.
D ###.. #..#. #...# #...# #...# #..#. ###..
E ##### #.... #.... ####. #.... #.... #####
F ##### #.... #.... ####. #.... #.... #....
G .###. #...# #.... #.### #...# #...# .####
H #...# #...# #...# ##### #...# #...# #...#
I .###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.
J ..### ...#. ...#. ...#. ...#. #..#. .##..
K #...# #..#. #.#.. ##... #.#.. #..#. #...#
L #.... #.... #.... #.... #.... #.... #####
M #...# ##.## #.#.# #.#.# #...# #...# #...#
N #...# #...# ##..# #.#.# #..## #...# #...#
O .###. #...# #...# #...# #...# #...# .###.
P ####. #...# #...# ####. #.... #.... #....
Q .###. #...# #...# #...# #.#.# #..#. .##.#
R ####. #...# #...# ####. #.#.. #..#. #...#
S .#### #.... #.... .###. ....# ....# ####.
T ##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..
U #...# #...# #...# #...# #...# #...# .###.
V #...# #...# #...# #...# #...# .#.#. ..#..
W #...# #...# #...# #.#.# #.#.# #.#.# .#.#.
X #...# #...# .#.#. ..#.. .#.#. #...# #...#
Y #...# #...# .#.#. ..#.. ..#.. ..#.. ..#..
Z ##### ....# ...#. ..#.. .#... #.... #####
0 .###. #...# #..## #.#.# ##..# #...# .###.
1 ..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.
2 .###. #...# ....# ...#. ..#.. .#... #####
3 ##### ...#. ..#.. ...#. ....# #...# .###.
4 ...#. ..##. .#.#. #..#. ##### ...#. ...#.
5 ##### #.... ####. ....# ....# #...# .###.
6 ..##. .#... #.... ####. #...# #...# .###.
7 ##### ....# ...#. ..#.. .#... .#... .#...
8 .###. #...# #...# .###. #...# #...# .###.
9 .###. #...# #...# .#### ....# ...#. .##..
. ..... ..... ..... ..... ..... .##.. .##..
, ..... ..... ..... ..... .##.. ..#.. .#...
; ..... .##.. .##.. ..... .##.. ..#.. .#...
: ..... .##.. .##.. ..... .##.. .##.. .....
- ..... ..... ..... ##### ..... ..... .....
+ ..... ..#.. ..#.. ##### ..#.. ..#.. .....
% ##... ##..# ...#. ..#.. .#... #..## ...##
/ ..... ....# ...#. ..#.. .#... #.... .....
_ ..... ..... ..... ..... ..... ..... #####
> .#... ..#.. ...#. ....# ...#. ..#.. .#...
< ...#. ..#.. .#... #.... .#... ..#.. ...#.
( ...#. ..#.. .#... .#... .#... ..#.. ...#.
) .#... ..#.. ...#. ...#. ...#. ..#.. .#...
[ .###. .#... .#... .#... .#... .#... .###.
] .###. ...#. ...#. ...#. ...#. ...#. .###.
= ..... ..... ##### ..... ##### ..... .....
# .#.#. .#.#. ##### .#.#. ##### .#.#. .#.#.
' ..#.. ..#.. .#... ..... ..... ..... .....
! ..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#..
? .###. #...# ....# ...#. ..#.. ..... ..#..
* ..... ..#.. #.#.# .###. #.#.# ..#.. .....
& .##.. #..#. #.#.. .#... #.#.# #..#. .##.#
"""


def _build_font():
    font = {}
    for line in _FONT_SRC.strip().splitlines():
        ch, *rows = line.split()
        font[ch] = np.array([[c == "#" for c in r] for r in rows], bool)
    font[" "] = np.zeros((7, 5), bool)
    return font


class LCD:
    def __init__(self, glow=False, palette="Orange"):
        self.font = _build_font()
        self._scaled = {}
        self._strings = {}
        self.lit = np.zeros((LCD_H, LCD_W), bool)
        self.holes = np.zeros((LCD_H, LCD_W), bool)
        self.bars = []
        self.glow = glow
        self.palette = palette if palette in PALETTES else "Orange"
        self.bar_col = PALETTES[self.palette][3]
        # what is on screen now, so compose() only redraws what changed
        self.prev = None
        # Pre-rendered full-screen layers: unlit dots, and the extra brightness of a lit dot.
        tile = np.array([[0.55, 0.95, 0.55, 0], [0.95, 1, 0.95, 0], [0.55, 0.95, 0.55, 0], [0, 0, 0, 0]], np.float32)
        cells = np.tile(tile, (LCD_H, LCD_W))[..., None]
        bg, ghost, lit, bar = (np.array(c, np.float32) for c in PALETTES[self.palette])
        off = bg + cells * (ghost - bg)
        on = bg + cells * (lit - bg)
        self.off_s = self._surf(off)
        self.diff_s = self._surf(on - off)
        self.hole_s = self._surf(cells * (bar - bg))       # subtracted from a bar where text is punched out
        self.low = pygame.Surface((LCD_W, LCD_H), 0, 32)
        self.low_h = pygame.Surface((LCD_W, LCD_H), 0, 32)
        self.tmp = pygame.Surface((LCD_W * DOT, LCD_H * DOT), 0, 32)
        self.glow_col = np.uint32((40 << 16) | (14 << 8) | 3)

    @staticmethod
    def _surf(a):
        return pygame.surfarray.make_surface(np.clip(a, 0, 255).astype(np.uint8).transpose(1, 0, 2))

    def clear(self):
        self.lit[:] = False
        self.holes[:] = False
        self.bars = []

    # ---- drawing (coordinates in dots)
    def glyph(self, ch, scale):
        key = (ch, scale)
        g = self._scaled.get(key)
        if g is None:
            base = self.font.get(ch)
            if base is None:
                base = self.font["?"]
            g = self._scaled[key] = np.kron(base, np.ones((scale, scale), bool)) if scale > 1 else base
        return g

    @staticmethod
    def width(s, scale=1):
        return max(0, len(s) * 6 * scale - scale)

    def fit(self, s, max_w, scale=1):
        s = s.upper()
        n = (max_w + scale) // (6 * scale)
        return s if len(s) <= n else s[:max(0, n - 1)] + "."

    def string(self, s, scale):
        """Whole string as one dot array, built once and reused (much cheaper than per letter)."""
        key = (s, scale)
        a = self._strings.get(key)
        if a is None:
            if len(self._strings) > 3000:
                self._strings.clear()
            parts = []
            gap = np.zeros((7 * scale, scale), bool)
            for i, ch in enumerate(s):
                if i:
                    parts.append(gap)
                parts.append(self.glyph(ch, scale))
            a = self._strings[key] = np.hstack(parts) if parts else np.zeros((7 * scale, 0), bool)
        return a

    def text(self, x, y, s, inv=False, scale=1):
        s = s.upper()
        g = self.string(s, scale)
        h, w = g.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, LCD_W), min(y + h, LCD_H)
        if x1 > x0 and y1 > y0:
            sub = g[y0 - y:y1 - y, x0 - x:x1 - x]
            if inv:
                self.holes[y0:y1, x0:x1] |= sub
                self.lit[y0:y1, x0:x1] &= ~sub
            else:
                self.lit[y0:y1, x0:x1] |= sub
        return self.width(s, scale)

    def rtext(self, xr, y, s, inv=False, scale=1):
        return self.text(xr - self.width(s, scale), y, s, inv, scale)

    def ctext(self, xc, y, s, inv=False, scale=1):
        return self.text(xc - self.width(s, scale) // 2, y, s, inv, scale)

    def fill(self, x, y, w, h, on=True):
        self.lit[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = on

    def erase(self, x, y, w, h):
        """Blank an area completely: dots, punched text and the parts of any selection bar inside it."""
        self.lit[y:y + h, x:x + w] = False
        self.holes[y:y + h, x:x + w] = False
        kept = []
        for bx, by, bw, bh in self.bars:
            if bx >= x + w or bx + bw <= x or by >= y + h or by + bh <= y:
                kept.append((bx, by, bw, bh))
                continue
            if by < y:
                kept.append((bx, by, bw, y - by))
            if by + bh > y + h:
                kept.append((bx, y + h, bw, by + bh - (y + h)))
            top, bottom = max(by, y), min(by + bh, y + h)
            if bx < x:
                kept.append((bx, top, x - bx, bottom - top))
            if bx + bw > x + w:
                kept.append((x + w, top, bx + bw - (x + w), bottom - top))
        self.bars = [b for b in kept if b[2] > 0 and b[3] > 0]

    def bar(self, x, y, w, h):
        """Solid selection block; text drawn into it with inv=True is punched out."""
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, LCD_W - x), min(h, LCD_H - y)
        self.lit[y:y + h, x:x + w] = False
        self.bars.append((x, y, w, h))

    def frame(self, x, y, w, h):
        self.lit[y, x:x + w] = True
        self.lit[y + h - 1, x:x + w] = True
        self.lit[y:y + h, x] = True
        self.lit[y:y + h, x + w - 1] = True

    def hline(self, x, y, w, dotted=False):
        self.lit[y, x:x + w:2 if dotted else 1] = True

    def vline(self, x, y, h, dotted=False):
        self.lit[y:y + h:2 if dotted else 1, x] = True

    def line(self, x0, y0, x1, y1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.clip(np.rint(np.linspace(x0, x1, n)).astype(int), 0, LCD_W - 1)
        ys = np.clip(np.rint(np.linspace(y0, y1, n)).astype(int), 0, LCD_H - 1)
        self.lit[ys, xs] = True

    def polyline(self, pts):
        for a, b in zip(pts, pts[1:]):
            self.line(a[0], a[1], b[0], b[1])

    def columns(self, x, top, bottom, dotted=False):
        """Vertical runs per column: rows top[i]..bottom[i] lit in column x+i (waveforms, meters)."""
        top, bottom = np.asarray(top), np.asarray(bottom)
        y0, y1 = int(max(top.min(), 0)), int(min(bottom.max() + 1, LCD_H))
        if y1 <= y0:
            return
        ys = np.arange(y0, y1)[:, None]
        m = (ys >= top[None, :]) & (ys <= bottom[None, :])
        if dotted:
            m &= ((ys + np.arange(len(top))[None, :]) % 2 == 0)
        self.lit[y0:y1, x:x + len(top)] |= m

    # ---- compose the dots onto the real screen
    def compose(self, dest, full=False):
        """Turn the dot grid into pixels. Only the bands of the display that changed since the last
        call are redrawn; returns the changed screen rectangles (empty list = nothing to update)."""
        barmask = np.zeros((LCD_H, LCD_W), bool)
        for x, y, w, h in self.bars:
            barmask[y:y + h, x:x + w] = True
        state = (self.lit.copy(), self.holes.copy(), barmask)
        if full or self.glow or self.prev is None:
            regions = [(0, 0, LCD_W, LCD_H)]
        else:
            diff = (state[0] ^ self.prev[0]) | (state[1] ^ self.prev[1]) | (state[2] ^ self.prev[2])
            regions = self._regions(diff)
        self.prev = state
        if not regions:
            return []
        pygame.surfarray.blit_array(self.low, np.where(self.lit, np.uint32(0xFFFFFF), np.uint32(0)).T)
        has_holes = bool(self.holes.any())
        if has_holes:
            pygame.surfarray.blit_array(self.low_h, np.where(self.holes, np.uint32(0xFFFFFF), np.uint32(0)).T)
        rects = []
        for x, y, w, h in regions:
            r = pygame.Rect(x * DOT, y * DOT, w * DOT, h * DOT)
            dest.blit(self.off_s, r.topleft, area=r)
            part = pygame.transform.scale(self.low.subsurface((x, y, w, h)), r.size)
            part.blit(self.diff_s, (0, 0), area=r, special_flags=pygame.BLEND_MULT)
            if self.glow:
                halo = pygame.transform.smoothscale(self.low, (LCD_W * 2, LCD_H * 2))
                halo = pygame.transform.smoothscale(halo, (LCD_W * DOT, LCD_H * DOT))
                halo.fill((40, 14, 3), special_flags=pygame.BLEND_MULT)
                dest.blit(halo, (0, 0), special_flags=pygame.BLEND_ADD)
            dest.blit(part, r.topleft, special_flags=pygame.BLEND_ADD)
            for bx, by, bw, bh in self.bars:
                ix, iy = max(bx, x), max(by, y)
                iw, ih = min(bx + bw, x + w) - ix, min(by + bh, y + h) - iy
                if iw <= 0 or ih <= 0:
                    continue
                br = pygame.Rect(ix * DOT, iy * DOT, iw * DOT, ih * DOT)
                dest.fill(self.bar_col, br)
                if has_holes and self.holes[iy:iy + ih, ix:ix + iw].any():
                    hp = pygame.transform.scale(self.low_h.subsurface((ix, iy, iw, ih)), br.size)
                    hp.blit(self.hole_s, (0, 0), area=br, special_flags=pygame.BLEND_MULT)
                    dest.blit(hp, br.topleft, special_flags=pygame.BLEND_SUB)
            rects.append(r)
        return rects

    @staticmethod
    def _regions(diff):
        """Group changed dots into a few rectangles: runs of changed rows, each with its column span."""
        rows = np.flatnonzero(diff.any(axis=1))
        if not len(rows):
            return []
        out = []
        start = prev = rows[0]
        for r in list(rows[1:]) + [None]:
            if r is not None and r - prev <= 3:
                prev = r
                continue
            band = diff[start:prev + 1]
            cols = np.flatnonzero(band.any(axis=0))
            out.append((int(cols[0]), int(start), int(cols[-1] - cols[0] + 1), int(prev - start + 1)))
            if r is not None:
                start = prev = r
        return out


PAGES = ["I/O", "CHAIN", "DRV", "MOD", "DLY", "REV", "EQ", "SET"]
FX_PAGES = ("CHAIN", "DRV", "MOD", "DLY", "REV", "EQ")
# oscilloscope time windows (frames shown across the screen) and gains
SCOPE_SPANS = [64, 128, 256, 512, 1024, 2048, 4096]
SCOPE_GAINS = [1, 2, 4, 8, 16]
SCOPE_MODES = ["L / R", "OVERLAY", "XY"]
P_DISPLAY = Param("display", "Colour", "enum", opts=PALETTE_NAMES, scope="ui")


class App:
    def __init__(self, args):
        self.args = args
        pygame.display.init()
        pygame.font.init()
        pygame.joystick.init()
        self.controllers, self.joysticks, self.ctrl_mod = [], [], None
        raw = ON_DEVICE if RAW_JOYSTICK is None else RAW_JOYSTICK
        if not raw:
            try:
                from pygame._sdl2 import controller as ctrl_mod
                ctrl_mod.init()
                self.ctrl_mod = ctrl_mod
            except Exception:
                pass
        self._open_pads()
        fullscreen = args.fullscreen or ON_DEVICE
        self.scale = 1.0 if fullscreen else args.scale
        size = (SCREEN_W, SCREEN_H) if fullscreen else (int(SCREEN_W * self.scale), int(SCREEN_H * self.scale))
        self.window = pygame.display.set_mode(size, pygame.FULLSCREEN if fullscreen else 0)
        pygame.display.set_caption("BrickFX")
        if fullscreen:
            pygame.mouse.set_visible(False)
        self.surf = pygame.Surface((SCREEN_W, SCREEN_H)) if self.scale != 1.0 else self.window
        print(f"video: driver {pygame.display.get_driver()}, window {self.window.get_size()}", flush=True)

        self._lcds = {}
        self.load_settings()
        self.lcd = self.make_lcd()
        self.splash("1/3  Loading presets")
        self.load_presets()
        self.splash("2/3  Starting audio")
        self.engine = Engine(load_lib())
        self.cur_card = None
        self.audio = None
        self.last_cap_change = time.time()
        if ON_DEVICE or AUDIO_BACKEND == "alsa":
            card = self.resolve_card()
            self.cur_card = card
            print(f"audio: input card {card}, devices {self.cap_devs(card)}; cards {capture_cards()}", flush=True)
            if not self.engine.start_alsa(self.cap_devs(card), LATENCY_FRAMES[self.ui["latency"]]):
                print("audio: libasound not usable", flush=True)
        else:
            self.audio = DesktopAudio(self.engine)
        self.splash("3/3  Starting MIDI")
        self.midi = Midi(self)
        if self.engine.threaded:
            threading.Thread(target=self._audio_log, daemon=True).start()
        self.push()
        time.sleep(max(0.0, 1.0 - (time.time() - self._splash_t0)))

        self.page = 1
        self.sel = [0] * len(PAGES)
        self.held = {}
        self.toast = ("", 0.0)
        self.select_combo = False
        self.y_used = False
        self.taps = []
        self.tap_flash = 0.0
        self.scope_on = False
        self._scope_ref = None
        self.select_used = False
        self._scope_buf = np.zeros((SCOPE_SPANS[-1] * 2, 2), np.float32)
        self.learn = None
        self.dirty = True
        self.presets_dirty = 0.0
        self.running = True
        self.last_input = "pad" if ON_DEVICE else "keyboard"
        self.last_card_scan = 0.0
        self.logged_inputs = 0
        self.key_map = {
            pygame.KSCAN_UP: "up", pygame.KSCAN_DOWN: "down", pygame.KSCAN_LEFT: "left", pygame.KSCAN_RIGHT: "right",
            pygame.KSCAN_Z: "a", pygame.KSCAN_RETURN: "a", pygame.KSCAN_X: "b", pygame.KSCAN_ESCAPE: "b",
            pygame.KSCAN_BACKSPACE: "b", pygame.KSCAN_SPACE: "x", pygame.KSCAN_LSHIFT: "y", pygame.KSCAN_RSHIFT: "y",
            pygame.KSCAN_T: "tap", pygame.KSCAN_LEFTBRACKET: "l1", pygame.KSCAN_RIGHTBRACKET: "r1",
            pygame.KSCAN_COMMA: "l2", pygame.KSCAN_PERIOD: "r2", pygame.KSCAN_TAB: "select", pygame.KSCAN_O: "scope",
        }
        if self.engine.threaded and self.cur_card is None:
            self.show_toast("Connect a USB audio device", 5)
        else:
            self.show_toast("Hold SELECT to see the controls" if ON_DEVICE else "Desktop test: built-in test sound", 4)

    # ---------- setup
    def splash(self, step):
        print("startup:", step, flush=True)
        if not hasattr(self, "_splash_t0"):
            self._splash_t0 = time.time()
        try:
            n, total = (int(v) for v in step.split()[0].split("/"))
            label = step.split(None, 1)[1]
        except Exception:
            n, total, label = 0, 3, step
        L = self.lcd
        L.clear()
        L.frame(14, 22, 228, 104)
        L.frame(16, 24, 224, 100)
        L.ctext(128, 34, "BRICKFX", scale=3)
        L.ctext(128, 62, "STEREO MULTI-EFFECT")
        L.hline(28, 98, 201, dotted=True)
        x, a = 30, 22.0
        for k in range(9):                                  # ping-pong echoes, as on the delay page
            h = max(1, int(a))
            if k % 2 == 0:
                L.fill(x, 98 - h, 2, h)
            else:
                L.fill(x, 99, 2, h)
            x += 24
            a *= 0.7
        seg_w = (228 - 4 * (total - 1)) / total
        for i in range(total):
            xs = 14 + int(i * (seg_w + 4))
            (L.fill if i < n else L.frame)(xs, 138, int(seg_w), 9)
        L.ctext(128, 156, label)
        L.text(2, 183, "TRIMUI BRICK / KNULLI" if ON_DEVICE else "DESKTOP TEST BUILD")
        L.rtext(254, 183, f"V{VERSION}")
        self.present(full=True)
        pygame.event.pump()

    def present(self, full=False):
        rects = self.lcd.compose(self.surf, full=full)
        if not rects:
            return
        if self.surf is not self.window:
            pygame.transform.smoothscale(self.surf, self.window.get_size(), self.window)
            pygame.display.flip()
        elif full:
            pygame.display.flip()
        else:
            pygame.display.update(rects)

    def _open_pads(self):
        self.controllers, self.joysticks = [], []
        for i in range(pygame.joystick.get_count()):
            try:
                if self.ctrl_mod and self.ctrl_mod.is_controller(i):
                    self.controllers.append(self.ctrl_mod.Controller(i))
                else:
                    j = pygame.joystick.Joystick(i)
                    j.init()
                    self.joysticks.append(j)
            except Exception as e:
                print("pad open failed:", e)
        for j in self.joysticks:
            print(f"input: joystick {j.get_name()!r} buttons={j.get_numbuttons()}", flush=True)

    def make_lcd(self):
        name = PALETTE_NAMES[self.ui["display"]]
        if name not in self._lcds:
            self._lcds[name] = LCD(glow=not ON_DEVICE, palette=name)
        lcd = self._lcds[name]
        lcd.prev = None
        return lcd

    # ---------- settings and presets
    def load_settings(self):
        self.ui = dict(display=0, latency=0, source="auto", route=0, ingain=1.0, outgain=1.0, volume=1.0,
                       trails=1, preset=0, clock=1, ccmap={}, scope_span=3, scope_gain=1, scope_mode=0, scope_src=1)
        try:
            with open(SETTINGS_FILE) as f:
                d = json.load(f)
            self.ui.update({k: v for k, v in d.items() if k in self.ui})
        except Exception:
            pass
        u = self.ui
        u["display"] = int(clamp(int(u["display"]), 0, len(PALETTE_NAMES) - 1))
        u["latency"] = int(clamp(int(u["latency"]), 0, len(LATENCY_NAMES) - 1))
        u["route"] = int(clamp(int(u["route"]), 0, len(ROUTES) - 1))
        u["preset"] = int(clamp(int(u["preset"]), 0, NUM_PRESETS - 1))
        u["scope_span"] = int(clamp(int(u["scope_span"]), 0, len(SCOPE_SPANS) - 1))
        u["scope_gain"] = int(clamp(int(u["scope_gain"]), 0, len(SCOPE_GAINS) - 1))
        u["scope_mode"] = int(clamp(int(u["scope_mode"]), 0, len(SCOPE_MODES) - 1))
        u["scope_src"] = 1 if u["scope_src"] else 0
        for k, lo, hi in (("ingain", 0, 4), ("outgain", 0, 1.5), ("volume", 0, 1)):
            u[k] = float(clamp(float(u[k]), lo, hi))
        cm = u["ccmap"] if isinstance(u["ccmap"], dict) else {}
        u["ccmap"] = {str(int(c)): k for c, k in cm.items() if str(c).isdigit() and k in CC_TARGETS}

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self.ui, f, indent=1)
        except Exception as e:
            print("settings not saved:", e, flush=True)

    def load_presets(self):
        self.presets = [dict(p) for p in FACTORY]
        try:
            with open(PRESETS_FILE) as f:
                d = json.load(f)
            if d.get("version", 1) < PRESETS_VERSION:
                print("presets: older version, factory presets loaded", flush=True)
                return                  # v0.1 presets had only the delay: start from the new factory set
            for i, p in enumerate(d.get("presets", [])[:NUM_PRESETS]):
                self.presets[i] = dict(PRESET_DEFAULTS, **{k: v for k, v in p.items() if k in PRESET_DEFAULTS})
        except Exception:
            pass

    def save_presets(self, toast=False):
        try:
            with open(PRESETS_FILE, "w") as f:
                json.dump({"version": PRESETS_VERSION, "presets": self.presets}, f, indent=1)
            if toast:
                self.show_toast(f"Saved preset {self.ui['preset'] + 1}")
        except Exception as e:
            self.show_toast(f"Not saved: {e}", 4)
        self.presets_dirty = 0.0

    @property
    def p(self):
        return self.presets[self.ui["preset"]]

    def set_preset(self, i):
        self.ui["preset"] = i % NUM_PRESETS
        self.save_settings()
        self.show_toast(f"PRESET {self.ui['preset'] + 1}  {self.preset_desc()}")

    def preset_desc(self):
        p = self.p
        on = ["EQ" if k == "eq_type" else opts[int(p[k])].upper() for _, k, opts in SECTIONS if int(p[k])]
        return " + ".join(on) if on else "ALL OFF"

    # ---------- audio
    def resolve_card(self):
        cards = [c for c, _ in capture_cards()]
        src = self.ui["source"]
        if src == "auto":
            return cards[0] if cards else None
        return src if src in cards else None

    @staticmethod
    def cap_devs(card):
        if card is None:
            return []
        d = f"CARD={card},DEV=0"
        devs = [f"plughw:{d}", f"hw:{d}"]
        num = card_number(card)
        if num is not None:                 # by number too, in case the name doesn't resolve
            devs += [f"plughw:{num},0", f"hw:{num},0"]
        return devs

    def check_card(self):
        """Hot-plug: follow the USB card as it comes and goes (checked every 2 s)."""
        now = time.time()
        if not self.engine.threaded or now - self.last_card_scan < 2:
            return
        self.last_card_scan = now
        card = self.resolve_card()
        if card != self.cur_card:
            print(f"audio: input card {self.cur_card} -> {card}  devices {self.cap_devs(card)}", flush=True)
            self.cur_card = card
            self.engine.set_capture(self.cap_devs(card))
            self.show_toast("USB input connected" if card else "USB input disconnected", 3)

    @staticmethod
    def usb_label(cid):
        """On screen a device is just "USB" ("USB 1", "USB 2" when several are plugged in); its own name
        (a synth, an audio interface...) only goes to log.txt, so the app isn't tied to one instrument."""
        cards = [c for c, _ in capture_cards()]
        if cid not in cards:
            return "USB (off)"
        return f"USB {cards.index(cid) + 1}" if len(cards) > 1 else "USB"

    def card_name(self, cid):
        if cid is None:
            return "none"
        return dict(capture_cards()).get(cid, cid)

    def bpm(self):
        if self.ui["clock"] and self.midi.clock_running() and int(self.p["sync"]):
            return self.midi.clock_bpm
        return None

    def engine_values(self):
        p, u = self.p, self.ui
        v = dict(p)
        v["time"] = delay_seconds(p, self.bpm())
        v["trails"], v["route"], v["ingain"] = u["trails"], u["route"], u["ingain"]
        v["outgain"] = u["outgain"] * u["volume"] ** 2
        return v

    def push(self):
        self.engine.push(self.engine_values())

    def _audio_log(self):
        """Write what the audio threads are doing to log.txt: every second for the first 20 s, then
        every 5 s when something changed. This is what to read when there is no sound."""
        t0, last, last_frames, last_key = time.time(), 0.0, -1.0, None
        while True:
            time.sleep(1.0)
            e, now = self.engine, time.time()
            buf = np.zeros(len(STATUS_KEYS), np.float32)            # own buffer: the UI thread reads too
            e.lib.eng_status(e.h, buf.ctypes.data, len(buf))
            s = dict(zip(STATUS_KEYS, buf.tolist()))
            frames = s["cap_frames"]
            if frames != last_frames:
                self.last_cap_change = now
            rate = (frames - last_frames) / (now - last) if last and last_frames >= 0 else 0.0
            last_frames, last = frames, now
            key = (int(s["cap_open"]), int(s["play_open"]), int(s["priming"]), int(s["cap_opens"]),
                   e.text("eng_cap_error"), e.text("eng_error"), int(s["underruns"] + s["rs_under"] + s["cap_xruns"]))
            if now - t0 < 20 or (key != last_key and int(now - t0) % 5 == 0) or int(now - t0) % 60 == 0:
                print(f"audio: in {e.text('eng_cap_device') or '-'} open={key[0]} opens={key[3]} ch={int(s['cap_ch'])} "
                      f"period={int(s['cap_period'])} buf={int(s['cap_buffer'])} reads={int(s['cap_reads'])} "
                      f"frames={int(frames)} ({rate:.0f}/s) usb_peak={20 * math.log10(max(s['cap_raw'], 1e-6)):.0f}dB "
                      f"in_peak={20 * math.log10(max(s['in_l'], s['in_r'], 1e-6)):.0f}dB err='{key[4]}' | "
                      f"ring fill={s['fill']:.0f}/{int(s['target'])} priming={key[2]} drift={s['ppm']:+.0f}ppm "
                      f"under={int(s['rs_under'])} | out {e.text('eng_device') or '-'} open={key[1]} "
                      f"buf={int(s['play_buffer'])} underruns={int(s['underruns'])} out_peak="
                      f"{20 * math.log10(max(s['out_l'], s['out_r'], 1e-6)):.0f}dB err='{key[5]}' cpu={s['load'] * 100:.0f}% "
                      f"nan_resets={int(s['nan_resets'])}",
                      flush=True)
                last_key = key

    # ---------- params
    def page_layout(self):
        name = PAGES[self.page]
        if name == "DLY":
            t = P_TEMPO if int(self.p["sync"]) else P_TIME
            return [[("DELAY", [P_DLY_TYPE, t, P_SYNC, P_FB, P_MIX, P_MODE])],
                    [("TONE", [P_WIDTH, P_HICUT, P_LOCUT, P_MODD, P_MODR])]]
        if name == "CHAIN":
            return [[("CHAIN", CHAIN_PARAMS)], [("TEMPO", [P_TEMPO])]]
        if name == "DRV":
            return [[("DRIVE", [P_DRV_TYPE, P_DRV_AMT])], [("OUTPUT", [P_DRV_TONE, P_DRV_LEVEL])]]
        if name == "MOD":
            return [[("MOD", [P_MOD_TYPE, P_MOD_RATE])], [("AMOUNT", [P_MOD_DEPTH, P_MOD_MIX])]]
        if name == "EQ":
            return [[("EQ", [P_EQ_TYPE, P_EQ_LOW, P_EQ_LOWF, P_EQ_MID])], [("BANDS", [P_EQ_MIDF, P_EQ_Q, P_EQ_HIGH, P_EQ_HIGHF])]]
        if name == "REV":
            return [[("REVERB", [P_REV_TYPE, P_REV_DECAY, P_REV_PRE])], [("TONE", [P_REV_DAMP, P_REV_MOD, P_REV_MIX])]]
        if name == "I/O":
            return [[("INPUT", [P_SOURCE, P_ROUTE, P_INGAIN]), ("OUTPUT", [P_LEVEL, P_TRAILS, P_LATENCY])], []]
        return [[("PRESET", [P_PRESET, P_RESET]), ("MIDI", [P_CLOCK, P_CCMAP, P_CCRESET]), ("DISPLAY", [P_DISPLAY])], []]

    def page_params(self):
        return [p for col in self.page_layout() for _, ps in col for p in ps]

    def target(self, p):
        return self.ui if p.scope == "ui" else self.p

    def value_text(self, p):
        if p.kind == "source":
            src = self.ui["source"]
            if src == "auto":
                return "USB" if self.cur_card else "USB (none)"          # any USB audio device, first found
            return self.usb_label(src)
        if p.kind == "preset":
            return str(self.ui["preset"] + 1)
        if p.kind == "action":
            if p.key == "ccmap":
                return f"{len(self.ui['ccmap'])} CC"
            return "Press A"
        if p.key == "latency":
            return f"{p.opts[self.ui['latency']]}  {LATENCY_FRAMES[self.ui['latency']] / SR * 1000:.0f} ms"
        if p.key == "tempo" and self.bpm():
            return f"{self.midi.clock_bpm:.0f} MIDI"
        v = self.target(p)[p.key]
        if p.kind == "enum":
            return p.opts[int(v)]
        return p.fmt(v)

    def adjust(self, p, direction, fine):
        if p.kind == "source":
            ids = ["auto"] + [c for c, _ in capture_cards()]
            cur = self.ui["source"]
            i = ids.index(cur) if cur in ids else 0
            self.ui["source"] = ids[clamp(i + direction, 0, len(ids) - 1)]
            self.last_card_scan = 0.0
            return
        if p.kind == "preset":
            self.set_preset(self.ui["preset"] + direction)
            return
        if p.kind == "action":
            return
        d = self.target(p)
        v = d[p.key]
        if p.kind == "enum":
            v = clamp(int(v) + direction, 0, len(p.opts) - 1)
        elif p.kind == "exp":
            v = clamp(v * (1 + p.step) ** (direction * (0.15 if fine else 1)), p.lo, p.hi)
        else:
            step = p.step * (0.1 if fine and p.step < 1 else 1)
            v = round(clamp(v + direction * step, p.lo, p.hi), 6)
            if p.step >= 1:
                v = int(v)
        d[p.key] = v
        if p.scope == "ui":
            self.save_settings()
            if p.key == "latency" and self.engine.threaded:
                self.engine.lib.eng_set_buffer(self.engine.h, LATENCY_FRAMES[v])
            elif p.key == "display":
                self.lcd = self.make_lcd()
        else:
            self.presets_dirty = self.presets_dirty or time.time()

    def do_action(self, key):
        if key == "reset":
            i = self.ui["preset"]
            self.presets[i] = dict(FACTORY[i])
            self.save_presets()
            self.show_toast(f"Preset {i + 1} back to factory")
        elif key == "ccmap":
            m = self.ui["ccmap"]
            self.show_toast("  ".join(f"CC{c} {cc_label(k)}" for c, k in sorted(m.items(), key=lambda x: int(x[0])))
                            or "No CC mapped. B on a value to learn", 5)
        elif key == "ccreset":
            self.ui["ccmap"] = {}
            self.save_settings()
            self.show_toast("CC map cleared")

    def show_toast(self, msg, secs=2.0):
        self.dirty = True
        self.toast = (msg, time.time() + secs)

    def toggle_on(self):
        self.p["on"] = 0 if self.p["on"] else 1
        self.presets_dirty = self.presets_dirty or time.time()
        self.show_toast("ON" if self.p["on"] else "BYPASS", 1.0)

    def tap(self, t=None):
        """Tap tempo: free time follows the tap interval, synced delays get the tapped tempo."""
        t = t or time.time()
        if self.taps and t - self.taps[-1] > 2.0:
            self.taps = []
        self.taps.append(t)
        self.taps = self.taps[-5:]
        self.tap_flash = time.time()
        self.dirty = True
        if len(self.taps) < 2:
            return
        iv = (self.taps[-1] - self.taps[0]) / (len(self.taps) - 1)
        p = self.p
        if int(p["sync"]):
            p["tempo"] = float(clamp(round(60.0 / iv), P_TEMPO.lo, P_TEMPO.hi))
            self.show_toast(f"TAP {p['tempo']:.0f} BPM", 1.2)
        else:
            p["time"] = float(clamp(iv, P_TIME.lo, P_TIME.hi))
            self.show_toast(f"TAP {f_time(p['time']).upper()}", 1.2)
        self.presets_dirty = self.presets_dirty or time.time()

    def handle_midi(self):
        for kind, a, b in self.midi.take():
            if kind == "pc":
                self.set_preset(a % NUM_PRESETS)
                continue
            if a in FIXED_CCS:
                continue
            if self.learn is not None:
                key = self.learn
                for c in [c for c, k in self.ui["ccmap"].items() if k == key]:
                    del self.ui["ccmap"][c]
                self.ui["ccmap"][str(a)] = key
                self.learn = None
                self.save_settings()
                self.show_toast(f"CC{a} -> {cc_label(key)}", 3)
                continue
            key = self.ui["ccmap"].get(str(a))
            p = CC_TARGETS.get(key) if key else None
            if p is None:
                continue
            x = b / 127.0
            if p.kind == "switch":
                self.p["on"] = 1 if b >= 64 else 0
            elif p.kind == "tap":
                prev = getattr(self, "_tap_cc", 0)
                self._tap_cc = b
                if b >= 64 and prev < 64:
                    self.tap()
                continue
            elif p.kind == "enum":
                self.p[p.key] = min(len(p.opts) - 1, int(x * len(p.opts)))
            elif p.kind == "exp":
                self.p[p.key] = p.lo * (p.hi / p.lo) ** x
            else:
                v = p.lo + (p.hi - p.lo) * x
                self.p[p.key] = int(round(v)) if p.step >= 1 else v
            self.presets_dirty = self.presets_dirty or time.time()
            self.dirty = True

    # ---------- buttons
    def press(self, btn):
        now = time.time()
        self.held[btn] = (now, now + 0.32)
        if btn == "y":
            self.y_used = False
            self.y_down = now
        elif "y" in self.held:
            self.y_used = True         # Y was a modifier (fine adjust), not a tap
        if btn == "select":
            self.select_used = False
        elif "select" in self.held:
            self.select_used = True    # SELECT was part of a combo (SELECT+START)
        self.on_button(btn)

    def release_btn(self, btn):
        t0 = self.held.pop(btn, (time.time(), 0))[0]
        self.dirty = True
        if btn == "y" and not self.y_used and self.dialog_free() and time.time() - t0 < 0.6:
            self.tap(t0)                # the press time, so the tempo is as you tapped it
        if btn == "select":
            self.select_combo = False
            if not self.select_used and time.time() - t0 < 0.45 and self.dialog_free():
                self.scope_on = not self.scope_on          # short press: oscilloscope on / off
            self.select_used = False
        if btn == "scope":
            self.scope_on = not self.scope_on

    def dialog_free(self):
        return self.learn is None

    def repeat_buttons(self):
        now = time.time()
        for btn in ("up", "down", "left", "right"):
            if btn in self.held:
                t0, tn = self.held[btn]
                if now >= tn:
                    self.held[btn] = (t0, tn + 0.055)
                    self.on_button(btn, repeat=True)

    def on_button(self, btn, repeat=False):
        if btn in ("vol_up", "vol_down"):
            v = round(clamp(self.ui["volume"] + (0.05 if btn == "vol_up" else -0.05), 0.0, 1.0), 2)
            self.ui["volume"] = v
            self.save_settings()
            self.show_toast(f"VOLUME {round(v * 100)}%", 1.2)
            return
        self.dirty = True
        fine = "y" in self.held
        if btn == "start":
            if "select" in self.held:
                self.select_combo = True
                self.running = False
            else:
                self.save_presets(toast=True)
            return
        if btn == "tap":
            self.tap()
            return
        if self.learn is not None and btn == "b":
            self.learn = None
            self.show_toast("MIDI learn cancelled")
            return
        if btn == "x" and not repeat:
            self.toggle_on()
            return
        if btn in ("l2", "r2") and not repeat:
            self.set_preset(self.ui["preset"] + (1 if btn == "r2" else -1))
            return
        if self.scope_on:
            if btn in ("left", "right", "up", "down", "a", "b") and not (repeat and btn in ("a", "b")):
                self.scope_button(btn)
            return
        params = self.page_params()
        sel = self.sel[self.page] = clamp(self.sel[self.page], 0, len(params) - 1)
        p = params[sel]
        if btn == "up":
            self.sel[self.page] = (sel - 1) % len(params)
        elif btn == "down":
            self.sel[self.page] = (sel + 1) % len(params)
        elif btn in ("left", "right"):
            self.adjust(p, -1 if btn == "left" else 1, fine)
        elif btn == "l1" and not repeat:
            self.page = (self.page - 1) % len(PAGES)
        elif btn == "r1" and not repeat:
            self.page = (self.page + 1) % len(PAGES)
        elif btn == "a":
            if p.kind == "action":
                self.do_action(p.key)
            elif p.kind == "enum" and (p.scope == "fx" or p.key in ("trails", "clock")):
                d = self.target(p)
                d[p.key] = (int(d[p.key]) + 1) % len(p.opts)
                if p.scope == "ui":
                    self.save_settings()
                else:
                    self.presets_dirty = self.presets_dirty or time.time()
        elif btn == "b" and PAGES[self.page] in FX_PAGES and p.key in CC_TARGETS:
            key = p.key
            self.learn = key
            self.show_toast(f"Move a MIDI knob for {cc_label(key)}  (B cancels)", 8)

    # ---------- events
    def handle_events(self):
        for ev in pygame.event.get():
            t = ev.type
            if t == pygame.QUIT:
                self.running = False
            elif t == pygame.KEYDOWN:
                self.last_input = "keyboard"
                mods = pygame.key.get_mods()
                if mods & (pygame.KMOD_CTRL | pygame.KMOD_META):
                    if ev.key == pygame.K_s:
                        self.save_presets(toast=True)
                    elif ev.key == pygame.K_q:
                        self.running = False
                    continue
                sc = ev.scancode
                if sc in self.key_map and self.key_map[sc] not in self.held:
                    self.press(self.key_map[sc])
            elif t == pygame.KEYUP:
                if ev.scancode in self.key_map:
                    self.release_btn(self.key_map[ev.scancode])
            elif t == getattr(pygame, "CONTROLLERBUTTONDOWN", -1):
                self.last_input = "pad"
                btn = self.ctrl_button(ev.button)
                if self.args.debug_input:
                    self.show_toast(f"controller button {ev.button} -> {btn}")
                if btn:
                    self.press(btn)
            elif t == getattr(pygame, "CONTROLLERBUTTONUP", -1):
                btn = self.ctrl_button(ev.button)
                if btn:
                    self.release_btn(btn)
            elif t == getattr(pygame, "CONTROLLERAXISMOTION", -1):
                for axis, btn in ((pygame.CONTROLLER_AXIS_TRIGGERLEFT, "l2"), (pygame.CONTROLLER_AXIS_TRIGGERRIGHT, "r2")):
                    if ev.axis == axis:
                        if ev.value > 16000 and btn not in self.held:
                            self.press(btn)
                        elif ev.value < 8000 and btn in self.held:
                            self.release_btn(btn)
            elif t in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
                if self.controllers:
                    continue
                btn = JOY_BUTTONS.get(ev.button)
                if t == pygame.JOYBUTTONDOWN and (self.args.debug_input or self.logged_inputs < 40):
                    self.logged_inputs += 1
                    print(f"input: joystick button {ev.button} -> {btn}", flush=True)
                    if self.args.debug_input:
                        self.show_toast(f"joystick button {ev.button} -> {btn}")
                if btn:
                    self.last_input = "pad"
                    (self.press if t == pygame.JOYBUTTONDOWN else self.release_btn)(btn)
            elif t == pygame.JOYHATMOTION and not self.controllers:
                x, y = ev.value
                for btn, on in (("left", x < 0), ("right", x > 0), ("down", y < 0), ("up", y > 0)):
                    if on and btn not in self.held:
                        self.press(btn)
                    elif not on and btn in self.held:
                        self.release_btn(btn)
            elif t == pygame.JOYAXISMOTION and not self.controllers and ev.axis in (0, 1):
                neg, pos = ("left", "right") if ev.axis == 0 else ("up", "down")
                for btn, on in ((neg, ev.value < -0.6), (pos, ev.value > 0.6)):
                    if on and btn not in self.held:
                        self.press(btn)
                    elif not on and btn in self.held and -0.3 < ev.value < 0.3:
                        self.release_btn(btn)
            elif t in (getattr(pygame, "CONTROLLERDEVICEADDED", -1), pygame.JOYDEVICEADDED):
                self._open_pads()

    def ctrl_button(self, b):
        m = {
            pygame.CONTROLLER_BUTTON_DPAD_UP: "up", pygame.CONTROLLER_BUTTON_DPAD_DOWN: "down",
            pygame.CONTROLLER_BUTTON_DPAD_LEFT: "left", pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "right",
            pygame.CONTROLLER_BUTTON_A: "b" if SWAP_AB else "a", pygame.CONTROLLER_BUTTON_B: "a" if SWAP_AB else "b",
            pygame.CONTROLLER_BUTTON_X: "y" if SWAP_AB else "x", pygame.CONTROLLER_BUTTON_Y: "x" if SWAP_AB else "y",
            pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "l1", pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "r1",
            pygame.CONTROLLER_BUTTON_BACK: "select", pygame.CONTROLLER_BUTTON_START: "start",
        }
        return m.get(b)

    # ---------- drawing (dots, 256 x 192)
    def draw_sections(self, y0, row_h=11, title_h=11, gap=2):
        L = self.lcd
        sel = self.sel[self.page]
        col_w = 123
        i = 0
        for c, col in enumerate(self.page_layout()):
            x, y = 2 + c * (col_w + 6), y0
            for title, params in col:
                L.text(x, y, title)
                tw = L.width(title)
                L.hline(x + tw + 2, y + 6, col_w - tw - 2, dotted=True)
                y += title_h
                for p in params:
                    is_sel = i == sel
                    learning = self.learn is not None and self.learn == p.key
                    if is_sel:
                        L.bar(x - 1, y - 2, col_w + 2, 11)
                    lw = L.text(x + 1, y, p.label, inv=is_sel)
                    mapped = [c for c, k in self.ui["ccmap"].items() if k == p.key]
                    val = "LEARN..." if learning else self.value_text(p)
                    if mapped and not learning and PAGES[self.page] in FX_PAGES:
                        L.text(x + lw + 4, y, "*", inv=is_sel)          # a MIDI CC moves this one
                    val = L.fit(val, col_w - lw - 14)
                    L.rtext(x + col_w - 2, y, val, inv=is_sel)
                    i += 1
                    y += row_h
                y += gap

    def draw_header(self):
        L = self.lcd
        x = 1
        for i, name in enumerate(PAGES):
            w = L.width(name) + 6
            if i == self.page:
                L.bar(x, 1, w, 11)
                L.text(x + 3, 3, name, inv=True)
            else:
                L.text(x + 3, 3, name)
            x += w + 1
        xr = 254
        on = bool(self.p["on"])
        tag = "ON" if on else "BYP"
        w = L.width(tag) + 6
        if on:
            L.bar(xr - w, 1, w, 11)
            L.text(xr - w + 3, 3, tag, inv=True)
        else:
            L.frame(xr - w, 1, w, 11)
            L.text(xr - w + 3, 3, tag)
        xr -= w + 5
        xr -= L.rtext(xr, 3, f"P{self.ui['preset'] + 1}") + 6
        # tempo blink: quarter notes when synced, otherwise the delay time
        p = self.p
        bpm = self.bpm() or (p["tempo"] if int(p["sync"]) else None)
        period = 60.0 / bpm if bpm else max(0.06, delay_seconds(p))
        lit = (time.time() % period) < min(0.12, period / 2) or time.time() - getattr(self, "tap_flash", 0) < 0.1
        (L.fill if lit else L.frame)(xr - 5, 4, 5, 5)
        L.hline(0, 14, LCD_W)

    def draw_meters(self, y):
        """IN and OUT, left and right, as dotted bars (-48 dB .. 0 dB)."""
        L = self.lcd
        s = self.engine.s
        for col, (label, a, b) in enumerate((("IN", "in_l", "in_r"), ("OUT", "out_l", "out_r"))):
            x0 = 2 + col * 129
            L.text(x0, y + 4, label)
            for row, key in enumerate((a, b)):
                yy = y + row * 8
                L.text(x0 + 21, yy, "LR"[row])
                db = 20 * math.log10(max(s.get(key, 0.0), 1e-5))
                n = int(clamp((db + 48) / 48, 0, 1) * 31 + 0.5)
                for k in range(31):
                    bx = x0 + 29 + k * 3
                    if k < n:
                        L.fill(bx, yy, 2, 7)
                    else:
                        L.fill(bx, yy + 3, 2, 1)
                if db > -0.5:
                    L.text(x0 + 29 + 31 * 3, yy, "!")

    def draw_echoes(self, x, y, w, h):
        """The repeats the delay makes: dry hit, then each echo, height = level. Ping-pong: L up, R down."""
        L = self.lcd
        p = self.p
        if not int(p["dly_type"]):
            self.off_box(x, y, w, h, "DELAY")
            return
        L.frame(x, y, w, h)
        t = delay_seconds(p, self.bpm())
        s = int(p["sync"])
        L.text(x + 4, y + 3, f_time(t).upper().replace(" ", ""), scale=2)
        right = SYNC[s][0] + ("  " + (f"{self.bpm():.0f} MIDI" if self.bpm() else f"{p['tempo']:.0f} BPM")) if s else "FREE"
        L.rtext(x + w - 4, y + 3, right)
        L.rtext(x + w - 4, y + 12, DLY_TYPES[int(p["dly_type"])].upper() + " " + MODES[int(p["mode"])].upper())
        g_top, g_bot = y + 20, y + h - 2
        mid = (g_top + g_bot) // 2
        amp_h = (g_bot - g_top) // 2 - 1
        gx0, gx1 = x + 13, x + w - 8
        L.hline(x + 2, mid, w - 4, dotted=True)
        L.text(x + 3, mid - 10, "L")
        L.text(x + 3, mid + 4, "R")
        mix = p["mix"]
        dry = 1.0 if mix < 0.5 else 2 * (1 - mix)
        wet = 2 * mix if mix < 0.5 else 1.0
        fb = p["fb"]
        mode = int(p["mode"])
        width = p["width"]
        # tone loss per pass, roughly: a dark or thin echo path makes the repeats fade faster
        loss = (min(p["hicut"], 12000) / 12000) ** 0.2 * (1 - min(p["locut"], 1000) / 4000)
        floor_db = -36.0

        def height(a):                          # bar height on a dB scale, so the whole tail shows
            db = 20 * math.log10(max(a, 1e-6))
            return int(clamp((db - floor_db) / -floor_db, 0, 1) * amp_h)

        amps, a = [], wet
        for k in range(24):
            if height(a) < 1:
                break
            amps.append(a)
            a *= fb * loss
        n = clamp(len(amps), 4, 12)
        step = (gx1 - gx0) / n
        bypass = not p["on"]
        dh = max(1, height(dry))
        L.columns(gx0, [mid - dh, mid - dh], [mid + dh, mid + dh])          # the dry hit
        for k, a in enumerate(amps[:n]):
            ex = int(gx0 + (k + 1) * step)
            L.vline(ex, mid - 1, 3)                                         # time grid: one tick per repeat
            hh = max(1, height(a))
            if mode == 1:                       # ping-pong: odd repeats left (up), even right (down)
                side = int(hh * (1 - width))
                up, down = (hh, side) if k % 2 == 0 else (side, hh)
            else:
                up = down = hh
            top, bot = mid - up, mid + down
            L.columns(ex - 1, [top] * 3, [bot] * 3, dotted=bypass)

    # ---- section pictures: one box (2, 35, 252 x 62) above the settings, the same on every FX page
    BOX = (2, 35, 252, 62)

    def off_box(self, x, y, w, h, name):
        L = self.lcd
        for xx in range(x, x + w, 2):
            L.lit[y, xx] = L.lit[y + h - 1, xx] = True
        for yy in range(y, y + h, 2):
            L.lit[yy, x] = L.lit[yy, x + w - 1] = True
        L.ctext(x + w // 2, y + h // 2 - 7, f"{name} OFF", scale=2)
        L.ctext(x + w // 2, y + h // 2 + 12, "LEFT / RIGHT ON TYPE SWITCHES IT ON")

    def draw_chain_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        bypass = not p["on"]
        bw, gap, by, bh = 46, 5, y + 4, 36
        L.hline(x, by + bh // 2, w, dotted=True)
        for i, (name, key, opts) in enumerate(SECTIONS):
            bx = x + i * (bw + gap)
            t = int(p[key])
            on = t and not bypass
            tail = not t and self.engine.s.get({"DELAY": "dly_tail", "REVERB": "rev_tail"}.get(name, ""), 0)
            label = opts[t].upper() if t else ("TAIL" if tail else "OFF")      # switched off, echoes still ringing
            L.erase(bx, by, bw, bh)
            if on:
                L.bar(bx, by, bw, bh)
                L.ctext(bx + bw // 2, by + 7, name, inv=True)
                L.ctext(bx + bw // 2, by + 20, L.fit(label, bw - 4), inv=True)
            else:
                (L.frame if t else self._dotted_frame)(bx, by, bw, bh)
                L.ctext(bx + bw // 2, by + 7, name)
                L.ctext(bx + bw // 2, by + 20, L.fit(label, bw - 4))
            if i < len(SECTIONS) - 1:
                L.text(bx + bw, by + bh // 2 - 3, ">")
        L.ctext(x + w // 2, y + h - 9, "BYPASS - DRY SIGNAL" if bypass else f"PRESET {self.ui['preset'] + 1}  {self.preset_desc()}")

    def _dotted_frame(self, x, y, w, h):
        L = self.lcd
        for xx in range(x, x + w, 2):
            L.lit[y, xx] = L.lit[y + h - 1, xx] = True
        for yy in range(y, y + h, 2):
            L.lit[yy, x] = L.lit[yy, x + w - 1] = True

    def curve(self, x0, ys):
        """A connected line through one y per column, drawn in one go (polyline is far too slow on the Brick)."""
        ys = np.clip(np.rint(np.asarray(ys, float)).astype(int), 0, LCD_H - 1)
        if len(ys) < 2:
            return
        nxt = np.append(ys[1:], ys[-1])
        self.lcd.columns(x0, np.minimum(ys, nxt), np.maximum(ys, nxt))

    def draw_drive_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        t = int(p["drv_type"])
        if not t:
            self.off_box(x, y, w, h, "DRIVE")
            return
        L.frame(x, y, w, h)
        amt, tone = p["drv_amt"], p["drv_tone"]
        L.text(x + 4, y + 3, DRV_TYPES[t].upper(), scale=2)
        L.rtext(x + w - 4, y + 3, f"AMOUNT {amt * 100:.0f}%")
        if t == 3:
            bits = 15 - amt * 12
            hold = 1 + int((1 - tone) ** 2 * 31)
            L.rtext(x + w - 4, y + 12, f"{bits:.0f} BIT  {SR / hold / 1000:.1f} KHZ")
        else:
            L.rtext(x + w - 4, y + 12, f"TONE {700 * 24 ** tone / 1000:.1f} KHZ")
        # transfer curve: input left to right, output bottom to top
        gx0, gx1, top, bot = x + 60, x + w - 60, y + 20, y + h - 4
        mid_x, mid_y = (gx0 + gx1) // 2, (top + bot) // 2
        L.hline(gx0, mid_y, gx1 - gx0, dotted=True)
        L.vline(mid_x, top, bot - top, dotted=True)
        xs = np.linspace(-1, 1, gx1 - gx0)
        if t == 1:
            g = 1 + 24 * amt * amt
            ys = np.tanh(xs * g + 0.12) - np.tanh(0.12)
        elif t == 2:
            g = 1 + 80 * amt * amt
            ys = 0.85 * np.clip(xs * g, -1, 1) + 0.15 * np.tanh(xs * g * 0.5)
        else:
            levels = 2 ** max(1.0, min(6.0, 15 - amt * 12 - 9))     # what is visible at this size
            ys = np.round(xs * levels) / levels
        ys = ys / max(1e-6, np.abs(ys).max())
        self.curve(gx0, mid_y - ys * (bot - top) / 2)
        L.text(x + 4, bot - 7, "IN")
        L.rtext(x + w - 4, bot - 7, f"LEVEL {f_gain(p['drv_level']).upper()}")

    def draw_mod_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        t = int(p["mod_type"])
        if not t:
            self.off_box(x, y, w, h, "MOD")
            return
        L.frame(x, y, w, h)
        rate, depth = p["mod_rate"], p["mod_depth"]
        L.text(x + 4, y + 3, MOD_TYPES[t].upper(), scale=2)
        L.rtext(x + w - 4, y + 3, f_rate(rate).upper())
        L.rtext(x + w - 4, y + 12, f"DEPTH {depth * 100:.0f}%  MIX {p['mod_mix'] * 100:.0f}%")
        # the LFO for left and right, moving in real time; 90 deg apart (180 for auto-pan, 0 for tremolo)
        offset = {3: 0.0, 4: 0.5}.get(t, 0.25)
        cycles = clamp(rate * 2.0, 0.5, 14)
        gx0, gx1 = x + 12, x + w - 4
        n = gx1 - gx0
        ph0 = (time.time() * rate) % 1.0
        amp = max(1.0, depth * 7.5)
        for row, (label, off) in enumerate((("L", 0.0), ("R", offset))):
            cy = y + 29 + row * 17
            L.text(x + 3, cy - 3, label)
            L.hline(gx0, cy, n, dotted=True)
            ph = ph0 + off - np.arange(n) / n * cycles
            self.curve(gx0, cy - np.sin(2 * np.pi * ph) * amp)

    def rt60(self, p=None):
        """Rough reverb time: tank loop time and gain (decay applied 4 times per loop), less the damping."""
        p = p or self.p
        size = REV_SIZE[int(p["rev_type"])]
        loop = 21589 * size / 29761
        dec = 0.2 + 0.78 * p["rev_decay"]
        db_loop = -4 * 20 * math.log10(dec)
        return loop * 60 / db_loop * (0.75 - 0.3 * p["rev_damp"])

    def draw_rev_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        t = int(p["rev_type"])
        if not t:
            self.off_box(x, y, w, h, "REVERB")
            return
        L.frame(x, y, w, h)
        rt = self.rt60()
        pre = p["rev_pre"]
        L.text(x + 4, y + 3, REV_TYPES[t].upper(), scale=2)
        L.rtext(x + w - 4, y + 3, f"DECAY {rt:.1f} S")
        L.rtext(x + w - 4, y + 12, f"PRE-DELAY {pre * 1000:.0f} MS")
        # energy over time: a gap for the pre-delay, dense early part, then the tail (dB scale)
        gx0, gx1, top, bot = x + 4, x + w - 4, y + 21, y + h - 3
        span = clamp(max(rt * 1.15 + pre, 0.6), 0.6, 12.0)
        L.hline(gx0, bot, gx1 - gx0)
        n = gx1 - gx0
        jit = 0.55 + 0.45 * np.random.default_rng(7).random(n)
        tt = np.arange(n) / n * span - pre
        db = -60 * np.maximum(tt, 0) / max(rt, 0.05)
        att = np.minimum(1.0, np.maximum(tt, 0) / 0.02 + 0.2)
        hgt = (np.clip((db + 48) / 48, 0, 1) * (bot - top) * jit * att).astype(int)
        hgt[(tt < 0) | (np.arange(n) % 2 == 1)] = 0                 # a bar every second column
        L.columns(gx0, bot - hgt, np.where(hgt > 0, bot - 1, bot - hgt - 1))
        secs = 1 if span < 4 else 2
        for k in range(1, int(span / secs) + 1):
            xm = gx0 + int(k * secs / span * (gx1 - gx0))
            if xm < gx1 - 10:
                L.vline(xm, bot + 1, 2)
        L.rtext(gx1 - 2, top, f"{span:.0f} S" if span >= 2 else f"{span:.1f} S")

    @staticmethod
    def eq_response(p, freqs):
        """Magnitude in dB of the three EQ bands at freqs (same formulas as eq_coef in bfx.c)."""
        z = np.exp(-2j * np.pi * freqs / SR)
        total = np.zeros(len(freqs))
        for kind, gdb, f, q in ((0, p["eq_low"], p["eq_lowf"], 0.707), (1, p["eq_mid"], p["eq_midf"], p["eq_q"]),
                                (2, p["eq_high"], p["eq_highf"], 0.707)):
            if abs(gdb) < 0.05:
                continue
            A = 10 ** (gdb / 40)
            w0 = 2 * np.pi * min(f, SR * 0.45) / SR
            c, sn = math.cos(w0), math.sin(w0)
            if kind == 1:
                al = sn / (2 * q)
                b, a = (1 + al * A, -2 * c, 1 - al * A), (1 + al / A, -2 * c, 1 - al / A)
            else:
                sa = 2 * math.sqrt(A) * sn / 2 * math.sqrt(2)
                if kind == 0:
                    b = (A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa))
                    a = ((A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa)
                else:
                    b = (A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa))
                    a = ((A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa)
            h = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
            total += 20 * np.log10(np.abs(h))
        return total

    def draw_eq_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        if not int(p["eq_type"]):
            self.off_box(x, y, w, h, "EQ")
            return
        L.frame(x, y, w, h)
        L.text(x + 4, y + 3, "EQ", scale=2)
        L.rtext(x + w - 4, y + 3, f"LOW {p['eq_low']:+.1f}  MID {p['eq_mid']:+.1f}  HIGH {p['eq_high']:+.1f}")
        L.rtext(x + w - 4, y + 12, f"{f_hz(p['eq_lowf'])}  {f_hz(p['eq_midf'])} Q{p['eq_q']:.1f}  {f_hz(p['eq_highf'])}".upper())
        # response 20 Hz .. 20 kHz on a log axis, +-15 dB
        gx0, gx1, top, bot = x + 30, x + w - 4, y + 21, y + h - 3
        mid_y = (top + bot) // 2
        n = gx1 - gx0
        cache_key = tuple(p[k] for k in ("eq_low", "eq_lowf", "eq_mid", "eq_midf", "eq_q", "eq_high", "eq_highf"))
        if getattr(self, "_eq_cache", (None,))[0] != cache_key:
            freqs = 20 * 1000 ** (np.arange(n) / (n - 1))
            self._eq_cache = (cache_key, self.eq_response(p, freqs))
        db = self._eq_cache[1]
        L.hline(gx0, mid_y, n, dotted=True)
        for f in (100, 1000, 10000):
            xm = gx0 + int(math.log10(f / 20) / 3 * (n - 1))
            L.vline(xm, top, bot - top, dotted=True)
            L.ctext(xm, bot - 7, {100: "100", 1000: "1K", 10000: "10K"}[f])
        L.text(x + 3, top - 1, "+15")
        L.text(x + 3, mid_y - 3, "0")
        L.text(x + 3, bot - 7, "-15")
        self.curve(gx0, mid_y - np.clip(db, -15, 15) / 15 * (mid_y - top))

    def footer(self):
        L = self.lcd
        L.hline(0, 178, LCD_W, dotted=True)
        hint = "X BYPASS  Y TAP  L2/R2 PRESET  B LEARN" if self.last_input == "pad" else \
            "SPACE BYPASS  T TAP  , . PRESET  X LEARN"
        L.ctext(128, 183, hint)

    def fx_page(self, graph):
        self.draw_meters(18)
        graph(*self.BOX)
        self.draw_sections(100)
        self.footer()

    def draw_chain(self):
        self.fx_page(self.draw_chain_graph)

    def draw_drv(self):
        self.fx_page(self.draw_drive_graph)

    def draw_mod(self):
        self.fx_page(self.draw_mod_graph)

    def draw_dly(self):
        self.fx_page(self.draw_echoes)

    def draw_eq(self):
        self.fx_page(self.draw_eq_graph)

    def draw_rev(self):
        self.fx_page(self.draw_rev_graph)

    def draw_io(self):
        L = self.lcd
        self.draw_meters(17)
        L.hline(0, 33, LCD_W, dotted=True)
        self.draw_sections(37)
        s = self.engine.s
        x, w, ey = 131, 123, 37
        L.text(x, ey, "STATUS")
        L.hline(x + L.width("STATUS") + 2, ey + 6, w - L.width("STATUS") - 2, dotted=True)
        if self.engine.threaded:
            no_data = s["cap_open"] and time.time() - self.last_cap_change > 1.5
            in_state = ("NO INPUT" if self.cur_card is None else "OPENING" if not s["cap_open"]
                        else "NO DATA" if no_data else "SYNCING" if s["priming"] else "RUNNING")
            usb_db = 20 * math.log10(max(s["cap_raw"], 1e-6))
            rows = [("INPUT", in_state),
                    ("DEVICE", self.usb_label(self.cur_card) if self.cur_card else "-"),
                    ("USB LEVEL", f"{usb_db:.0f} DB" if s["cap_open"] and usb_db > -110 else "SILENT" if s["cap_open"] else "-"),
                    ("RATE", f"{SR / 1000:g} KHZ {int(s['cap_ch'])}CH" if s["cap_open"] else "-"),
                    ("DRIFT", f"{s['ppm']:+.0f} PPM"),
                    ("LATENCY", f"{(s['target'] + s['play_buffer'] + s['cap_period']) / SR * 1000:.0f} MS"),
                    ("DROPOUTS", f"{int(s['underruns'] + s['rs_under'] + s['cap_xruns'])}"),
                    ("CPU", f"{s['load'] * 100:.0f}%")]
        else:
            rows = [("INPUT", "TEST SOUND"), ("OUTPUT", "DESKTOP SDL"), ("RATE", f"{SR / 1000:g} KHZ")]
        for i, (k, v) in enumerate(rows):
            yy = ey + 12 + i * 12
            lw = L.text(x + 1, yy, k)
            L.rtext(x + w - 2, yy, L.fit(v, w - lw - 8))
        err = self.engine.text("eng_cap_error") if self.engine.threaded and not s["cap_open"] else ""
        lines = [L.fit("IN: " + err, 250), ""] if err else ["USB AUDIO IN, BRICK JACK OUT.", "ANY USB SYNTH OR AUDIO INTERFACE"]
        for i, line in enumerate(lines):
            L.text(3, 166 + i * 11, line)

    def draw_set(self):
        L = self.lcd
        self.draw_sections(18)
        x, w, ey = 131, 123, 18
        L.text(x, ey, "ENGINE")
        L.hline(x + L.width("ENGINE") + 2, ey + 6, w - L.width("ENGINE") - 2, dotted=True)
        s = self.engine.s
        rows = [("VERSION", VERSION), ("DSP", "NATIVE C"), ("BLOCK", f"{BLOCK} FR"),
                ("REALTIME", "YES" if s["rt"] else "NO") if self.engine.threaded else ("OUTPUT", "SDL"),
                ("MIDI", ", ".join(self.midi.devices) or "NONE"),
                ("CLOCK", f"{self.midi.clock_bpm:.1f} BPM" if self.midi.clock_running() else "-")]
        for i, (k, v) in enumerate(rows):
            yy = ey + 12 + i * 12
            lw = L.text(x + 1, yy, k)
            L.rtext(x + w - 2, yy, L.fit(v, w - lw - 8))
        for i, line in enumerate(["PRESETS SAVE THEMSELVES. START SAVES NOW.", "B ON AN FX VALUE: MIDI LEARN.",
                                  "PROGRAM CHANGE 0-7 PICKS THE PRESET.", "TAP SELECT: FULL-SCREEN OSCILLOSCOPE."]):
            L.text(3, 133 + i * 12, line)

    def draw_overlays(self):
        L = self.lcd
        msg, until = self.toast
        if msg and time.time() < until:
            msg = L.fit(msg, 236)
            w = L.width(msg) + 12
            x, y = (LCD_W - w) // 2, LCD_H - 22
            L.erase(x - 2, y - 2, w + 4, 19)
            L.frame(x, y, w, 15)
            L.frame(x + 1, y + 1, w - 2, 13)
            L.text(x + 6, y + 4, msg)
        if "select" in self.held and time.time() - self.held["select"][0] > 0.45:
            self.draw_help()

    def draw_help(self):
        L = self.lcd
        L.erase(0, 15, LCD_W, LCD_H - 15)
        if self.last_input == "pad":
            title, rows = "CONTROLS", [("UP / DOWN", "SELECT"), ("LEFT / RIGHT", "CHANGE VALUE"), ("HOLD Y", "FINE ADJUST"),
                                        ("TAP Y", "TAP TEMPO"), ("X", "ON / BYPASS"), ("A", "STEP A SETTING"),
                                        ("B", "MIDI LEARN"), ("L1 / R1", "PAGE"), ("L2 / R2", "PRESET"),
                                        ("TAP SELECT", "OSCILLOSCOPE"), ("START", "SAVE"), ("SELECT+START", "QUIT"),
                                        ("VOLUME", "OUTPUT VOLUME")]
        else:
            title, rows = "KEYBOARD", [("UP / DOWN", "SELECT"), ("LEFT / RIGHT", "VALUE (SHIFT FINE)"), ("T", "TAP TEMPO"),
                                        ("SPACE", "ON / BYPASS"), ("Z / ENTER", "STEP A SETTING"), ("X / ESC", "MIDI LEARN"),
                                        ("[ ]", "PAGE"), (", .", "PRESET"), ("O / TAP TAB", "OSCILLOSCOPE"),
                                        ("CTRL+S / Q", "SAVE / QUIT")]
        L.text(2, 18, title)
        L.rtext(254, 18, "RELEASE TO CLOSE")
        L.hline(0, 28, LCD_W, dotted=True)
        for i, (k, v) in enumerate(rows):
            y = 33 + i * 12
            L.text(4, y, k)
            L.text(100, y, v)

    # ---- full-screen oscilloscope
    def draw_scope(self):
        """Input or output over the whole screen. Triggered on a rising zero crossing so a held note stands still;
        L/R lanes, overlay, or XY (mid up, side across: a goniometer)."""
        L = self.lcd
        u = self.ui
        span = SCOPE_SPANS[u["scope_span"]]
        gain = SCOPE_GAINS[u["scope_gain"]]
        mode = u["scope_mode"]
        src = "OUT" if u["scope_src"] else "IN"
        ms = span / SR * 1000
        L.text(2, 2, f"SCOPE {src}")
        L.ctext(128, 2, f"{SCOPE_MODES[mode]}  X{gain}" if mode == 2 else f"{SCOPE_MODES[mode]}  {ms:.1f} MS  X{gain}")
        L.rtext(254, 2, "ON" if self.p["on"] else "BYP")
        top, bot = 12, 180
        if self.engine.threaded or self.audio:
            buf = self._scope_buf
            got = self.engine.lib.eng_scope(self.engine.h, u["scope_src"], buf.ctypes.data, len(buf))
            data = buf[:got]
        else:
            data = np.zeros((0, 2), np.float32)
        L.ctext(128, 184, "A IN/OUT  B MODE  D-PAD TIME+GAIN  SEL EXIT")
        if mode == 2:                                           # XY / goniometer
            cx, cy, r = 128, (top + bot) // 2, (bot - top) // 2 - 2
            L.hline(cx - r, cy, 2 * r, dotted=True)
            L.vline(cx, top, bot - top, dotted=True)
            for k in range(0, 360, 6):                          # dotted circle at full scale
                a = math.radians(k)
                L.lit[int(cy + r * math.sin(a)), int(cx + r * math.cos(a))] = True
            d = data[-2048:] * gain
            mid = (d[:, 0] + d[:, 1]) * 0.5
            side = (d[:, 0] - d[:, 1]) * 0.5
            xs = np.clip(np.rint(cx + side * r * 1.4), 0, LCD_W - 1).astype(int)
            ys = np.clip(np.rint(cy - mid * r * 1.4), top, bot).astype(int)
            L.lit[ys, xs] = True
            L.text(cx + 3, top + 1, "M")
            L.text(cx + r - 6, cy + 3, "S")
            return
        # trigger (Schmitt, like a real scope): armed below -h, fires going above +h; the picture starts at the
        # zero crossing just before, placed between samples. The newest trigger that leaves a full window wins.
        mono = data[:, 0] + data[:, 1]
        self._scope_start = None
        look = len(mono) - span - 2
        start = float(max(0, look))
        if look > 32:
            sm = np.convolve(mono[:look + 1], np.ones(5) / 5, mode="same")     # a little smoothing against noise
            h = max(1e-4, np.abs(sm).max() * 0.15)
            n = len(sm)
            ev = np.zeros(n, np.int8)
            ev[sm < -h] = -1
            ev[sm > h] = 1
            ar = np.arange(n)
            last = np.maximum.accumulate(np.where(ev != 0, ar, 0))
            state = ev[last]
            trig = np.nonzero((state[1:] == 1) & (state[:-1] == -1))[0] + 1
            if len(trig):
                lastneg = np.maximum.accumulate(np.where(sm < 0, ar, 0))
                cands = []
                for t in trig[-12:]:                    # the newest few triggers, as fractional zero crossings
                    i = int(lastneg[t])
                    if i + 1 < n:
                        d = sm[i + 1] - sm[i]
                        cands.append(i + (-sm[i] / d if d > 0 else 0.0))
                if cands:
                    start = cands[-1]
                    # waves with several crossings per cycle (FM, sync): keep the one that matches the last picture
                    ref = self._scope_ref
                    if ref is not None and len(cands) > 1 and len(ref) == 128:
                        grid = np.arange(128) * (span / 128)
                        best = None
                        for c in cands:
                            seg_v = np.interp(c + grid, ar, mono[:n])
                            err = float(((seg_v - ref) ** 2).sum())
                            if best is None or err < best[0] * 0.98:
                                best = (err, c)
                        start = best[1]
        grid = np.arange(128) * (span / 128)
        self._scope_ref = np.interp(start + grid, np.arange(len(mono)), mono) if len(mono) else None
        x0, x1 = 2, 254
        ncol = x1 - x0
        lanes = [(0, top, bot)] if mode == 1 else [(0, top, (top + bot) // 2 - 1), (1, (top + bot) // 2 + 1, bot)]
        if mode == 1:
            lanes = [(0, top, bot), (1, top, bot)]
        for ch, ly0, ly1 in lanes:
            cy, amp = (ly0 + ly1) // 2, (ly1 - ly0) // 2 - 1
            if ch == 0 or mode == 0:
                L.hline(x0, cy, ncol, dotted=True)
                for k in range(1, 8):
                    L.vline(x0 + k * ncol // 8, ly0, 3)
                    L.vline(x0 + k * ncol // 8, ly1 - 3, 3)
            if mode == 0:
                L.text(x0, ly0 + 1, "LR"[ch])
            if len(data) < span + 2:
                continue
            # resample the window from the fractional start: 4 points per column, then min..max per column
            sub = 4 if span > ncol else 1
            pos = start + np.arange(ncol * sub) * (span / (ncol * sub))
            v = np.clip(np.interp(pos, np.arange(len(data)), data[:, ch]) * gain, -1, 1)
            if sub > 1:
                blk = v.reshape(ncol, sub)
                lo, hi = blk.min(axis=1), blk.max(axis=1)
                tops = np.rint(cy - hi * amp).astype(int)
                bots = np.rint(cy - lo * amp).astype(int)
                nxt_t = np.append(tops[1:], tops[-1])
                nxt_b = np.append(bots[1:], bots[-1])
                L.columns(x0, np.minimum(tops, nxt_b), np.maximum(bots, nxt_t))
            else:
                self.curve(x0, cy - v * amp)

    def scope_button(self, btn):
        u = self.ui
        if btn in ("left", "right"):
            u["scope_span"] = clamp(u["scope_span"] + (1 if btn == "right" else -1), 0, len(SCOPE_SPANS) - 1)
        elif btn in ("up", "down"):
            u["scope_gain"] = clamp(u["scope_gain"] + (1 if btn == "up" else -1), 0, len(SCOPE_GAINS) - 1)
        elif btn == "a":
            u["scope_src"] = 1 - u["scope_src"]
        elif btn == "b":
            u["scope_mode"] = (u["scope_mode"] + 1) % len(SCOPE_MODES)
        else:
            return False
        self._scope_ref = None
        self.save_settings()
        return True

    def draw(self):
        self.lcd.clear()
        if self.scope_on:
            self.draw_scope()
            self.draw_overlays()
            self.present()
            return
        self.draw_header()
        {"CHAIN": self.draw_chain, "DRV": self.draw_drv, "MOD": self.draw_mod, "DLY": self.draw_dly, "REV": self.draw_rev,
         "EQ": self.draw_eq, "I/O": self.draw_io, "SET": self.draw_set}[PAGES[self.page]]()
        self.draw_overlays()
        self.present()

    # ---------- main loop
    def run(self):
        clock = pygame.time.Clock()
        fps = self.args.fps or (12 if ON_DEVICE else 30)
        last_draw = 0.0
        while self.running:
            n_held = len(self.held)
            self.handle_events()
            self.repeat_buttons()
            self.handle_midi()
            self.check_card()
            self.push()
            self.engine.refresh()
            now = time.time()
            if self.presets_dirty and now - self.presets_dirty > 2.0:
                self.save_presets()
            if self.dirty or n_held != len(self.held) or now - last_draw >= 1 / (fps * (1.5 if self.scope_on else 1)):
                self.draw()
                self.dirty = False
                last_draw = now
            if not STARTED.is_set():
                STARTED.set()
                faulthandler.cancel_dump_traceback_later()
                print("startup: first frame drawn", flush=True)
            clock.tick(60)
        self.shutdown()

    def shutdown(self):
        self.save_presets()
        self.save_settings()
        try:
            pygame.display.quit()
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(0)


JOY_BUTTONS = {0: "a", 1: "b", 2: "x", 3: "y", 4: "l1", 5: "r1", 6: "select", 7: "start", 9: "l2", 10: "r2"}
VOLUME_BUTTONS = {0: "vol_down", 1: "vol_up"}
STARTED = threading.Event()
AUDIO_BACKEND = "auto"          # auto: native ALSA threads on the Brick, SDL test sound elsewhere; "alsa" forces ALSA
ALSA_DEVICE = "default"
RAW_JOYSTICK = None


def load_config():
    global SWAP_AB, JOY_BUTTONS, AUDIO_BACKEND, ALSA_DEVICE, RAW_JOYSTICK
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        SWAP_AB = bool(cfg.get("swap_ab", SWAP_AB))
        AUDIO_BACKEND = cfg.get("audio_backend", AUDIO_BACKEND)
        ALSA_DEVICE = cfg.get("alsa_device", ALSA_DEVICE)
        if "raw_joystick" in cfg:
            RAW_JOYSTICK = bool(cfg["raw_joystick"])
        jb = cfg.get("joy_buttons")
        if isinstance(jb, dict):
            JOY_BUTTONS = {int(k): v for k, v in jb.items()}
        for k, v in VOLUME_BUTTONS.items():
            JOY_BUTTONS.setdefault(k, v)
        print("config:", cfg, flush=True)
    except Exception as e:
        print("controls.json ignored:", e, flush=True)


def main():
    print(f"BrickFX {VERSION} starting: python {sys.version.split()[0]}, pygame {pygame.version.ver}, "
          f"numpy {np.__version__}, device={ON_DEVICE}", flush=True)
    load_config()
    ap = argparse.ArgumentParser(description="BrickFX")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale on desktop, e.g. 0.8")
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--debug-input", action="store_true", help="show button numbers as you press them")
    args = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, exit=True)
    App(args).run()


if __name__ == "__main__":
    main()
