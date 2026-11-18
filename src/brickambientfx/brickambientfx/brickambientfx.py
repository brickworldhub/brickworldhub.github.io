#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Alessandro Pietrosanti (BrickWorld, brickworldhub.com)
"""
BrickAmbientFX (Sarta Edition) - an ambient stereo multi-effect for the TrimUI Brick (KNULLI).
Built on BrickFX 0.2.2. Idea from the Zoom Neurig: compressor, amp + cab, modulation, delay and reverb,
serial or split (parallel) routing with stereo spread, stereo out, presets. Delay, reverb and modulation
go further for ambient: shimmer, cloud and bloom reverbs; reverse, dual, shimmer and diffused delays;
ensemble, flanger and glitch; FREEZE on delay and reverb.

  USB audio device (synth, audio interface...) --USB--> Brick --headphone jack--> audio interface

The audio path is native C (native/bafx.c): a capture thread reads the USB card, a writer thread plays on
the Brick's jack, and an adaptive resampler between them absorbs the clock difference of the two cards.
This file is only the screen, buttons, MIDI and presets. Same look and controls as BrickSampler and BrickFX.

  pip install pygame numpy
  python3 brickambientfx.py     # desktop test: a built-in test sound plays through the chain
"""
import os, sys, time, json, glob, math, argparse, select, threading, re, faulthandler, shutil, subprocess

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
        for i, line in enumerate(["BrickAmbientFX can't start", ""] + [msg[j:j + 60] for j in range(0, len(msg), 60)]
                                 + ["", "Details in brickambientfx/log.txt. Closing in 10 s."]):
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

APP_NAME = "BRICKAMBIENTFX"
EDITION = "SARTA EDITION"
VERSION = "1.0.2"
SR = 48000
BLOCK = 128                     # native processing block: 2.7 ms
SCREEN_W, SCREEN_H = 1024, 768
SWAP_AB = False
ON_DEVICE = os.path.isdir("/userdata/roms")
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "controls.json")
SETTINGS_FILE = os.path.join(HERE, "settings.json")
PRESETS_FILE = os.path.join(HERE, "presets.json")
NUM_PRESETS = 16
ASOUND_CARDS = "/proc/asound/cards"


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ----------------------------------------------------------------------------- parameters

def f_pct(v): return f"{v * 100:.0f}%"
def f_time(v): return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"
def f_ms(v): return f"{v * 1000:.1f} ms" if v < 0.01 else f"{v * 1000:.0f} ms"
def f_gain(v): return "-inf dB" if v <= 0.0001 else f"{20 * math.log10(v):+.1f} dB"
def f_db(v): return f"{v:+.1f} dB" if abs(v) < 10 else f"{v:+.0f} dB"
def f_bpm(v): return f"{v:.0f} BPM"
def f_hicut(v): return "Off" if v >= 19000 else (f"{v:.0f} Hz" if v < 1000 else f"{v / 1000:.1f} kHz")
def f_locut(v): return "Off" if v <= 21 else f"{v:.0f} Hz"
def f_rate(v): return f"{v:.2f} Hz"
def f_ratio(v): return f"{v:.1f}:1" if v < 10 else f"{v:.0f}:1"
def f_tone(v): return f"{(v - 0.5) * 24:+.0f} dB"
def f_bal(v): return "Even" if abs(v - 0.5) < 0.01 else (f"Dly +{(0.5 - v) * 200:.0f}%" if v < 0.5 else f"Rev +{(v - 0.5) * 200:.0f}%")


class Param:
    """label: a string, or {type index: label} with the section's type key in `by` (labels that follow the type)."""

    def __init__(self, key, label, kind, lo=0.0, hi=1.0, step=0.01, fmt=f_pct, opts=None, scope="fx", by=None):
        self.key, self.label, self.kind, self.lo, self.hi = key, label, kind, lo, hi
        self.step, self.fmt, self.opts, self.scope, self.by = step, fmt, opts, scope, by

    def lab(self, p=None):
        if isinstance(self.label, dict):
            t = int(p[self.by]) if p is not None and self.by else 0
            return self.label.get(t, self.label.get(0, self.key))
        return self.label


OFFON = ["Off", "On"]
# note values: (label, length in quarter notes)
SYNC = [("Off", 0), ("1/1", 4), ("1/2", 2), ("1/2.", 3), ("1/2T", 4 / 3), ("1/4", 1), ("1/4.", 1.5),
        ("1/4T", 2 / 3), ("1/8", 0.5), ("1/8.", 0.75), ("1/8T", 1 / 3), ("1/16", 0.25), ("1/16.", 0.375),
        ("1/16T", 1 / 6), ("1/32", 0.125)]
MODES = ["Stereo", "Ping-pong", "Mono"]
MAX_TIME = 4.0
REV_MAX_TIME = 2.6              # reverse delay window

P_TEMPO = Param("tempo", "Tempo", "lin", 40, 300, 1, f_bpm)

# ---- COMP
CMP_TYPES = ["Off", "Studio", "Opto", "Limiter", "Swell"]
P_CMP_TYPE = Param("cmp_type", "Type", "enum", opts=CMP_TYPES)
P_CMP_THR = Param("cmp_thr", "Threshold", "lin", -60, 0, 1, lambda v: f"{v:.0f} dB")
P_CMP_RATIO = Param("cmp_ratio", "Ratio", "exp", 1, 20, 0.06, f_ratio)
P_CMP_ATK = Param("cmp_atk", "Attack", "exp", 0.0001, 0.2, 0.08, f_ms)
P_CMP_REL = Param("cmp_rel", {0: "Release", 4: "Reset"}, "exp", 0.02, 3.0, 0.06, f_time, by="cmp_type")
P_CMP_MAKEUP = Param("cmp_makeup", "Makeup", "lin", 0, 24, 0.5, lambda v: f"+{v:.1f} dB")
P_CMP_MIX = Param("cmp_mix", "Mix", "lin", 0, 1, 0.02)
P_CMP_SWELL = Param("cmp_swell", "Swell time", "exp", 0.05, 5.0, 0.06, f_time)
COMP_PARAMS = [P_CMP_TYPE, P_CMP_THR, P_CMP_RATIO, P_CMP_ATK, P_CMP_REL, P_CMP_MAKEUP, P_CMP_MIX, P_CMP_SWELL]

# ---- AMP + CAB
AMP_TYPES = ["Off", "Clean", "Crunch", "Lead", "Fuzz", "Lo-fi"]
CAB_TYPES = ["Off", "1x12", "2x12", "4x12", "Radio"]
P_AMP_TYPE = Param("amp_type", "Type", "enum", opts=AMP_TYPES)
P_AMP_GAIN = Param("amp_gain", {0: "Gain", 5: "Crush"}, "lin", 0, 1, 0.02, by="amp_type")
P_AMP_BASS = Param("amp_bass", "Bass", "lin", 0, 1, 1 / 48, f_tone)
P_AMP_MID = Param("amp_mid", "Middle", "lin", 0, 1, 1 / 48, f_tone)
P_AMP_TREBLE = Param("amp_treble", "Treble", "lin", 0, 1, 1 / 48, f_tone)
P_AMP_CAB = Param("amp_cab", "Cabinet", "enum", opts=CAB_TYPES)
P_AMP_LEVEL = Param("amp_level", "Level", "lin", 0, 1.5, 0.02, f_gain)
AMP_PARAMS = [P_AMP_TYPE, P_AMP_GAIN, P_AMP_BASS, P_AMP_MID, P_AMP_TREBLE, P_AMP_CAB, P_AMP_LEVEL]

# ---- MOD
MOD_TYPES = ["Off", "Chorus", "Ensemble", "Flanger", "Phaser", "Tremolo", "Auto-pan", "Vibrato", "Glitch"]
P_MOD_TYPE = Param("mod_type", "Type", "enum", opts=MOD_TYPES)
P_MOD_RATE = Param("mod_rate", {0: "Rate", 8: "Slices"}, "exp", 0.05, 20.0, 0.05, f_rate, by="mod_type")
P_MOD_SYNC = Param("mod_sync", "Sync", "enum", opts=[s for s, _ in SYNC])
P_MOD_DEPTH = Param("mod_depth", {0: "Depth", 8: "Chance"}, "lin", 0, 1, 0.02, by="mod_type")
P_MOD_COLOR = Param("mod_color", {0: "Color", 1: "Spread", 2: "Spread", 3: "Feedback", 4: "Feedback", 5: "Shape",
                                  6: "Shape", 7: "Drift", 8: "Variety"}, "lin", 0, 1, 0.02, by="mod_type")
P_MOD_MIX = Param("mod_mix", "Mix", "lin", 0, 1, 0.02)
MOD_PARAMS = [P_MOD_TYPE, P_MOD_RATE, P_MOD_SYNC, P_MOD_DEPTH, P_MOD_COLOR, P_MOD_MIX]

# ---- DELAY
DLY_TYPES = ["Off", "Digital", "Tape", "Analog", "Dual", "Reverse", "Shimmer"]
DUAL_RATIOS = ["1/1", "7/8", "3/4", "2/3", "1/2", "3/8", "1/3", "1/4"]
DUAL_RATIO_V = [1.0, 0.875, 0.75, 2 / 3, 0.5, 0.375, 1 / 3, 0.25]
P_DLY_TYPE = Param("dly_type", "Type", "enum", opts=DLY_TYPES)
P_TIME = Param("dly_time", "Time", "exp", 0.02, MAX_TIME, 0.03, f_time)
P_SYNC = Param("dly_sync", "Sync", "enum", opts=[s for s, _ in SYNC])
P_FB = Param("dly_fb", "Feedback", "lin", 0, 1, 0.02)
P_MIX = Param("dly_mix", "Mix", "lin", 0, 1, 0.02)
P_MODE = Param("dly_mode", "Mode", "enum", opts=MODES)
P_WIDTH = Param("dly_width", "Width", "lin", 0, 1, 0.05)
P_HICUT = Param("dly_hicut", "Hi cut", "exp", 1000, 20000, 0.05, f_hicut)
P_LOCUT = Param("dly_locut", "Lo cut", "exp", 20, 1000, 0.06, f_locut)
P_MODD = Param("dly_moddepth", "Mod depth", "lin", 0, 1, 0.05)
P_MODR = Param("dly_modrate", "Mod rate", "exp", 0.1, 5.0, 0.06, f_rate)
P_DLY_COLOR = Param("dly_color", {0: "Color", 1: "Diffuse", 2: "Age", 3: "Age", 4: "R ratio", 5: "Diffuse", 6: "Octave"},
                    "lin", 0, 1, 0.02, by="dly_type")
DELAY_PARAMS = [P_DLY_TYPE, P_TIME, P_TEMPO, P_SYNC, P_FB, P_MIX, P_MODE, P_WIDTH, P_HICUT, P_LOCUT, P_MODD, P_MODR,
                P_DLY_COLOR]

# ---- REVERB
REV_TYPES = ["Off", "Room", "Plate", "Hall", "Cloud", "Shimmer", "Bloom"]
REV_SIZE = [1.0, 0.55, 1.0, 1.7, 2.6, 2.0, 2.2]
P_REV_TYPE = Param("rev_type", "Type", "enum", opts=REV_TYPES)
P_REV_DECAY = Param("rev_decay", "Decay", "lin", 0, 1, 0.02)
P_REV_PRE = Param("rev_pre", "Pre-delay", "lin", 0, 0.3, 0.005, f_time)
P_REV_DAMP = Param("rev_damp", "Damping", "lin", 0, 1, 0.02)
P_REV_LOCUT = Param("rev_locut", "Lo cut", "exp", 20, 800, 0.06, f_locut)
P_REV_MOD = Param("rev_mod", "Mod", "lin", 0, 1, 0.05)
P_REV_COLOR = Param("rev_color", {0: "Diffuse", 5: "Shimmer", 6: "Bloom"}, "lin", 0, 1, 0.02, by="rev_type")
P_REV_MIX = Param("rev_mix", "Mix", "lin", 0, 1, 0.02)
REVERB_PARAMS = [P_REV_TYPE, P_REV_DECAY, P_REV_PRE, P_REV_DAMP, P_REV_LOCUT, P_REV_MOD, P_REV_COLOR, P_REV_MIX]

# ---- routing / global (stored in the preset)
ROUTINGS = ["Serial", "Split"]
P_ROUTING = Param("routing", "Routing", "enum", opts=ROUTINGS)
P_BALANCE = Param("balance", "Balance", "lin", 0, 1, 0.02, f_bal)
P_SPREAD = Param("spread", "Spread", "lin", 0, 1, 0.02)
P_OWIDTH = Param("owidth", "Width", "lin", 0, 2, 0.05)
P_FREEZE = Param("freeze", "Freeze", "enum", opts=OFFON)
GLOBAL_PARAMS = [P_ROUTING, P_BALANCE, P_SPREAD, P_OWIDTH, P_FREEZE, P_TEMPO]

# the five sections, in signal order: (name, type key, type names)
SECTIONS = [("COMP", "cmp_type", CMP_TYPES), ("AMP", "amp_type", AMP_TYPES), ("MOD", "mod_type", MOD_TYPES),
            ("DELAY", "dly_type", DLY_TYPES), ("REVERB", "rev_type", REV_TYPES)]
CHAIN_PARAMS = [Param(k, n.capitalize(), "enum", opts=o) for n, k, o in SECTIONS]

PRESETS_VERSION = 1
PRESET_DEFAULTS = dict(on=1, tempo=90.0, routing=0, balance=0.5, spread=0.0, owidth=1.0, freeze=0,
                       cmp_type=0, cmp_thr=-24.0, cmp_ratio=3.0, cmp_atk=0.01, cmp_rel=0.25, cmp_makeup=3.0,
                       cmp_mix=1.0, cmp_swell=0.6,
                       amp_type=0, amp_gain=0.3, amp_bass=0.5, amp_mid=0.5, amp_treble=0.5, amp_cab=0, amp_level=1.0,
                       mod_type=0, mod_rate=0.5, mod_sync=0, mod_depth=0.5, mod_color=0.5, mod_mix=0.5,
                       dly_type=1, dly_time=0.48, dly_sync=0, dly_fb=0.45, dly_mix=0.3, dly_mode=1, dly_width=1.0,
                       dly_hicut=6000.0, dly_locut=120.0, dly_moddepth=0.2, dly_modrate=0.5, dly_color=0.3,
                       rev_type=3, rev_decay=0.7, rev_pre=0.03, rev_damp=0.45, rev_locut=120.0, rev_mod=0.4,
                       rev_color=0.8, rev_mix=0.35)


def _pr(**kw):
    return dict(PRESET_DEFAULTS, **kw)


FACTORY = [
    # 1 the Sarta sound: gentle opto, slow chorus, diffused echoes into a cloud
    _pr(cmp_type=2, cmp_thr=-26.0, cmp_ratio=2.5, mod_type=1, mod_rate=0.2, mod_depth=0.6, mod_color=0.7, mod_mix=0.45,
        dly_type=1, dly_time=0.52, dly_fb=0.55, dly_mix=0.3, dly_color=0.6, rev_type=4, rev_decay=0.85, rev_mix=0.45),
    # 2 shimmer reverb, dotted 1/8 echoes, ensemble
    _pr(mod_type=2, mod_rate=0.3, mod_depth=0.5, mod_color=0.6, mod_mix=0.4, dly_sync=9, dly_fb=0.45, dly_mix=0.25,
        rev_type=5, rev_decay=0.82, rev_color=0.6, rev_mix=0.5, rev_locut=200.0),
    # 3 worn tape, warbling vibrato, hall
    _pr(amp_type=1, amp_gain=0.25, mod_type=7, mod_rate=0.7, mod_depth=0.35, mod_color=0.55, mod_mix=1.0,
        dly_type=2, dly_time=0.55, dly_fb=0.6, dly_mix=0.35, dly_mode=0, dly_color=0.6, dly_hicut=4500.0,
        rev_type=3, rev_decay=0.7, rev_mix=0.35),
    # 4 volume swells into reverse echoes and a long hall
    _pr(cmp_type=4, cmp_thr=-36.0, cmp_swell=0.9, cmp_rel=0.3, cmp_makeup=0.0, dly_type=5, dly_time=0.8, dly_fb=0.4,
        dly_mix=0.35, dly_color=0.3, rev_type=3, rev_decay=0.85, rev_mix=0.45),
    # 5 reverse rain on a plate
    _pr(dly_type=5, dly_time=0.6, dly_fb=0.45, dly_mix=0.45, dly_mode=0, dly_color=0.0, rev_type=2, rev_decay=0.6,
        rev_mix=0.3),
    # 6 slow blooming pad
    _pr(cmp_type=1, cmp_thr=-24.0, cmp_ratio=3.0, mod_type=2, mod_rate=0.25, mod_depth=0.6, mod_color=0.8, mod_mix=0.5,
        dly_type=0, rev_type=6, rev_decay=0.88, rev_color=0.75, rev_mix=0.55, rev_pre=0.06),
    # 7 glitch garden: 1/16 slices, dual echoes split from a plate
    _pr(tempo=96.0, mod_type=8, mod_sync=11, mod_depth=0.3, mod_color=0.75, mod_mix=1.0, routing=1, spread=0.4,
        dly_type=4, dly_sync=5, dly_fb=0.4, dly_mix=0.4, dly_color=0.3, rev_type=2, rev_decay=0.6, rev_mix=0.35),
    # 8 dual echoes (2/3), split from the hall, spread wide
    _pr(routing=1, spread=0.6, dly_type=4, dly_time=0.45, dly_fb=0.5, dly_mix=0.4, dly_color=0.4, rev_type=3,
        rev_decay=0.75, rev_mix=0.4),
    # 9 slow jet flanger, tape echo, hall
    _pr(mod_type=3, mod_rate=0.12, mod_depth=0.8, mod_color=0.65, mod_mix=0.5, dly_type=2, dly_time=0.42, dly_fb=0.4,
        dly_mix=0.25, rev_type=3, rev_decay=0.6, rev_mix=0.3),
    # 10 tremolo waves into dark analog echoes
    _pr(mod_type=5, mod_rate=3.5, mod_depth=0.6, mod_color=0.25, mod_mix=1.0, dly_type=3, dly_time=0.5, dly_fb=0.55,
        dly_mix=0.35, dly_color=0.4, rev_type=1, rev_decay=0.5, rev_mix=0.25),
    # 11 lo-fi radio, drifting
    _pr(amp_type=5, amp_gain=0.35, amp_cab=4, mod_type=7, mod_rate=0.5, mod_depth=0.5, mod_color=0.8, mod_mix=1.0,
        dly_type=2, dly_time=0.38, dly_fb=0.5, dly_mix=0.35, dly_color=0.85, dly_mode=2, rev_type=1, rev_decay=0.45,
        rev_mix=0.25),
    # 12 crunch amp in a big space
    _pr(cmp_type=1, cmp_thr=-20.0, cmp_ratio=4.0, amp_type=2, amp_gain=0.45, amp_cab=3, amp_treble=0.55,
        dly_type=3, dly_time=0.42, dly_fb=0.4, dly_mix=0.3, rev_type=3, rev_decay=0.7, rev_mix=0.35),
    # 13 lead into climbing octave echoes and a cloud
    _pr(cmp_type=1, amp_type=3, amp_gain=0.4, amp_cab=2, amp_mid=0.6, amp_level=0.7, dly_type=6, dly_sync=5, dly_fb=0.55, dly_mix=0.35,
        dly_color=0.5, rev_type=4, rev_decay=0.8, rev_mix=0.4),
    # 14 slow phaser, diffused echoes, plate
    _pr(mod_type=4, mod_rate=0.12, mod_depth=0.8, mod_color=0.6, mod_mix=0.5, dly_type=1, dly_time=0.6, dly_fb=0.5,
        dly_mix=0.3, dly_color=0.8, rev_type=2, rev_decay=0.65, rev_mix=0.3),
    # 15 split world: ping-pong left, cloud right, auto-pan
    _pr(routing=1, spread=1.0, owidth=1.3, mod_type=6, mod_rate=0.2, mod_depth=0.5, mod_mix=1.0, dly_type=1,
        dly_time=0.36, dly_fb=0.5, dly_mix=0.45, rev_type=4, rev_decay=0.85, rev_mix=0.5),
    # 16 clean amp, chorus, small room
    _pr(cmp_type=1, cmp_thr=-18.0, cmp_ratio=3.0, amp_type=1, amp_gain=0.2, amp_cab=1, mod_type=1, mod_rate=0.6,
        mod_depth=0.4, mod_mix=0.4, dly_type=0, rev_type=1, rev_decay=0.45, rev_mix=0.25),
]
PRESET_NAMES = ["SARTA CLOUD", "SHIMMER SKY", "TAPE DREAMS", "SWELL HALL", "REVERSE RAIN", "BLOOM PAD",
                "GLITCH GARDEN", "DUAL ECHOES", "SLOW JET", "TREMOLO WAVES", "LO-FI RADIO", "CRUNCH SPACE",
                "OCTAVE CLIMB", "PHASE DRIFT", "SPLIT WORLD", "CLEAN ROOM"]

ROUTES = ["1+2 Stereo", "1 to both", "2 to both", "Mono sum"]
LATENCY_NAMES = ["Low", "Mid", "Safe"]
LATENCY_FRAMES = [384, 768, 1536]          # output buffer: 8 / 16 / 32 ms
P_SOURCE = Param("source", "Source", "source", scope="ui")
P_USBIN = Param("usbin", "USB input", "usbin", scope="ui")
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

# What a MIDI CC can move. "on" switches the effect, "tap" is a tap-tempo footswitch, "freeze" holds the tails.
CC_TARGETS = {p.key: p for p in COMP_PARAMS + AMP_PARAMS + MOD_PARAMS + DELAY_PARAMS + REVERB_PARAMS + GLOBAL_PARAMS}
CC_TARGETS["on"] = Param("on", "On/bypass", "switch")
CC_TARGETS["tap"] = Param("tap", "Tap", "tap")
CC_TARGETS["freeze"] = Param("freeze", "Freeze", "switch")
FIXED_CCS = {120, 121, 123}


def cc_label(key, p=None):
    prm = CC_TARGETS.get(key)
    if prm is None:
        return key.upper()
    sec = {"cmp": "COMP ", "amp": "AMP ", "mod": "MOD ", "dly": "DELAY ", "rev": "REVERB "}.get(key.split("_")[0], "")
    return sec + prm.lab(p).upper()


# Order of P_* in bafx.c
ENGINE_KEYS = ["on", "ingain", "outgain", "route", "trails", "freeze", "routing", "balance", "spread", "owidth",
               "cmp_type", "cmp_thr", "cmp_ratio", "cmp_atk", "cmp_rel", "cmp_makeup", "cmp_mix", "cmp_swell",
               "amp_type", "amp_gain", "amp_bass", "amp_mid", "amp_treble", "amp_cab", "amp_level",
               "mod_type", "mod_rate", "mod_depth", "mod_color", "mod_mix",
               "dly_type", "dly_time", "dly_fb", "dly_mix", "dly_mode", "dly_width", "dly_hicut", "dly_locut",
               "dly_moddepth", "dly_modrate", "dly_color",
               "rev_type", "rev_decay", "rev_pre", "rev_damp", "rev_locut", "rev_mod", "rev_color", "rev_mix"]
STATUS_KEYS = ["in_l", "in_r", "out_l", "out_r", "load", "underruns", "cap_xruns", "rs_under", "fill", "ppm",
               "cap_open", "play_open", "rt", "cap_period", "play_buffer", "cur_time", "blocks", "cap_rate",
               "target", "priming", "cap_ch", "rs_over", "play_period", "rt_cap", "cap_frames", "cap_reads", "cap_raw",
               "cap_buffer", "cap_opens", "wpos", "nan_resets", "gr", "glitch_ev", "glitch_count", "frz", "tank_pk",
               "tank_clip", "dly_tail", "rev_tail", "cap_fmt", "cap_max_l", "cap_max_r"]
GLITCH_NAMES = ["", "REPEAT", "REVERSE", "OCTAVE", "STOP", "DROP"]


def delay_seconds(p, bpm=None):
    """Delay time actually used: free time, or the note value at the tempo (preset tempo or MIDI clock)."""
    s = int(p["dly_sync"])
    t = float(p["dly_time"]) if s == 0 else 60.0 / (bpm or p["tempo"]) * SYNC[s][1]
    top = REV_MAX_TIME if int(p["dly_type"]) == 5 else MAX_TIME
    while t > top:                      # too long for the line: drop an octave rather than clip
        t /= 2
    return t


def mod_rate_hz(p, bpm=None):
    """Mod LFO rate (glitch: slices per second): free, or one cycle per note value at the tempo."""
    s = int(p["mod_sync"])
    if s == 0:
        return float(p["mod_rate"])
    return (bpm or p["tempo"]) / 60.0 / SYNC[s][1]


def dual_ratio(p):
    return int(clamp(p["dly_color"] * 7.99, 0, 7))


# ---- small RBJ biquad helpers for the AMP page picture (the same formulas as bafx.c)
def _bq(kind, fc, q, db, f):
    w0 = 2 * math.pi * fc / SR
    c, s = math.cos(w0), math.sin(w0)
    al, A = s / (2 * q), 10 ** (db / 40)
    if kind == "hp":
        b, a = [(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al]
    elif kind == "lp":
        b, a = [(1 - c) / 2, 1 - c, (1 - c) / 2], [1 + al, -2 * c, 1 - al]
    elif kind == "peak":
        b, a = [1 + al * A, -2 * c, 1 - al * A], [1 + al / A, -2 * c, 1 - al / A]
    elif kind == "ls":
        sq = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    else:
        sq = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    z = np.exp(-1j * 2 * np.pi * f / SR)
    h = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-6))


CAB_FILTERS = [[], [("hp", 90, .7, 0), ("peak", 2200, 1, 3), ("lp", 5200, .9, 0), ("lp", 7500, .6, 0)],
               [("hp", 80, .7, 0), ("peak", 120, 1.2, 2.5), ("peak", 2500, 1, 2), ("lp", 5500, .8, 0)],
               [("hp", 70, .7, 0), ("peak", 110, 1.3, 4), ("peak", 400, 1, -3), ("lp", 4500, .9, 0)],
               [("hp", 400, .7, 0), ("peak", 1500, .8, 4), ("lp", 3000, .9, 0), ("lp", 3500, .7, 0)]]



# ----------------------------------------------------------------------------- native engine

def load_lib():
    import ctypes as C
    import platform
    m = platform.machine().lower()
    name = {"aarch64": "libbafx_aarch64.so", "arm64": "libbafx_aarch64.so", "x86_64": "libbafx_x86_64.so",
            "amd64": "libbafx_x86_64.so"}.get(m)
    if not name or not sys.platform.startswith("linux"):
        _fatal(f"no native engine for {sys.platform}/{m} (BrickAmbientFX needs Linux: the Brick or a Linux PC)")
    lib = C.CDLL(os.path.join(HERE, "native", name))
    vp, ci, cs = C.c_void_p, C.c_int, C.c_char_p
    lib.eng_create.restype, lib.eng_create.argtypes = vp, [ci, ci]
    for fn, args in (("eng_set_params", [vp, vp]), ("eng_process", [vp, vp, vp, ci]), ("eng_clear", [vp]),
                     ("eng_set_capture", [vp, cs]), ("eng_set_buffer", [vp, ci])):
        f = getattr(lib, fn)
        f.restype, f.argtypes = None, args
    lib.eng_status.restype, lib.eng_status.argtypes = ci, [vp, vp, ci]
    lib.eng_start.restype, lib.eng_start.argtypes = ci, [vp, cs, cs, ci]
    lib.eng_cap_format.restype, lib.eng_cap_format.argtypes = None, [vp, ci, ci]
    for fn in ("eng_device", "eng_cap_device", "eng_error", "eng_cap_error"):
        getattr(lib, fn).restype, getattr(lib, fn).argtypes = cs, [vp]
    print(f"dsp: native {name} (version {lib.bafx_version()})", flush=True)
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

    def set_format(self, fmt, rate):
        """Format and rate for the next capture open (the engine falls back to S16_LE by itself)."""
        self.lib.eng_cap_format(self.h, ALSA_FMT.get(fmt, 2), int(rate))

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


def capture_devices(cid):
    """Record devices of a card (the N of DEV=N), from /proc/asound/cardX/pcmNc. Some USB adapters have more
    than one (for example a mic input and a line input on the same card)."""
    num = card_number(cid) if cid else None
    if num is None:
        return []
    devs = []
    for d in glob.glob(os.path.join(os.path.dirname(ASOUND_CARDS), f"card{num}", "pcm*c")):
        m = re.search(r"pcm(\d+)c$", d)
        if m:
            devs.append(int(m.group(1)))
    return sorted(devs)


def capture_info(cid, dev):
    """Formats, channel counts and rates of a USB capture device, from the "Capture:" block of
    /proc/asound/cardN/streamX (USB audio makes one stream file per PCM device). {} when unknown."""
    n = card_number(cid) if cid else None
    if n is None:
        return {}
    try:
        with open(os.path.join(os.path.dirname(ASOUND_CARDS), f"card{n}", f"stream{dev}")) as f:
            text = f.read()
    except OSError:
        return {}
    i = text.find("Capture:")
    if i < 0:
        return {}
    block = text[i:]
    j = block.find("Playback:")
    if j > 0:
        block = block[:j]
    info = {"formats": [], "channels": [], "rates": [], "ranges": []}
    for ln in block.splitlines():
        ln = ln.strip()
        if ln.startswith("Format:"):
            for fm in ln.split(":", 1)[1].split():
                if fm not in info["formats"]:
                    info["formats"].append(fm)
        elif ln.startswith("Channels:"):
            try:
                info["channels"].append(int(ln.split(":", 1)[1]))
            except ValueError:
                pass
        elif ln.startswith("Rates:"):
            v = ln.split(":", 1)[1]
            rng = re.match(r"\s*(\d+)\s*-\s*(\d+)", v)
            if rng:
                info["ranges"].append((int(rng.group(1)), int(rng.group(2))))
            else:
                for r in re.findall(r"\d+", v):
                    if int(r) not in info["rates"]:
                        info["rates"].append(int(r))
    return info


def pick_rate(info):
    """48000 if the device has it, otherwise the first rate it lists; SR when unknown."""
    rates, ranges = info.get("rates", []), info.get("ranges", [])
    if 48000 in rates or any(a <= 48000 <= b for a, b in ranges):
        return 48000
    if rates:
        return rates[0]
    if ranges:
        return ranges[0][1]
    return SR


ALSA_FMT = {"S16_LE": 2, "S24_LE": 6, "S32_LE": 10, "S24_3LE": 32}       # ALSA format codes for the engine
ALSA_FMT_NAME = {v: k for k, v in ALSA_FMT.items()}


def pick_format(info):
    """The device's own best capture format (some USB inputs only send real audio in their 24-bit mode)."""
    fm = info.get("formats", [])
    for f in ("S24_3LE", "S32_LE", "S24_LE", "S16_LE"):
        if f in fm:
            return f
    return "S16_LE"


def usb_mixer_setup(cid):
    """Log the USB card's mixer and switch on any capture switch that is off, or a capture volume left at 0.
    (A computer often does this by itself; on the Brick nothing does, and the input then sends silence.)
    Other controls are not touched."""
    exe = shutil.which("amixer")
    n = card_number(cid)
    if not exe or n is None:
        if n is not None:
            print(f"audio: mixer card {n}: amixer not found, not checked", flush=True)
        return
    try:
        out = subprocess.run([exe, "-c", str(n), "scontents"], capture_output=True, text=True, timeout=3).stdout
    except Exception as e:
        print(f"audio: amixer -c {n}: {e}", flush=True)
        return
    print(f"audio: mixer card {n} (amixer -c {n} scontents):\n" + "\n".join("  " + ln for ln in out.splitlines()), flush=True)
    blocks = re.split(r"(?m)^Simple mixer control ", out)
    for blk in blocks[1:]:
        lines = blk.splitlines()
        m = re.match(r"'(.+)',(\d+)", lines[0])
        if not m:
            continue
        name = f"{m.group(1)},{m.group(2)}"
        caps = next((ln.split(":", 1)[1].split() for ln in lines if "Capabilities:" in ln), [])
        vals = [ln.strip() for ln in lines if "Capture" in ln and "[" in ln and "Limits" not in ln]
        todo = []
        if "cswitch" in caps and any("[off]" in v for v in vals):
            todo.append(["cap"])
        if any(c.startswith("cvolume") for c in caps) and vals and all("[0%]" in v for v in vals):
            # unity gain (0 dB) when the control has a dB scale: no extra analogue gain, so no extra noise
            todo.append(["capture", "0dB"] if any("dB]" in v for v in vals) else ["capture", "80%"])
        for args in todo:
            r = subprocess.run([exe, "-c", str(n), "-q", "sset", name] + args, capture_output=True, text=True, timeout=3)
            if r.returncode != 0 and args[-1] == "0dB":
                args = ["capture", "80%"]
                r = subprocess.run([exe, "-c", str(n), "-q", "sset", name] + args, capture_output=True, text=True,
                                   timeout=3)
            print(f"audio: mixer card {n} '{name}' {' '.join(args)}: " + ("ok" if r.returncode == 0 else r.stderr.strip()),
                  flush=True)


def device_hw_params(cid, dev):
    """What the hardware really runs at while the device is open (format, channels, rate, period, buffer)."""
    num = card_number(cid) if cid else None
    if num is None:
        return {}
    try:
        with open(os.path.join(os.path.dirname(ASOUND_CARDS), f"card{num}", f"pcm{dev}c", "sub0", "hw_params")) as f:
            return dict(ln.split(":", 1) for ln in f.read().splitlines() if ":" in ln)
    except (OSError, ValueError):
        return {}


def opened_dev(name):
    """DEV number from an ALSA name the engine opened (plughw:CARD=x,DEV=1 or plughw:1,1)."""
    m = re.search(r"DEV=(\d+)", name or "") or re.search(r"hw:\d+,(\d+)", name or "")
    return int(m.group(1)) if m else None


class DesktopAudio:
    """Desktop test: a built-in stereo test phrase goes through the chain and out of SDL."""

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


PAGES = ["I/O", "FX", "CMP", "AMP", "MOD", "DLY", "REV", "SET"]
FX_PAGES = ("FX", "CMP", "AMP", "MOD", "DLY", "REV")
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
        pygame.display.set_caption("BrickAmbientFX - Sarta Edition")
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
        self.cur_dev = 0
        self.mixer_done = set()
        self.audio = None
        self.last_cap_change = time.time()
        if ON_DEVICE or AUDIO_BACKEND == "alsa":
            card = self.resolve_card()
            self.cur_card = card
            for cid, cname in capture_cards():
                print(f"audio: card {cid} ({cname}): record devices {capture_devices(cid)}", flush=True)
            dev = self.want_dev(card)
            self.cur_dev = dev
            self.prepare_input(card, dev)
            print(f"audio: input devices {self.cap_devs(card, dev)}; cards {capture_cards()}", flush=True)
            if not self.engine.start_alsa(self.cap_devs(card, dev), LATENCY_FRAMES[self.ui["latency"]]):
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
        self.learn = None
        self.dirty = True
        self.presets_dirty = 0.0
        self.running = True
        self.last_input = "pad" if ON_DEVICE else "keyboard"
        self.last_card_scan = 0.0
        self.logged_inputs = 0
        self.glitch_hist = []
        self.glitch_seen = 0
        self.key_map = {
            pygame.KSCAN_UP: "up", pygame.KSCAN_DOWN: "down", pygame.KSCAN_LEFT: "left", pygame.KSCAN_RIGHT: "right",
            pygame.KSCAN_Z: "a", pygame.KSCAN_RETURN: "a", pygame.KSCAN_X: "b", pygame.KSCAN_ESCAPE: "b",
            pygame.KSCAN_BACKSPACE: "b", pygame.KSCAN_SPACE: "x", pygame.KSCAN_LSHIFT: "y", pygame.KSCAN_RSHIFT: "y",
            pygame.KSCAN_T: "tap", pygame.KSCAN_F: "freeze", pygame.KSCAN_LEFTBRACKET: "l1",
            pygame.KSCAN_RIGHTBRACKET: "r1", pygame.KSCAN_COMMA: "l2", pygame.KSCAN_PERIOD: "r2",
            pygame.KSCAN_TAB: "select",
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
        L.frame(14, 18, 228, 110)
        L.frame(16, 20, 224, 106)
        L.ctext(128, 28, "BRICK", scale=2)
        L.ctext(128, 46, "AMBIENT FX", scale=3)
        L.ctext(128, 72, EDITION)
        # a reverb tail: dense, then thinning out
        rng = np.random.default_rng(3)
        for k in range(48):
            x = 30 + k * 4
            h = int(max(1, 16 * math.exp(-k / 16) * (0.5 + 0.5 * rng.random())))
            L.fill(x, 116 - h, 2, h)
        L.hline(28, 117, 201, dotted=True)
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
                       trails=1, preset=0, clock=1, ccmap={}, usb_card="", usb_dev=0, usb_devs={})
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
        for k, lo, hi in (("ingain", 0, 4), ("outgain", 0, 1.5), ("volume", 0, 1)):
            u[k] = float(clamp(float(u[k]), lo, hi))
        u["usb_card"] = str(u["usb_card"] or "")
        try:
            u["usb_dev"] = max(0, int(u["usb_dev"]))
        except (TypeError, ValueError):
            u["usb_dev"] = 0
        ud = u["usb_devs"] if isinstance(u["usb_devs"], dict) else {}
        u["usb_devs"] = {str(k): int(v) for k, v in ud.items() if str(v).isdigit()}
        if u["usb_card"] and u["usb_card"] not in u["usb_devs"]:
            u["usb_devs"][u["usb_card"]] = u["usb_dev"]
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
        self.glitch_hist = []
        self.show_toast(f"PRESET {self.ui['preset'] + 1}  {self.preset_name()}")

    def preset_desc(self):
        p = self.p
        on = [opts[int(p[k])].upper() for _, k, opts in SECTIONS if int(p[k])]
        return " + ".join(on) if on else "ALL OFF"

    # ---------- audio
    def resolve_card(self):
        cards = [c for c, _ in capture_cards()]
        src = self.ui["source"]
        if src == "auto":
            return cards[0] if cards else None
        return src if src in cards else None

    def want_dev(self, card, log=True):
        """Record device to open on this card: the one saved for it, else the first. If the saved one is
        gone, back to DEV=0 (written to the log)."""
        devs = capture_devices(card)
        if not devs:
            return 0
        saved = self.ui["usb_devs"].get(card)          # every card keeps its own input
        if saved is None:
            return devs[0]
        if saved in devs:
            return saved
        if log:
            print(f"audio: saved input DEV={saved} not on card {card} (devices {devs}), using DEV={devs[0]}", flush=True)
        return devs[0]

    @staticmethod
    def cap_devs(card, dev=0):
        if card is None:
            return []
        out = []
        num = card_number(card)
        for d in ([dev, 0] if dev else [0]):     # the chosen device, then DEV=0 so the input never stays shut
            out += [f"plughw:CARD={card},DEV={d}", f"hw:CARD={card},DEV={d}"]
            if num is not None:                  # by number too, in case the name doesn't resolve
                out += [f"plughw:{num},{d}", f"hw:{num},{d}"]
        return out

    def prepare_input(self, card, dev):
        """Before opening: switch on the card's capture mixer (once per card) and tell the engine the
        device's own best format and rate."""
        if card is None:
            return
        if card not in self.mixer_done:
            self.mixer_done.add(card)
            usb_mixer_setup(card)
        info = capture_info(card, dev)
        fmt, rate = pick_format(info), pick_rate(info)
        self.engine.set_format(fmt, rate)
        conv = "" if rate == SR else f", converted to {SR} Hz in the engine"
        print(f"audio: input card {card} ({self.card_name(card)}) = {self.src_label(card, True)}, record devices "
              f"{capture_devices(card)}, chosen DEV={dev} = Input {self.in_number(card, dev)}; device formats "
              f"{info.get('formats') or '?'} channels {info.get('channels') or '?'} rates "
              f"{info.get('rates') or info.get('ranges') or '?'}; opening {fmt} {rate} Hz{conv}", flush=True)

    def check_card(self):
        """Hot-plug: follow the USB card as it comes and goes (checked every 2 s)."""
        now = time.time()
        if not self.engine.threaded or now - self.last_card_scan < 2:
            return
        self.last_card_scan = now
        card = self.resolve_card()
        dev = self.want_dev(card, log=card != self.cur_card) if card else 0
        if card != self.cur_card or dev != self.cur_dev:
            print(f"audio: input {self.cur_card}/DEV={self.cur_dev} -> {card}/DEV={dev}  devices {self.cap_devs(card, dev)}",
                  flush=True)
            self.prepare_input(card, dev)
            self.cur_card, self.cur_dev = card, dev
            self.engine.set_capture(self.cap_devs(card, dev))      # the engine fades out, switches, fades in
            self.show_toast(f"Input: {self.src_label(card, True)} IN {self.in_number(card, dev)}" if card
                            else "USB input disconnected", 3)

    def card_name(self, cid):
        if cid is None:
            return "none"
        return dict(capture_cards()).get(cid, cid)

    def src_label(self, cid, numbered=False):
        """How a USB input is shown: USB (USB 1, USB 2... when more than one is connected, or numbered=True).
        Never the brand: the real card name is only written to log.txt."""
        if cid is None:
            return "none"
        ids = [c for c, _ in capture_cards()]
        if cid not in ids:
            return "USB"
        return f"USB {ids.index(cid) + 1}" if numbered or len(ids) > 1 else "USB"

    @staticmethod
    def in_number(card, dev):
        """Input 1, 2, 3... = place of the record device on its card (only devices that exist)."""
        devs = capture_devices(card)
        return devs.index(dev) + 1 if dev in devs else 1

    def bpm(self):
        """MIDI clock tempo when following it and something is synced."""
        if self.ui["clock"] and self.midi.clock_running() and (int(self.p["dly_sync"]) or int(self.p["mod_sync"])):
            return self.midi.clock_bpm
        return None

    def engine_values(self):
        p, u = self.p, self.ui
        v = dict(p)
        b = self.bpm()
        v["dly_time"] = delay_seconds(p, b)
        v["mod_rate"] = mod_rate_hz(p, b)
        v["trails"], v["route"], v["ingain"] = u["trails"], u["route"], u["ingain"]
        v["outgain"] = u["outgain"] * u["volume"] ** 2
        return v

    def push(self):
        self.engine.push(self.engine_values())

    def _audio_log(self):
        """Write what the audio threads are doing to log.txt: every second for the first 20 s, then
        every 5 s when something changed. This is what to read when there is no sound."""
        t0, last, last_frames, last_key = time.time(), 0.0, -1.0, None
        last_opens, level_at = 0, 0.0
        while True:
            time.sleep(1.0)
            e, now = self.engine, time.time()
            buf = np.zeros(len(STATUS_KEYS), np.float32)            # own buffer: the UI thread reads too
            e.lib.eng_status(e.h, buf.ctypes.data, len(buf))
            s = dict(zip(STATUS_KEYS, buf.tolist()))
            if int(s["cap_opens"]) != last_opens and s["cap_open"]:
                last_opens = int(s["cap_opens"])
                name = e.text("eng_cap_device")
                card, dev = self.cur_card, opened_dev(name)
                hw = device_hw_params(card, dev) if dev is not None else {}
                hwtxt = ", ".join(f"{k.strip()} {v.strip()}" for k, v in hw.items()
                                  if k.strip() in ("format", "channels", "rate", "period_size", "buffer_size")) or "not readable"
                print(f"audio: input OPEN card {card} ({self.card_name(card)}) device {name} = DEV={dev} "
                      f"(Input {self.in_number(card, dev) if dev is not None else '?'}"
                      f"{', NOT the chosen DEV=' + str(self.cur_dev) if dev is not None and dev != self.cur_dev else ''}); "
                      f"format {ALSA_FMT_NAME.get(int(s['cap_fmt']), int(s['cap_fmt']))} channels {int(s['cap_ch'])} "
                      f"rate {int(s['cap_rate'])} Hz period {int(s['cap_period'])} buffer {int(s['cap_buffer'])}; "
                      f"hardware: {hwtxt}", flush=True)
                level_at = now + 2.0
            if level_at and now >= level_at:
                level_at = 0.0
                lv = [s["cap_max_l"], s["cap_max_r"]]
                txt = "silent (all 0)" if max(lv) <= 0 else \
                    "L " + " / R ".join("silent (all 0)" if v <= 0 else f"{20 * math.log10(v):.1f} dBFS" for v in lv)
                print(f"audio: input level after 2 s (peak per channel): {txt}", flush=True)
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
                      f"nan_resets={int(s['nan_resets'])} rev_tank_peak={s['tank_pk']:.2f} rev_tank_clips={int(s['tank_clip'])}",
                      flush=True)
                last_key = key

    # ---------- params
    def page_layout(self):
        name = PAGES[self.page]
        p = self.p
        if name == "DLY":
            t = P_TEMPO if int(p["dly_sync"]) else P_TIME
            return [[("DELAY", [P_DLY_TYPE, t, P_SYNC, P_FB, P_MIX, P_MODE])],
                    [("TONE", [P_WIDTH, P_HICUT, P_LOCUT, P_MODD, P_MODR, P_DLY_COLOR])]]
        if name == "FX":
            return [[("CHAIN", CHAIN_PARAMS)], [("ROUTING", GLOBAL_PARAMS)]]
        if name == "CMP":
            t = int(p["cmp_type"])
            if t == 4:
                left = [P_CMP_TYPE, P_CMP_THR, P_CMP_SWELL]
            elif t == 3:
                left = [P_CMP_TYPE, P_CMP_THR, P_CMP_ATK]
            else:
                left = [P_CMP_TYPE, P_CMP_THR, P_CMP_RATIO, P_CMP_ATK]
            return [[("COMP", left)], [("OUTPUT", [P_CMP_REL, P_CMP_MAKEUP, P_CMP_MIX])]]
        if name == "AMP":
            return [[("AMP", [P_AMP_TYPE, P_AMP_GAIN, P_AMP_CAB, P_AMP_LEVEL])],
                    [("TONE", [P_AMP_BASS, P_AMP_MID, P_AMP_TREBLE])]]
        if name == "MOD":
            r = P_TEMPO if int(p["mod_sync"]) else P_MOD_RATE
            return [[("MOD", [P_MOD_TYPE, r, P_MOD_SYNC, P_MOD_DEPTH])], [("AMOUNT", [P_MOD_COLOR, P_MOD_MIX])]]
        if name == "REV":
            return [[("REVERB", [P_REV_TYPE, P_REV_DECAY, P_REV_PRE, P_REV_MIX])],
                    [("TONE", [P_REV_DAMP, P_REV_LOCUT, P_REV_MOD, P_REV_COLOR])]]
        if name == "I/O":
            return [[("INPUT", [P_SOURCE, P_USBIN, P_ROUTE, P_INGAIN]), ("OUTPUT", [P_LEVEL, P_TRAILS, P_LATENCY])], []]
        return [[("PRESET", [P_PRESET, P_RESET]), ("MIDI", [P_CLOCK, P_CCMAP, P_CCRESET]), ("DISPLAY", [P_DISPLAY])], []]

    def page_params(self):
        return [p for col in self.page_layout() for _, ps in col for p in ps]

    def target(self, p):
        return self.ui if p.scope == "ui" else self.p

    def value_text(self, p):
        if p.kind == "source":
            src = self.ui["source"]
            if src == "auto":
                return "Auto  " + (self.src_label(self.cur_card) if self.cur_card else "none")
            return self.src_label(src) if src in dict(capture_cards()) else "USB (off)"
        if p.kind == "usbin":
            if not self.cur_card:
                return "-"
            n = len(capture_devices(self.cur_card))
            return f"Input {self.in_number(self.cur_card, self.cur_dev)}" + (f"/{n}" if n > 1 else "")
        if p.kind == "preset":
            return f"{self.ui['preset'] + 1} {self.preset_name()}"
        if p.kind == "action":
            if p.key == "ccmap":
                return f"{len(self.ui['ccmap'])} CC"
            return "Press A"
        if p.key == "latency":
            return f"{p.opts[self.ui['latency']]}  {LATENCY_FRAMES[self.ui['latency']] / SR * 1000:.0f} ms"
        if p.key == "tempo" and self.bpm():
            return f"{self.midi.clock_bpm:.0f} MIDI"
        pr = self.p
        if p.key in ("balance", "spread") and int(pr["routing"]) == 0:
            return "Split only"
        if p.key == "dly_color" and int(pr["dly_type"]) == 4:
            return DUAL_RATIOS[dual_ratio(pr)]
        if p.key == "dly_mode" and int(pr["dly_type"]) == 4:
            return "L / R"
        if p.key == "mod_rate" and int(pr["mod_type"]) == 8:
            return f"{1000 / max(mod_rate_hz(pr), 0.5):.0f} ms"
        if p.key == "cmp_ratio" and int(pr["cmp_type"]) == 3:
            return "Limit"
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
        if p.kind == "usbin":
            devs = capture_devices(self.cur_card) if self.cur_card else []
            if len(devs) > 1:
                i = devs.index(self.cur_dev) if self.cur_dev in devs else 0
                j = clamp(i + direction, 0, len(devs) - 1)
                if j != i:
                    self.ui["usb_card"], self.ui["usb_dev"] = self.cur_card, devs[j]
                    self.ui["usb_devs"][self.cur_card] = devs[j]
                    self.save_settings()
                    self.last_card_scan = 0.0        # check_card switches it now
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
        """Tap tempo: a free delay time follows the tap interval; synced delay / mod get the tapped tempo."""
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
        if int(p["dly_sync"]) or (not int(p["dly_type"]) and int(p["mod_sync"])):
            p["tempo"] = float(clamp(round(60.0 / iv), P_TEMPO.lo, P_TEMPO.hi))
            self.show_toast(f"TAP {p['tempo']:.0f} BPM", 1.2)
        else:
            p["dly_time"] = float(clamp(iv, P_TIME.lo, P_TIME.hi))
            p["tempo"] = float(clamp(round(60.0 / iv), P_TEMPO.lo, P_TEMPO.hi))
            self.show_toast(f"TAP {f_time(p['dly_time']).upper()}", 1.2)
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
                self.show_toast(f"CC{a} -> {cc_label(key, self.p)}", 3)
                continue
            key = self.ui["ccmap"].get(str(a))
            p = CC_TARGETS.get(key) if key else None
            if p is None:
                continue
            x = b / 127.0
            if p.kind == "switch":
                self.p[p.key] = 1 if b >= 64 else 0
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
        self.on_button(btn)

    def release_btn(self, btn):
        t0 = self.held.pop(btn, (time.time(), 0))[0]
        self.dirty = True
        if btn == "y" and not self.y_used and self.dialog_free() and time.time() - t0 < 0.6:
            self.tap(t0)                # the press time, so the tempo is as you tapped it
        if btn == "select":
            if not self.select_combo and time.time() - t0 < 0.4 and self.dialog_free():
                self.toggle_freeze()    # a short tap on SELECT freezes; holding it shows the controls
            self.select_combo = False

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
        if btn == "select":
            return
        if "select" in self.held:
            self.select_combo = True
        if btn == "start":
            if "select" in self.held:
                self.running = False
            else:
                self.save_presets(toast=True)
            return
        if btn == "tap":
            self.tap()
            return
        if btn == "freeze":
            if not repeat:
                self.toggle_freeze()
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
            self.show_toast(f"Move a MIDI knob for {cc_label(key, self.p)}  (B cancels)", 8)

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
                    lw = L.text(x + 1, y, p.lab(self.p), inv=is_sel)
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
            x += w + 2
        xr = 254
        on = bool(self.p["on"])
        frz = on and bool(self.p["freeze"])
        tag = "FRZ" if frz else "ON" if on else "BYP"
        w = L.width(tag) + 6
        if on:
            L.bar(xr - w, 1, w, 11)
            L.text(xr - w + 3, 3, tag, inv=True)
        else:
            L.frame(xr - w, 1, w, 11)
            L.text(xr - w + 3, 3, tag)
        xr -= w + 4
        xr -= L.rtext(xr, 3, f"P{self.ui['preset'] + 1}") + 4
        # tempo blink: quarter notes when synced, otherwise the delay time
        p = self.p
        bpm = self.bpm() or (p["tempo"] if int(p["dly_sync"]) or int(p["mod_sync"]) else None)
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
        """The repeats the delay makes: dry hit, then each echo, height = level. Ping-pong: L up, R down.
        Dual: left repeats up, right repeats (at the ratio) down. Reverse: each repeat is a rising ramp."""
        L = self.lcd
        p = self.p
        t_type = int(p["dly_type"])
        if not t_type:
            self.off_box(x, y, w, h, "DELAY")
            return
        L.frame(x, y, w, h)
        t = delay_seconds(p, self.bpm())
        s = int(p["dly_sync"])
        L.text(x + 4, y + 3, f_time(t).upper().replace(" ", ""), scale=2)
        right = SYNC[s][0] + ("  " + (f"{self.bpm():.0f} MIDI" if self.bpm() else f"{p['tempo']:.0f} BPM")) if s else "FREE"
        L.rtext(x + w - 4, y + 3, right)
        extra = {4: "R " + DUAL_RATIOS[dual_ratio(p)], 6: f"OCT {p['dly_color'] * 100:.0f}%"}.get(t_type, MODES[int(p["dly_mode"])].upper())
        L.rtext(x + w - 4, y + 12, DLY_TYPES[t_type].upper() + " " + extra.upper())
        g_top, g_bot = y + 20, y + h - 2
        mid = (g_top + g_bot) // 2
        amp_h = (g_bot - g_top) // 2 - 1
        gx0, gx1 = x + 13, x + w - 8
        L.hline(x + 2, mid, w - 4, dotted=True)
        L.text(x + 3, mid - 10, "L")
        L.text(x + 3, mid + 4, "R")
        mix = p["dly_mix"]
        dry = 1.0 if mix < 0.5 else 2 * (1 - mix)
        wet = 2 * mix if mix < 0.5 else 1.0
        frozen = bool(p["freeze"])
        fb = 1.0 if frozen else p["dly_fb"]
        mode = int(p["dly_mode"])
        width = p["dly_width"]
        loss = 1.0 if frozen else (min(p["dly_hicut"], 12000) / 12000) ** 0.2 * (1 - min(p["dly_locut"], 1000) / 4000)
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
        if t_type == 4:                         # dual: two trains of repeats at different spacing
            ratio = DUAL_RATIO_V[dual_ratio(p)]
            for side, r in ((0, 1.0), (1, ratio)):
                for k, a in enumerate(amps[:24]):
                    ex = int(gx0 + (k + 1) * step * r)
                    if ex > gx1:
                        break
                    hh = max(1, height(a))
                    top, bot = (mid - hh, mid - 1) if side == 0 else (mid + 1, mid + hh)
                    L.columns(ex - 1, [top] * 3, [bot] * 3, dotted=bypass)
        else:
            for k, a in enumerate(amps[:n]):
                ex = int(gx0 + (k + 1) * step)
                L.vline(ex, mid - 1, 3)                                     # time grid: one tick per repeat
                hh = max(1, height(a))
                if mode == 1:                   # ping-pong: odd repeats left (up), even right (down)
                    side = int(hh * (1 - width))
                    up, down = (hh, side) if k % 2 == 0 else (side, hh)
                else:
                    up = down = hh
                if t_type == 5:                 # reverse: each repeat swells up to its end
                    rw = max(3, int(step * 0.8))
                    xs = np.arange(rw)
                    hu = (up * (xs + 1) / rw).astype(int)
                    hd = (down * (xs + 1) / rw).astype(int)
                    L.columns(ex - rw + 1, mid - hu, mid + hd, dotted=bypass)
                else:
                    L.columns(ex - 1, [mid - up] * 3, [mid + down] * 3, dotted=bypass)
        if frozen:
            self.frozen_tag(x + w - 4, y + 21)

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
        par = int(p["routing"]) == 1
        by, bh = y + 4, 36
        L.hline(x, by + bh // 2, w, dotted=True)

        tails = {"dly_type": self.engine.s.get("dly_tail", 0), "rev_type": self.engine.s.get("rev_tail", 0)}

        def box(bx, byy, bw, bhh, name, key, opts, one_line=False):
            t = int(p[key])
            on = t and not bypass
            label = opts[t].upper() if t else ("TAIL" if tails.get(key) else "OFF")
            L.erase(bx, byy, bw, bhh)
            if on:
                L.bar(bx, byy, bw, bhh)
            else:
                (L.frame if t else self._dotted_frame)(bx, byy, bw, bhh)
            if one_line:
                L.ctext(bx + bw // 2, byy + (bhh - 7) // 2, L.fit(f"{name} {label}", bw - 4), inv=on)
            else:
                L.ctext(bx + bw // 2, byy + 7, name, inv=on)
                L.ctext(bx + bw // 2, byy + 20, L.fit(label, bw - 4), inv=on)

        if not par:
            bw, gap = 44, 8
            for i, (name, key, opts) in enumerate(SECTIONS):
                bx = x + i * (bw + gap)
                box(bx, by, bw, bh, name, key, opts)
                if i < 4:
                    L.text(bx + bw + 2, by + bh // 2 - 3, ">")
        else:
            bw, gap = 42, 8
            for i, (name, key, opts) in enumerate(SECTIONS[:3]):
                bx = x + i * (bw + gap)
                box(bx, by, bw, bh, name, key, opts)
                if i < 2:
                    L.text(bx + bw + 2, by + bh // 2 - 3, ">")
            fx = x + 3 * (bw + gap) - gap + 1               # the split: a fork into two paths
            L.erase(fx, by, x + w - fx, bh)
            L.hline(fx, by + bh // 2, 4)
            L.vline(fx + 4, by + 8, bh - 16)
            L.hline(fx + 4, by + 8, 4)
            L.hline(fx + 4, by + bh - 9, 4)
            px = fx + 9
            pw = x + w - px
            box(px, by, pw, 17, "DLY", "dly_type", DLY_TYPES, one_line=True)
            box(px, by + 19, pw, 17, "REV", "rev_type", REV_TYPES, one_line=True)
        if bypass:
            msg = "BYPASS - DRY SIGNAL"
        elif par:
            msg = f"SPLIT  {f_bal(p['balance']).upper()}  SPREAD {p['spread'] * 100:.0f}%"
        else:
            msg = f"P{self.ui['preset'] + 1} {self.preset_name()}"
        L.ctext(x + w // 2, y + h - 18, msg)
        info = f"WIDTH {p['owidth'] * 100:.0f}%" + ("   FREEZE ON" if p["freeze"] else "")
        L.ctext(x + w // 2, y + h - 8, info)

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


    def draw_mod_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        t = int(p["mod_type"])
        if not t:
            self.off_box(x, y, w, h, "MOD")
            return
        L.frame(x, y, w, h)
        rate, depth = mod_rate_hz(p, self.bpm()), p["mod_depth"]
        L.text(x + 4, y + 3, MOD_TYPES[t].upper(), scale=2)
        sync = SYNC[int(p["mod_sync"])][0]
        if t == 8:
            L.rtext(x + w - 4, y + 3, f"SLICE {1000 / max(rate, 0.5):.0f} MS" + (f" {sync}" if int(p["mod_sync"]) else ""))
            L.rtext(x + w - 4, y + 12, f"CHANCE {depth * 100:.0f}%  VARIETY {p['mod_color'] * 100:.0f}%")
            ev = int(self.engine.s.get("glitch_ev", 0))
            L.ctext(x + w // 2, y + 25, GLITCH_NAMES[ev] if ev else "- LIVE -", scale=2)
            cw = (w - 8) // 14
            for k, e in enumerate(self.glitch_hist):
                bx = x + 4 + k * cw
                L.erase(bx, y + 44, cw - 2, 14)
                if k == len(self.glitch_hist) - 1 and ev:
                    L.bar(bx, y + 44, cw - 2, 14)
                    L.ctext(bx + cw // 2 - 1, y + 48, GLITCH_NAMES[e][:2], inv=True)
                else:
                    L.frame(bx, y + 44, cw - 2, 14)
                    L.ctext(bx + cw // 2 - 1, y + 48, GLITCH_NAMES[e][:2])
            return
        L.rtext(x + w - 4, y + 3, f_rate(rate).upper() + (f" {sync}" if int(p["mod_sync"]) else ""))
        L.rtext(x + w - 4, y + 12, f"DEPTH {depth * 100:.0f}%  {P_MOD_COLOR.lab(p).upper()} {p['mod_color'] * 100:.0f}%")
        # the LFO for left and right, moving in real time
        offset = {5: 0.0, 6: 0.5, 7: 0.0}.get(t, 0.25 if t in (3, 4) else 0.5 * p["mod_color"] + 1 / 6)
        cycles = clamp(rate * 2.0, 0.5, 14)
        gx0, gx1 = x + 12, x + w - 4
        n = gx1 - gx0
        ph0 = (time.time() * rate) % 1.0
        amp = max(1.0, depth * 7.5)
        shape = p["mod_color"] if t in (5, 6) else 0.0
        k = 1 + shape * 12
        for row, (label, off) in enumerate((("L", 0.0), ("R", offset))):
            cy = y + 29 + row * 17
            L.text(x + 3, cy - 3, label)
            L.hline(gx0, cy, n, dotted=True)
            ph = ph0 + off - np.arange(n) / n * cycles
            sw = np.sin(2 * np.pi * ph)
            if shape > 0.01:
                sw = np.tanh(k * sw) / math.tanh(k)
            if t == 7 and p["mod_color"] > 0.01:        # drift: some random wander on the vibrato
                rng = np.random.default_rng(int(time.time() * rate * 2) % 1000)
                wander = np.interp(np.arange(n), np.linspace(0, n, 6), rng.uniform(-1, 1, 6))
                sw = (1 - p["mod_color"]) * sw + p["mod_color"] * wander
            self.curve(gx0, cy - sw * amp)

    def rt60(self, p=None):
        """Rough reverb time: tank loop time and gain (decay applied 4 times per loop), less the damping."""
        p = p or self.p
        size = REV_SIZE[int(p["rev_type"])]
        loop = 21589 * size / 29761
        dec = 0.2 + 0.795 * p["rev_decay"]
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
        frozen = bool(p["freeze"])
        rt = self.rt60()
        pre = p["rev_pre"]
        L.text(x + 4, y + 3, REV_TYPES[t].upper(), scale=2)
        L.rtext(x + w - 4, y + 3, "DECAY HELD" if frozen else f"DECAY {rt:.1f} S")
        sub = f"PRE-DELAY {pre * 1000:.0f} MS"
        if t in (5, 6):
            sub = f"{P_REV_COLOR.lab(p).upper()} {p['rev_color'] * 100:.0f}%  PRE {pre * 1000:.0f} MS"
        L.rtext(x + w - 4, y + 12, sub)
        # energy over time: a gap for the pre-delay, dense early part, then the tail (dB scale)
        gx0, gx1, top, bot = x + 4, x + w - 4, y + 21, y + h - 3
        span = clamp(max(rt * 1.15 + pre, 0.6), 0.6, 16.0)
        L.hline(gx0, bot, gx1 - gx0)
        n = gx1 - gx0
        jit = 0.55 + 0.45 * np.random.default_rng(7).random(n)
        tt = np.arange(n) / n * span - pre
        db = np.zeros(n) if frozen else -60 * np.maximum(tt, 0) / max(rt, 0.05)
        rise = 0.02 if t != 6 else 0.15 + 1.2 * p["rev_color"]          # bloom: a slow build
        att = np.minimum(1.0, np.maximum(tt, 0) / rise + (0.2 if t != 6 else 0.0))
        hgt = (np.clip((db + 48) / 48, 0, 1) * (bot - top) * jit * att).astype(int)
        hgt[(tt < 0) | (np.arange(n) % 2 == 1)] = 0                 # a bar every second column
        L.columns(gx0, bot - hgt, np.where(hgt > 0, bot - 1, bot - hgt - 1))
        secs = 1 if span < 4 else 2 if span < 9 else 4
        for k in range(1, int(span / secs) + 1):
            xm = gx0 + int(k * secs / span * (gx1 - gx0))
            if xm < gx1 - 10:
                L.vline(xm, bot + 1, 2)
        if frozen:
            self.frozen_tag(x + w - 4, y + 21)
        else:
            L.rtext(gx1 - 2, top, f"{span:.0f} S" if span >= 2 else f"{span:.1f} S")

    def footer(self):
        L = self.lcd
        L.hline(0, 178, LCD_W, dotted=True)
        hint = "X BYP  Y TAP  SEL FREEZE  L2/R2 PRESET" if self.last_input == "pad" else \
            "SPACE BYP  T TAP  F FREEZE  , . PRESET"
        L.ctext(128, 183, hint)

    def fx_page(self, graph):
        self.draw_meters(18)
        graph(*self.BOX)
        self.draw_sections(100)
        self.footer()

    def draw_chain(self):
        self.fx_page(self.draw_chain_graph)


    def draw_mod(self):
        self.fx_page(self.draw_mod_graph)

    def draw_dly(self):
        self.fx_page(self.draw_echoes)

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
                    ("DEVICE", self.device_status()),
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
        lines = [L.fit("IN: " + err, 250), ""] if err else ["USB AUDIO IN, BRICK JACK OUT.", "AUTO TAKES THE FIRST USB CARD"]
        for i, line in enumerate(lines):
            L.text(3, 166 + i * 11, line)

    def device_status(self):
        """USB n IN m of the device the engine really has open (it may have fallen back to DEV=0)."""
        if not self.cur_card:
            return "-"
        d = opened_dev(self.engine.text("eng_cap_device")) if self.engine.s.get("cap_open") else None
        d = self.cur_dev if d is None else d
        return f"{self.src_label(self.cur_card, True)} IN {self.in_number(self.cur_card, d)}"

    def draw_set(self):
        L = self.lcd
        self.draw_sections(18)
        x, w, ey = 131, 123, 18
        L.text(x, ey, "ENGINE")
        L.hline(x + L.width("ENGINE") + 2, ey + 6, w - L.width("ENGINE") - 2, dotted=True)
        s = self.engine.s
        rows = [("VERSION", f"{VERSION} SARTA"), ("DSP", "NATIVE C"), ("BLOCK", f"{BLOCK} FR"),
                ("REALTIME", "YES" if s["rt"] else "NO") if self.engine.threaded else ("OUTPUT", "SDL"),
                ("MIDI", ", ".join(self.midi.devices) or "NONE"),
                ("CLOCK", f"{self.midi.clock_bpm:.1f} BPM" if self.midi.clock_running() else "-")]
        for i, (k, v) in enumerate(rows):
            yy = ey + 12 + i * 12
            lw = L.text(x + 1, yy, k)
            L.rtext(x + w - 2, yy, L.fit(v, w - lw - 8))
        for i, line in enumerate(["PRESETS SAVE THEMSELVES. START SAVES NOW.", "B ON AN FX VALUE: MIDI LEARN.",
                                  "PROGRAM CHANGE 0-15 PICKS THE PRESET.", "* = MOVED BY A MIDI CC.",
                                  "TAP SELECT: FREEZE DELAY AND REVERB."]):
            L.text(3, 125 + i * 11, line)

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
                                        ("TAP Y", "TAP TEMPO"), ("X", "ON / BYPASS"), ("TAP SELECT", "FREEZE"),
                                        ("A", "STEP A SETTING"), ("B", "MIDI LEARN"), ("L1 / R1", "PAGE"),
                                        ("L2 / R2", "PRESET"), ("START", "SAVE"), ("SELECT+START", "QUIT"),
                                        ("VOLUME", "OUTPUT VOLUME")]
        else:
            title, rows = "KEYBOARD", [("UP / DOWN", "SELECT"), ("LEFT / RIGHT", "VALUE (SHIFT FINE)"), ("T", "TAP TEMPO"),
                                        ("SPACE", "ON / BYPASS"), ("F", "FREEZE"), ("Z / ENTER", "STEP A SETTING"),
                                        ("X / ESC", "MIDI LEARN"), ("[ ]", "PAGE"), (", .", "PRESET"),
                                        ("CTRL+S / Q", "SAVE / QUIT")]
        L.text(2, 18, title)
        L.rtext(254, 18, "RELEASE TO CLOSE")
        L.hline(0, 28, LCD_W, dotted=True)
        for i, (k, v) in enumerate(rows):
            y = 32 + i * 11
            L.text(4, y, k)
            L.text(100, y, v)

    def draw(self):
        self.lcd.clear()
        self.draw_header()
        {"FX": self.draw_chain, "CMP": self.draw_cmp, "AMP": self.draw_amp, "MOD": self.draw_mod, "DLY": self.draw_dly,
         "REV": self.draw_rev, "I/O": self.draw_io, "SET": self.draw_set}[PAGES[self.page]]()
        self.draw_overlays()
        self.present()

    def preset_name(self, i=None):
        i = self.ui["preset"] if i is None else i
        return PRESET_NAMES[i] if i < len(PRESET_NAMES) else f"USER {i + 1}"

    def toggle_freeze(self):
        self.p["freeze"] = 0 if self.p["freeze"] else 1
        self.presets_dirty = self.presets_dirty or time.time()
        self.show_toast("FREEZE" if self.p["freeze"] else "FREEZE OFF", 1.0)

    def frozen_tag(self, xr, y):
        L = self.lcd
        w = L.width("FROZEN") + 6
        L.erase(xr - w - 1, y - 1, w + 2, 13)
        L.bar(xr - w, y, w, 11)
        L.text(xr - w + 3, y + 2, "FROZEN", inv=True)

    def draw_comp_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        t = int(p["cmp_type"])
        if not t:
            self.off_box(x, y, w, h, "COMP")
            return
        L.frame(x, y, w, h)
        L.text(x + 4, y + 3, CMP_TYPES[t].upper(), scale=2)
        gr = self.engine.s.get("gr", 0.0)
        gx0, gx1, top, bot = x + 6, x + 150, y + 21, y + h - 4
        if t == 4:                              # swell: the volume rising after each new note
            sw = p["cmp_swell"]
            L.rtext(x + w - 4, y + 3, f"SWELL {f_time(sw).upper()}")
            L.rtext(x + w - 4, y + 12, f"THRESHOLD {p['cmp_thr']:.0f} DB")
            span = max(sw * 1.6, 0.2)
            n = gx1 - gx0
            tt = np.arange(n) / n * span
            ys = np.minimum(1, tt / sw) ** 2
            L.hline(gx0, bot, n, dotted=True)
            self.curve(gx0, bot - ys * (bot - top))
            L.text(gx0, top, "VOLUME")
        else:
            thr, ratio = p["cmp_thr"], (100.0 if t == 3 else p["cmp_ratio"])
            knee = 12.0 if t == 2 else 1.0 if t == 3 else 6.0
            L.rtext(x + w - 4, y + 3, f"{thr:.0f} DB  " + ("LIMIT" if t == 3 else f_ratio(ratio).upper()))
            L.rtext(x + w - 4, y + 12, f"MAKEUP +{p['cmp_makeup']:.0f} DB")
            # static curve: input -60..0 dB left to right, output bottom to top (before makeup)
            n = gx1 - gx0
            xin = np.linspace(-60, 0, n)
            over = xin - thr
            slope = 1 / ratio - 1
            red = np.where(over <= -knee / 2, 0, np.where(over < knee / 2, slope * (over + knee / 2) ** 2 / (2 * knee), slope * over))
            yout = xin + red
            L.hline(gx0, bot, n, dotted=True)
            L.vline(gx0, top, bot - top, dotted=True)
            tx = gx0 + int((thr + 60) / 60 * n)
            L.vline(tx, top, bot - top, dotted=True)
            self.curve(gx0, bot - (yout + 60) / 60 * (bot - top))
            L.text(gx0 + 3, top, "OUT")
            L.rtext(gx1, bot - 8, "IN")
        # live gain reduction meter
        mx = x + 168
        L.text(mx, y + 26, "GR" if t != 4 else "VOL")
        db = clamp(-gr, 0, 24)
        mw = w - (mx - x) - 28
        n = int((24 - db if t == 4 else db) / 24 * (mw // 3))      # swell: the bar is the volume
        for k in range(mw // 3):
            bx = mx + 22 + k * 3
            if k < n:
                L.fill(bx, y + 25, 2, 9)
            else:
                L.fill(bx, y + 29, 2, 1)
        L.text(mx, y + 40, f"-{db:.1f} DB")
        L.text(mx, y + 50, f"MIX {p['cmp_mix'] * 100:.0f}%")

    def draw_amp_graph(self, x, y, w, h):
        L = self.lcd
        p = self.p
        t = int(p["amp_type"])
        if not t:
            self.off_box(x, y, w, h, "AMP")
            return
        L.frame(x, y, w, h)
        amt = p["amp_gain"]
        L.text(x + 4, y + 3, AMP_TYPES[t].upper(), scale=2)
        L.rtext(x + w - 4, y + 3, f"CAB {CAB_TYPES[int(p['amp_cab'])].upper()}")
        L.rtext(x + w - 4, y + 12, f"{P_AMP_GAIN.lab(p).upper()} {amt * 100:.0f}%")
        # left: transfer curve
        gx0, gx1, top, bot = x + 6, x + 76, y + 22, y + h - 4
        mid_x, mid_y = (gx0 + gx1) // 2, (top + bot) // 2
        L.hline(gx0, mid_y, gx1 - gx0, dotted=True)
        L.vline(mid_x, top, bot - top, dotted=True)
        xs = np.linspace(-1, 1, gx1 - gx0)
        if t == 5:
            levels = 2 ** max(1.0, min(6.0, 15 - amt * 11 - 9))
            ys = np.round(xs * levels) / levels
        elif t == 4:
            g = 1 + 80 * amt * amt
            ys = 0.85 * np.clip(xs * g, -1, 1) + 0.15 * np.tanh(xs * g * 0.5)
        else:
            g = {1: 1 + 7 * amt * amt, 2: 1 + 40 * amt * amt, 3: 2 + 150 * amt * amt}[t]
            b = {1: 0.05, 2: 0.15, 3: 0.2}[t]
            ys = np.tanh(xs * g + b) - np.tanh(b)
            if t == 2:
                ys = np.tanh(1.8 * ys)
            elif t == 3:
                ys = np.tanh(3 * ys)
        ys = ys / max(1e-6, np.abs(ys).max())
        self.curve(gx0, mid_y - ys * (bot - top) / 2)
        # right: tone stack + cabinet response, 40 Hz .. 12 kHz, -24 .. +12 dB
        ex0, ex1 = x + 92, x + w - 6
        n = ex1 - ex0
        f = 40 * (12000 / 40) ** (np.arange(n) / (n - 1))
        resp = (_bq("ls", 150, .7, (p["amp_bass"] - .5) * 24, f) + _bq("peak", 750, .7, (p["amp_mid"] - .5) * 24, f)
                + _bq("hs", 3200, .7, (p["amp_treble"] - .5) * 24, f))
        for kind, fc, q, db in CAB_FILTERS[int(p["amp_cab"])]:
            resp = resp + _bq(kind, fc, q, db, f)
        zero = top + int((bot - top) * 12 / 36)
        L.hline(ex0, zero, n, dotted=True)
        for fm in (100, 1000, 10000):
            L.vline(ex0 + int(math.log(fm / 40) / math.log(300) * (n - 1)), bot - 2, 3)
        self.curve(ex0, zero - np.clip(resp, -24, 12) / 36 * (bot - top))
        L.text(ex0, top, "EQ + CAB")

    def track_glitch(self):
        s = self.engine.s
        c = int(s.get("glitch_count", 0))
        if c != self.glitch_seen:
            if int(self.p["mod_type"]) == 8 and int(s.get("glitch_ev", 0)):
                self.glitch_hist = (self.glitch_hist + [int(s["glitch_ev"])])[-14:]
            self.glitch_seen = c

    def draw_cmp(self):
        self.fx_page(self.draw_comp_graph)

    def draw_amp(self):
        self.fx_page(self.draw_amp_graph)

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
            self.track_glitch()
            now = time.time()
            if self.presets_dirty and now - self.presets_dirty > 2.0:
                self.save_presets()
            if self.dirty or n_held != len(self.held) or now - last_draw >= 1 / fps:
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
    print(f"BrickAmbientFX {VERSION} ({EDITION}) starting: python {sys.version.split()[0]}, pygame {pygame.version.ver}, "
          f"numpy {np.__version__}, device={ON_DEVICE}", flush=True)
    load_config()
    ap = argparse.ArgumentParser(description="BrickAmbientFX")
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
