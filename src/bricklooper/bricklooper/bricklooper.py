#!/usr/bin/env python3
"""
BrickLooper - a 5-track loop station with the BrickSampler FX.
Target: TrimUI Brick running KNULLI (Python 3.12, Pygame, NumPy). Runs on desktop for testing.

  pip install pygame numpy
  python3 bricklooper.py             # options: --scale 0.8  --fps 30  --fullscreen  --debug-input

5 stereo tracks in sync with the first loop, recorded from a USB input (interface, mic, synth) or the
Brick's built-in mic. FX (drive, filter, delay, reverb) on the output. MIDI CCs can move the FX.

Copyright (C) 2026 Alessandro Pietrosanti / BrickWorld (brickworldhub.com)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file or
<https://www.gnu.org/licenses/>.
"""
import os, sys, time, json, glob, math, struct, wave, threading, collections, datetime, argparse, select, traceback
import shutil, subprocess, re

# Keep BLAS single-threaded: on ARM its worker threads spin-wait and starve the UI and audio threads.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import faulthandler


def _fatal(msg):
    """Show a startup error on screen (instead of a black screen), log it, and quit."""
    print("FATAL:", msg, flush=True)
    try:
        import pygame as pg
        pg.display.init()
        pg.font.init()
        scr = pg.display.set_mode((1024, 768), pg.FULLSCREEN if os.path.isdir("/userdata/roms") else 0)
        scr.fill((178, 185, 228))
        f = pg.font.Font(None, 40)
        for i, line in enumerate(["BrickLooper can't start", ""] + [msg[j:j + 60] for j in range(0, len(msg), 60)]
                                 + ["", "Details in bricklooper/log.txt. Closing in 10 s."]):
            scr.blit(f.render(line, True, (22, 30, 100)), (60, 120 + i * 44))
        pg.display.flip()
        time.sleep(10)
    except Exception:
        pass
    os._exit(1)


# NumPy: use the system copy if there is one, otherwise the copy bundled in vendor/ (KNULLI has none;
# the launcher unpacks vendor_numpy.zip there on first run).
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

APP_NAME = "BRICKLOOPER"
VERSION = "1.5.2"
# 48 kHz on the Brick: the system mixer and the codec run at 48k, and the speaker and mic share one
# clock, so asking the mic for 44.1k while playback runs at 48k can stall the capture.
SR = 48000 if os.path.isdir("/userdata/roms") else 44100
# Internal DSP block. Bigger blocks mean fewer Python/NumPy calls per second, which is what costs
# CPU on the Brick; 1024 frames there (23 ms), 512 on a computer. Reverb combs must stay >= BLOCK.
def _early_config(key, default):
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls.json")) as f:
            return json.load(f).get(key, default)
    except Exception:
        return default


# Internal DSP block: notes start at block boundaries, so smaller = less latency. The native DSP makes
# 256 frames (5 ms at 48 kHz) cheap enough on the Brick. controls.json "audio_block" overrides.
BLOCK = int(os.environ.get("BRICKLOOPER_BLOCK", _early_config("audio_block", 256)))
SUB = 256                   # sub-block for the biquad and the short reverb allpasses
DEVICE_CHUNK = 512          # audio device buffer (frames)
MAX_VOICES = 8
NUM_PADS = 16
KIT_BASE_NOTE = 36
SCREEN_W, SCREEN_H = 1024, 768
MAX_REC_SECONDS = 60
BEND_RANGE = 2.0
SWAP_AB = False             # set True if A/B feel swapped on the Brick (Nintendo layout)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "controls.json")
SAMPLES_DIR = os.path.join(HERE, "samples")
DEMO_DIR = os.path.join(SAMPLES_DIR, "demo")
REC_DIR = os.path.join(SAMPLES_DIR, "recordings")
PROJECTS_DIR = os.path.join(HERE, "projects")
SETTINGS_FILE = os.path.join(HERE, "settings.json")
NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
NAME_MAX = 20
KB_ROWS = [list("ABCDEFGHIJKLM"), list("NOPQRSTUVWXYZ"), list("0123456789-_")]


def clean_name(s):
    """Names for samples and projects: A-Z, 0-9, - and _, at most NAME_MAX characters."""
    return "".join(c for c in str(s).upper().replace(" ", "_") if c in NAME_CHARS)[:NAME_MAX]
ON_DEVICE = os.path.isdir("/userdata/roms")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(n):
    return f"{NOTE_NAMES[int(n) % 12]}{int(n) // 12 - 1}"


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ----------------------------------------------------------------------------- WAV I/O

def resample_linear(x, src, dst):
    n_out = max(1, int(len(x) * dst / src))
    t = np.arange(n_out) * (src / dst)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def load_wav(path):
    """PCM 8/16/24/32-bit and float 32/64 WAV, any channel count -> mono float32 at SR."""
    x = read_wav_frames(path).mean(axis=1)
    if len(x) < 4:
        raise ValueError("sample too short")
    return np.ascontiguousarray(x, dtype=np.float32)


def load_wav_stereo(path):
    """Like load_wav but keeps two channels (a mono file is copied to both): (frames, 2) float32 at SR."""
    x = read_wav_frames(path)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x[:, :2], dtype=np.float32)


def save_wav_stereo_float(path, x):
    """(frames, 2) float32 WAV at the engine rate. Float keeps overdubs that go past full scale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = np.ascontiguousarray(x, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, SR, SR * 8, 8, 32)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(raw)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", len(raw)) + raw)


def read_wav_frames(path):
    """Any WAV this app reads -> (frames, channels) float32 at SR."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError("missing fmt or data chunk")
    tag, ch, rate, bits = fmt
    width = bits // 8
    n = len(raw) // (width * ch)
    raw = raw[:n * width * ch]
    if tag == 1:
        if bits == 8:
            x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 16:
            x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        elif bits == 24:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float32) / 8388608.0
        elif bits == 32:
            x = (np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
        else:
            raise ValueError(f"unsupported bit depth {bits}")
    elif tag == 3:
        x = np.frombuffer(raw, "<f4" if bits == 32 else "<f8").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format {tag}")
    x = x.reshape(-1, ch)
    if rate != SR:
        x = np.stack([resample_linear(x[:, c], rate, SR) for c in range(ch)], axis=1)
    return x.astype(np.float32)


def save_wav(path, x, rate=None, bits=16):
    """Mono WAV at `rate` (default: engine rate) and 8, 16 or 24 bit."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(np.asarray(x, np.float64), -1, 1)
    if bits == 8:
        raw = np.clip(np.round(x * 127) + 128, 0, 255).astype(np.uint8).tobytes()      # 8-bit WAV is unsigned
    elif bits == 24:
        v = np.clip(np.round(x * 8388607), -8388608, 8388607).astype("<i4")
        raw = v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        raw = np.round(x * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(bits // 8)
        w.setframerate(int(rate or SR))
        w.writeframes(raw)


def resample_fft(x, src, dst):
    """Band-limited resampling (no aliasing when going down, e.g. 48k -> 32k)."""
    if src == dst:
        return np.asarray(x, np.float64)
    n_out = max(1, int(round(len(x) * dst / src)))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, complex)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))


def loop_bounds(p, n):
    """Loop start/end in samples, clamped the way the engine clamps them."""
    s = int(p["start"] * n + 1e-3)
    e = min(n, max(int(p["end"] * n + 1e-3), s + 2))
    s = min(s, e - 2)
    ls = min(max(int(p["lstart"] * n + 1e-3), s), e - 2)
    le = max(min(int(p["lend"] * n + 1e-3), e), ls + 2)
    return ls, le


def detect_onsets(x, sr, max_slices=16):
    """Slice points at the attacks of a mono signal: rises of the short-term level, strongest first,
    at least 60 ms apart; each one moved back to the quietest point just before the attack."""
    hop = max(1, int(sr * 0.005))
    nfr = len(x) // hop
    if nfr < 8:
        return [0, len(x)]
    e = np.sqrt(np.add.reduceat(x[:nfr * hop].astype(np.float64) ** 2, np.arange(0, nfr * hop, hop)) / hop)
    db = 20 * np.log10(e + 1e-6)
    rise = np.zeros(nfr)
    rise[2:] = db[2:] - db[:-2]
    loud = db > db.max() - 40
    cand = [i for i in range(2, nfr - 1) if loud[i] and rise[i] >= 6 and rise[i] >= rise[i - 1] and rise[i] > rise[i + 1]]
    cand.sort(key=lambda i: -rise[i])
    gap = int(0.06 * sr / hop)
    picked = []
    for i in cand:
        if all(abs(i - j) >= gap for j in picked) and i >= gap // 2:
            picked.append(i)
        if len(picked) >= max_slices - 1:
            break
    pts = []
    for i in sorted(picked):
        seg = db[max(0, i - 4):i + 1]                                     # back to the quiet just before:
        j = max(0, i - 4) + int(np.nonzero(seg <= seg.min() + 3)[0][-1])  # the last frame near the minimum
        pts.append(j * hop)
    return [0] + [q for q in pts if q > 0] + [len(x)]


def time_stretch(x, ratio, sr, progress=None):
    """WSOLA: change the length by `ratio` keeping the pitch. Overlap-adds 40 ms Hann windows; each one
    is taken from within +-10 ms of its ideal place where it best continues the previous one."""
    win = int(0.04 * sr) // 2 * 2
    hop = win // 2
    tol = int(0.01 * sr)
    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float32)   # periodic: sums to 1
    pad = np.concatenate([np.zeros(tol + win, np.float32), x.astype(np.float32), np.zeros(tol + 2 * win, np.float32)])
    off = tol + win
    n_out = int(len(x) * ratio)
    frames = n_out // hop + 2
    out = np.zeros(frames * hop + win, np.float32)
    nfft = 1 << int(np.ceil(np.log2(2 * tol + 2 * win)))
    prev = off - hop
    for k in range(frames):
        ideal = off - hop + int(k * hop / ratio)
        if k == 0:
            best = ideal
        else:
            tmpl = pad[prev + hop: prev + hop + win] * w        # what naturally follows the last window
            lo = max(0, ideal - tol)
            region = pad[lo: lo + 2 * tol + win]
            if len(region) < win + 1 or len(tmpl) < win:
                best = ideal
            else:
                c = np.fft.irfft(np.fft.rfft(region, nfft) * np.conj(np.fft.rfft(tmpl, nfft)), nfft)
                best = lo + int(np.argmax(c[:len(region) - win + 1]))
        seg = pad[best: best + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        out[k * hop: k * hop + win] += seg * w
        prev = best
        if progress and k % 32 == 0:
            progress(k / frames)
    return out[hop: hop + n_out]


def zero_cross(x, i, radius):
    """Nearest rising zero crossing to sample i within radius (i itself if there is none)."""
    a, b = max(1, i - radius), min(len(x) - 1, i + radius)
    if b <= a:
        return i
    seg = x[a - 1:b]
    z = np.nonzero((seg[:-1] <= 0) & (seg[1:] > 0))[0] + a
    return int(z[np.argmin(np.abs(z - i))]) if len(z) else i


def list_samples():
    files = glob.glob(os.path.join(SAMPLES_DIR, "**", "*.wav"), recursive=True)
    files += glob.glob(os.path.join(SAMPLES_DIR, "**", "*.WAV"), recursive=True)
    return sorted(set(files), key=lambda p: os.path.relpath(p, SAMPLES_DIR).lower())


def make_demo_samples():
    """A few synthesised sounds so there is something to play on first run."""
    if os.path.isdir(DEMO_DIR) and glob.glob(os.path.join(DEMO_DIR, "*.wav")):
        return
    rng = np.random.default_rng(7)

    def save(name, x):
        fade = np.minimum(1.0, np.arange(len(x)) / (SR * 0.002))
        x = x * fade
        save_wav(os.path.join(DEMO_DIR, name), x / max(1e-9, np.abs(x).max()) * 0.89)

    t = np.arange(int(SR * 3.0)) / SR
    f0 = 261.6256
    keys = sum((0.55 ** k) * np.sin(2 * np.pi * f0 * (k + 1) * t * (1 + 0.0007 * k)) * np.exp(-t * (0.9 + 1.7 * k))
               for k in range(7))
    save("keys_C4.wav", keys)

    t = np.arange(int(SR * 0.6)) / SR
    freq = 45 + 95 * np.exp(-t * 30)
    kick = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 7) + 0.3 * rng.standard_normal(len(t)) * np.exp(-t * 400)
    save("kick.wav", kick)

    t = np.arange(int(SR * 0.3)) / SR
    noise = rng.standard_normal(len(t))
    snare = 0.8 * np.diff(noise, prepend=0) * np.exp(-t * 17) + 0.6 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28)
    save("snare.wav", snare)

    t = np.arange(int(SR * 0.12)) / SR
    hat = np.diff(np.diff(rng.standard_normal(len(t)), prepend=0), prepend=0) * np.exp(-t * 55)
    save("hat.wav", hat)

    t = np.arange(int(SR * 0.5)) / SR
    freq = 95 + 60 * np.exp(-t * 18)
    tom = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 8)
    save("tom.wav", tom)


# ----------------------------------------------------------------------------- block IIR

class BlockLTI:
    """Runs a low-order IIR filter on whole blocks without a per-sample Python loop.

    The filter is written in state-space form. For a fixed block length N its output is
    y = x * T + s0 * F, and the state after the block is s0 * A^N + x * G, where T, F, G
    depend only on the coefficients. They are rebuilt when coefficients change, so each
    block costs two small matrix multiplies for any number of channels.
    """
    _toe_idx = {}

    def __init__(self, channels, order, n=BLOCK):
        self.n, self.order = n, order
        self.S = np.zeros((channels, order))
        self.key = None
        if n not in BlockLTI._toe_idx:
            i = np.arange(n)
            d = i[None, :] - i[:, None]           # [k, m] = m - k
            BlockLTI._toe_idx[n] = (np.clip(d, 0, n - 1), d >= 0)

    def reset(self):
        self.S[:] = 0

    def set(self, b, a):
        key = (tuple(b), tuple(a))
        if key == self.key:
            return
        self.key = key
        n, N = self.order, self.n
        a0 = a[0]
        b = [v / a0 for v in b] + [0.0] * (n + 1 - len(b))
        a = [v / a0 for v in a] + [0.0] * (n + 1 - len(a))
        A = np.zeros((n, n))
        for i in range(n):
            A[i, 0] = -a[i + 1]
            if i + 1 < n:
                A[i, i + 1] = 1.0
        B = np.array([b[i + 1] - a[i + 1] * b[0] for i in range(n)])
        # P[m] = A^m for m = 0..N-1, built by doubling
        P = np.empty((N, n, n))
        P[0] = np.eye(n)
        Ak, filled = A.copy(), 1
        while filled < N:
            cnt = min(filled, N - filled)
            P[filled:filled + cnt] = P[:cnt] @ Ak
            filled += cnt
            Ak = Ak @ Ak
        F = P[:, 0, :].T                        # C A^m, C = e0
        h = np.empty(N)
        h[0] = b[0]
        h[1:] = (P[:-1, 0, :] @ B)
        G = (P[::-1] @ B)                        # row k = A^(N-1-k) B
        AN = P[-1] @ A
        idx, mask = BlockLTI._toe_idx[N]
        self.T = np.where(mask, h[idx], 0.0)
        self.F, self.G, self.ANT = F, G, AN.T

    def process(self, X):
        Y = X @ self.T + self.S @ self.F
        self.S = self.S @ self.ANT + X @ self.G
        return Y.astype(np.float32)


class OnePole:
    """y[n] = c*y[n-1] + (1-c)*x[n] on whole blocks via a log2(N)-step prefix scan (exact, no BLAS)."""

    def __init__(self, channels, n=BLOCK):
        self.n = n
        self.y = np.zeros((channels, 1), np.float32)
        self.c = None

    def reset(self):
        self.y[:] = 0

    def set(self, c):
        if c == self.c:
            return
        self.c = c
        self.steps = []
        k = 1
        while k < self.n:
            self.steps.append((k, np.float32(c ** k)))
            k *= 2
        self.decay = (c ** np.arange(1, self.n + 1)).astype(np.float32)

    def process(self, X):
        y = X.astype(np.float32) * np.float32(1 - self.c)
        for k, p in self.steps:
            y[:, k:] += p * y[:, :-k]
        y += self.decay * self.y
        self.y = y[:, -1:].copy()
        return y


def filter_coeffs(ftype, fc, reso):
    fc = clamp(fc, 20.0, SR * 0.45)
    q = 0.707 * (2.0 ** (reso * 4.5))
    w0 = 2 * math.pi * fc / SR
    c, s = math.cos(w0), math.sin(w0)
    al = s / (2 * q)
    g = (0.707 / q) ** 0.5
    if ftype == 1:
        b = [(1 - c) / 2 * g, (1 - c) * g, (1 - c) / 2 * g]
    elif ftype == 2:
        b = [(1 + c) / 2 * g, -(1 + c) * g, (1 + c) / 2 * g]
    else:
        b = [al, 0.0, -al]
    return b, [1 + al, -2 * c, 1 - al]


# ----------------------------------------------------------------------------- FX chain

FX_DEFAULTS = dict(drive=0.0, bits=16, downsample=1, ftype=0, cutoff=6000.0, reso=0.2,
                   dtime=0.35, dfb=0.35, dtone=0.6, dping=0, dmix=0.0,
                   vsize=0.6, vdamp=0.5, vmix=0.15, master=0.9,
                   drive_on=1, filter_on=1, delay_on=1, reverb_on=1)
# Each FX section has an on/off switch; switching off keeps the settings and skips the processing.
FX_SECTION_SWITCH = {"DRIVE": "drive_on", "FILTER": "filter_on", "DELAY": "delay_on", "REVERB": "reverb_on"}
FX_SWITCH = {k: flag for flag, keys in (("drive_on", ("drive", "bits", "downsample")),
                                        ("filter_on", ("ftype", "cutoff", "reso")),
                                        ("delay_on", ("dtime", "dfb", "dtone", "dping", "dmix")),
                                        ("reverb_on", ("vsize", "vdamp", "vmix"))) for k in keys}


OUTPUT_VOLUME = 1.0       # the Brick's volume buttons (app-wide, settings.json)
VOLUME_IN_ENGINE = False  # the native engine (v4) applies it after the looper; otherwise it scales the FX master


def fx_effective(f):
    """The FX settings as the engine should run them, with switched-off sections neutralised."""
    if f["drive_on"] and f["filter_on"] and f["delay_on"] and f["reverb_on"] and (VOLUME_IN_ENGINE or OUTPUT_VOLUME == 1.0):
        return f
    e = dict(f)
    e["master"] = f["master"] * (1.0 if VOLUME_IN_ENGINE else OUTPUT_VOLUME)
    if not f["drive_on"]:
        e.update(drive=0.0, bits=16, downsample=1)
    if not f["filter_on"]:
        e["ftype"] = 0
    if not f["delay_on"]:
        e["dmix"] = 0.0
    if not f["reverb_on"]:
        e["vmix"] = 0.0
    return e


# ----------------------------------------------------------------------------- native DSP
# native/bsdsp.c does the same work as the NumPy code below (voices + FX), about 10x cheaper.
# It's used when the library for this machine is present; otherwise the NumPy engine runs.

NATIVE_FX_FLAGS = ["drive_on", "filter_on", "delay_on", "reverb_on"]   # C fades sections in / out itself
FX_KEYS = ["drive", "bits", "downsample", "ftype", "cutoff", "reso", "dtime", "dfb", "dtone", "dping", "dmix",
           "vsize", "vdamp", "vmix", "master"]
NATIVE_DSP = True       # controls.json "native_dsp": false forces the NumPy engine
_native = [None, False]  # [lib, tried]


def native_lib():
    if _native[1]:
        return _native[0]
    _native[1] = True
    if not NATIVE_DSP:
        print("dsp: native disabled in controls.json", flush=True)
        return None
    import ctypes as C
    import platform
    m = platform.machine().lower()
    if sys.platform == "win32" and m in ("amd64", "x86_64"):
        name = "bsdsp_x64.dll"
    elif sys.platform.startswith("linux") and m in ("aarch64", "arm64"):
        name = "libbsdsp_aarch64.so"
    elif sys.platform.startswith("linux") and m in ("x86_64", "amd64"):
        name = "libbsdsp_x86_64.so"
    else:
        print(f"dsp: no native library for {sys.platform}/{m}, using NumPy", flush=True)
        return None
    path = os.path.join(HERE, "native", name)
    try:
        lib = C.CDLL(path)
        vp, ci, cf, cd = C.c_void_p, C.c_int, C.c_float, C.c_double
        lib.fx_create.restype, lib.fx_create.argtypes = vp, [ci]
        lib.fx_clear.restype, lib.fx_clear.argtypes = None, [vp]
        lib.fx_process.restype, lib.fx_process.argtypes = None, [vp, vp, vp, ci, vp]
        lib.voice_render.restype = ci
        lib.voice_render.argtypes = [vp, ci, ci, C.POINTER(cd), cd, ci, ci, cf, cf, cf, cf, cf, vp, vp, ci]
        if lib.bsdsp_version() >= 3:
            lib.eng_create.restype, lib.eng_create.argtypes = vp, [ci, ci]
            for fn, args in (("eng_note_on", [vp, ci, ci]), ("eng_note_off", [vp, ci]), ("eng_cc", [vp, ci, ci]),
                             ("eng_bend", [vp, ci]), ("eng_all_off", [vp]), ("eng_config", [vp, ci, ci, ci, cf]),
                             ("eng_set_slot_data", [vp, ci, vp, ci]), ("eng_set_slot_params", [vp, ci, vp]),
                             ("eng_set_fx", [vp, vp]), ("eng_render", [vp, vp]), ("eng_set_lfo", [vp, vp])):
                f = getattr(lib, fn)
                f.restype, f.argtypes = None, args
            lib.eng_status.restype, lib.eng_status.argtypes = ci, [vp, vp, ci]
            if hasattr(lib, "eng_start"):
                lib.eng_start.restype, lib.eng_start.argtypes = ci, [vp, C.c_char_p, ci]
                lib.eng_set_buffer.restype, lib.eng_set_buffer.argtypes = None, [vp, ci]
                lib.eng_pause.restype, lib.eng_pause.argtypes = ci, [vp, ci]
                for fn in ("eng_device", "eng_error"):
                    getattr(lib, fn).restype, getattr(lib, fn).argtypes = C.c_char_p, [vp]
                lib.eng_period.restype, lib.eng_period.argtypes = ci, [vp]
        if lib.bsdsp_version() >= 4:
            ll = C.c_longlong
            lib.eng_loop.restype, lib.eng_loop.argtypes = None, [vp, ci, ci]
            lib.eng_loop_params.restype, lib.eng_loop_params.argtypes = None, [vp, vp]
            lib.eng_loop_max.restype, lib.eng_loop_max.argtypes = ll, [vp]
            lib.eng_loop_data.restype, lib.eng_loop_data.argtypes = C.POINTER(C.c_float), [vp, ci]
            lib.eng_loop_status.restype, lib.eng_loop_status.argtypes = ci, [vp, vp, ci]
            lib.eng_loop_load.restype, lib.eng_loop_load.argtypes = None, [vp, ci, vp, ll, ll]
            lib.eng_loop_reset.restype, lib.eng_loop_reset.argtypes = None, [vp]
            lib.eng_in_push.restype, lib.eng_in_push.argtypes = None, [vp, vp, ci]
            lib.eng_mic.restype, lib.eng_mic.argtypes = None, [vp, ci]
            lib.eng_tap.restype, lib.eng_tap.argtypes = None, [vp, ci]
            lib.eng_tap_read.restype, lib.eng_tap_read.argtypes = ci, [vp, vp, ci]
            if hasattr(lib, "eng_cap_open"):
                lib.eng_cap_open.restype, lib.eng_cap_open.argtypes = ci, [vp, C.c_char_p, ci]
            if hasattr(lib, "eng_cap_format"):
                lib.eng_cap_format.restype, lib.eng_cap_format.argtypes = None, [vp, ci, ci]
                lib.eng_cap_level.restype, lib.eng_cap_level.argtypes = ci, [vp, C.POINTER(C.c_double)]
                lib.eng_cap_error.restype, lib.eng_cap_error.argtypes = C.c_char_p, [vp]
        print(f"dsp: native {name} (version {lib.bsdsp_version()})", flush=True)
        _native[0] = lib
    except Exception as e:
        print(f"dsp: native library not loaded ({e}), using NumPy", flush=True)
    return _native[0]


class NativeFX:
    def __init__(self, lib):
        self.lib = lib
        self.h = lib.fx_create(SR)
        self.p = np.zeros(len(FX_KEYS), np.float32)

    def clear(self):
        self.lib.fx_clear(self.h)

    def process(self, x, f):
        p = self.p
        for i, k in enumerate(FX_KEYS):
            p[i] = f[k]
        self.lib.fx_process(self.h, x[0].ctypes.data, x[1].ctypes.data, x.shape[1], p.ctypes.data)
        return x


class FXChain:
    def __init__(self):
        N = BLOCK
        self.ar = np.arange(N)
        self.filt = BlockLTI(2, 2, SUB)
        self.sub_ar = np.arange(SUB)
        self.f_on = False
        self.fkey = None
        self.f_wait = 0
        self.ds_count = 0
        self.ds_hold = np.zeros(2, np.float32)
        # delay
        self.dsize = 1 << 17
        self.dmask = self.dsize - 1
        self.dbuf = np.zeros((2, self.dsize), np.float32)
        self.dw = 0
        self.dlti = OnePole(2)
        self.d_active = False
        # reverb (Freeverb topology; all lines longer than one block)
        combs = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])
        spread = 23
        self.cd = np.concatenate([combs, combs + spread])[:, None]
        self.cbuf = np.zeros((16, 4096), np.float32)
        self.cw = 0
        self.clti = OnePole(16)
        self.rows16 = np.arange(16)[:, None]
        self.rows2 = np.arange(2)[:, None]
        self.aps = [(np.array([[d], [d + spread]]), np.zeros((2, 1024), np.float32)) for d in (556, 441, 341, 277)]
        self.aw = 0
        self.v_active = False

    def clear(self):
        """Silence every effect tail (used before the engine goes idle)."""
        self.filt.reset()
        self.dbuf.fill(0)
        self.dlti.reset()
        self.cbuf.fill(0)
        for _, buf in self.aps:
            buf.fill(0)
        self.clti.reset()
        self.d_active = self.v_active = False

    def process(self, x, f):
        N = BLOCK
        dr = f["drive"]
        if dr > 0.001:
            g = 1 + 30 * dr * dr
            x = np.tanh(x * g) * (0.4 + 0.6 / g)

        bits = int(f["bits"])
        if bits < 16:
            q = 2.0 ** (bits - 1)
            x = np.round(x * q) / q

        k = int(f["downsample"])
        if k > 1:
            gidx = self.ds_count + self.ar
            start = gidx - (gidx % k) - self.ds_count
            ok = start >= 0
            held = np.where(ok[None, :], x[:, np.clip(start, 0, None)], self.ds_hold[:, None])
            if start[-1] >= 0:
                self.ds_hold = x[:, start[-1]].copy()
            x = held.astype(np.float32)
            self.ds_count = (self.ds_count + N) % k

        ft = int(f["ftype"])
        if ft:
            if not self.f_on:
                self.filt.reset()
                self.f_on = True
            key = (ft, round(f["cutoff"], 1), round(f["reso"], 3))
            self.f_wait -= 1
            if key != self.fkey and (self.f_wait <= 0 or self.fkey is None or key[0] != self.fkey[0]):
                self.fkey = key
                self.f_wait = 3          # rebuild at most every 3 blocks while sweeping
                self.filt.set(*filter_coeffs(*key))
            x = np.concatenate([self.filt.process(x[:, i:i + SUB]) for i in range(0, N, SUB)], axis=1)
        else:
            self.f_on = False

        mix = f["dmix"]
        if mix > 0.0005:
            self.d_active = True
            d = int(clamp(int(f["dtime"] * SR), N, self.dsize - N - 1))
            reads = self.dbuf[:, (self.dw - d + self.ar) & self.dmask]
            c = 0.85 * (1 - f["dtone"])
            self.dlti.set(c)
            wet = self.dlti.process(reads)
            fb = min(f["dfb"], 0.95)
            if f["dping"]:
                mono = 0.5 * (x[0] + x[1])
                w = np.stack([mono + fb * wet[1], fb * wet[0]])
            else:
                w = x + fb * wet
            self.dbuf[:, (self.dw + self.ar) & self.dmask] = w
            self.dw = (self.dw + N) & self.dmask
            x = x + mix * wet
        elif self.d_active:
            self.dbuf.fill(0)
            self.dlti.reset()
            self.d_active = False

        vm = f["vmix"]
        if vm > 0.0005:
            self.v_active = True
            inp = (x[0] + x[1]) * 0.015
            fb = f["vsize"] * 0.28 + 0.7
            d1 = f["vdamp"] * 0.4
            self.clti.set(d1)
            reads = self.cbuf[self.rows16, (self.cw - self.cd + self.ar[None, :]) & 4095]
            filt = self.clti.process(reads)
            self.cbuf[:, (self.cw + self.ar) & 4095] = inp[None, :] + fb * filt
            self.cw = (self.cw + N) & 4095
            comb_out = np.stack([reads[:8].sum(0), reads[8:].sum(0)])
            parts = []
            sar = self.sub_ar
            for i in range(0, N, SUB):          # allpass lines are shorter than BLOCK
                out = comb_out[:, i:i + SUB]
                for dl, buf in self.aps:
                    bo = buf[self.rows2, (self.aw - dl + sar[None, :]) & 1023]
                    y = bo - out
                    buf[:, (self.aw + sar) & 1023] = out + bo * 0.5
                    out = y
                self.aw = (self.aw + SUB) & 1023
                parts.append(out)
            x = x + vm * 3.0 * np.concatenate(parts, axis=1)
        elif self.v_active:
            self.cbuf.fill(0)
            for _, buf in self.aps:
                buf.fill(0)
            self.clti.reset()
            self.v_active = False

        x = x * f["master"]
        ax = np.abs(x)
        over = ax > 0.8
        if over.any():
            x = np.where(over, np.sign(x) * (0.8 + 0.2 * np.tanh((ax - 0.8) / 0.2)), x)
        return x.astype(np.float32)


# ----------------------------------------------------------------------------- sampler

SLOT_DEFAULTS = dict(root=60, tune=0, fine=0, gain=0.8, pan=0.0, start=0.0, end=1.0, loop=0, reverse=0, trigger=0,
                     attack=0.002, decay=0.4, sustain=1.0, release=0.3, velsens=0.7,
                     lstart=0.0, lend=1.0, xfade=0.0)       # loop points (clamped to start..end), crossfade 0..0.5
# One global LFO (part of the project): pitch, volume and filter cutoff.
LFO_DEFAULTS = dict(wave=0, rate=5.0, delay=0.0, pitch=0.0, volume=0.0, cutoff=0.0, modwheel=0)
LFO_KEYS = ["wave", "rate", "delay", "pitch", "volume", "cutoff", "modwheel"]     # order of LP_* in bsdsp.c
# MIDI CC -> parameter (app-wide, in settings.json; learned with B on the FX and LFO pages)
DEFAULT_CCMAP = {"74": "cutoff", "71": "reso", "7": "master"}
FIXED_CCS = {1, 64, 120, 121, 123}      # mod wheel, sustain, all off: never learned
PAD_DEFAULTS = dict(SLOT_DEFAULTS, trigger=1, attack=0.0005, release=0.05)


class Slot:
    def __init__(self, kit):
        self.p = dict(PAD_DEFAULTS if kit else SLOT_DEFAULTS)
        self.data = None
        self.path = None
        self.name = "(empty)"
        self.peaks = None

    def load(self, path):
        self.set_data(load_wav(path), path)

    def set_data(self, data, path):
        self.data = np.ascontiguousarray(data, dtype=np.float32)   # the native DSP reads it directly
        self.path = path
        self.name = os.path.basename(path) if path else "(recording)"
        self.peaks = None


SQ2 = math.sqrt(2)


class Voice:
    def __init__(self, slot_idx, slot, note, vel, keytrack):
        p = slot.p
        self.slot_idx, self.slot, self.note, self.keytrack = slot_idx, slot, note, keytrack
        self.data = slot.data
        n = len(self.data)
        s = int(p["start"] * n)
        e = min(n, max(int(p["end"] * n), s + 2))
        s = min(s, e - 2)
        self.s, self.e, self.L = s, e, e - s
        self.reverse = bool(p["reverse"])
        self.oneshot = bool(p["trigger"])
        self.loop = bool(p["loop"]) and not self.oneshot
        self.pos = 0.0
        vs = p["velsens"]
        self.amp = p["gain"] * ((1 - vs) + vs * (vel / 127.0) ** 1.5)
        ang = (p["pan"] + 1) * math.pi / 4
        self.gl, self.gr = math.cos(ang) * SQ2, math.sin(ang) * SQ2
        self.stage, self.level, self.rel_rate = 0, 0.0, 1.0
        self.done = False
        self.sustained = False

    def release(self, fast=False):
        if self.stage >= 3 and not fast:
            return
        r = 0.004 if fast else max(self.slot.p["release"], 0.002)
        self.stage = 3
        self.rel_rate = max(self.level, 1e-4) / r

    def _env(self, n):
        p = self.slot.p
        t, lvl, st = n / SR, self.level, self.stage
        while t > 1e-12 and st < 4:
            if st == 0:
                rate = 1.0 / max(p["attack"], 0.0005)
                need = (1.0 - lvl) / rate
                if need <= t:
                    lvl, t, st = 1.0, t - need, 1
                else:
                    lvl, t = lvl + rate * t, 0
            elif st == 1:
                sus = p["sustain"]
                if lvl <= sus:
                    st = 2
                    continue
                rate = (1.0 - sus) / max(p["decay"], 0.001)
                need = (lvl - sus) / rate
                if need <= t:
                    lvl, t, st = sus, t - need, 2
                else:
                    lvl, t = lvl - rate * t, 0
            elif st == 2:
                lvl = p["sustain"]
                if lvl <= 1e-5:
                    lvl, st = 0.0, 4
                t = 0
            else:
                need = lvl / self.rel_rate
                if need <= t:
                    lvl, st = 0.0, 4
                else:
                    lvl, t = lvl - self.rel_rate * t, 0
        start = self.level
        self.level, self.stage = lvl, st
        return start, lvl

    def render(self, n, ar, ramp, bend):
        p = self.slot.p
        semis = p["tune"] + p["fine"] / 100.0
        if self.keytrack:
            semis += self.note - p["root"] + bend
        inc = 2.0 ** (semis / 12.0)
        L = self.L
        ps = self.pos + inc * ar
        valid = None
        if self.loop:
            ps = np.mod(ps, L)
            self.pos = (self.pos + inc * n) % L
            i0 = ps.astype(np.int64)
            i1 = i0 + 1
            i1[i1 >= L] = 0
        else:
            self.pos += inc * n
            valid = ps < (L - 1)
            i0 = np.minimum(ps.astype(np.int64), L - 1)
            i1 = np.minimum(i0 + 1, L - 1)
        frac = (ps - np.floor(ps)).astype(np.float32)
        if self.reverse:
            a, b = self.e - 1 - i0, self.e - 1 - i1
        else:
            a, b = self.s + i0, self.s + i1
        d = self.data
        da = d[a]
        sig = da + (d[b] - da) * frac
        e0, e1 = self._env(n)
        sig *= (e0 + (e1 - e0) * ramp) * self.amp
        if valid is not None and not valid[-1]:
            sig *= valid
            self.done = True
        if self.stage >= 4:
            self.done = True
        return sig

    def render_native(self, lib, mix, bend):
        import ctypes as C
        p = self.slot.p
        semis = p["tune"] + p["fine"] / 100.0
        if self.keytrack:
            semis += self.note - p["root"] + bend
        inc = 2.0 ** (semis / 12.0)
        e0, e1 = self._env(BLOCK)
        pos = C.c_double(self.pos)
        done = lib.voice_render(self.data.ctypes.data, self.s, self.e, C.byref(pos), inc, int(self.loop),
                                int(self.reverse), e0, e1, self.amp, self.gl, self.gr,
                                mix[0].ctypes.data, mix[1].ctypes.data, BLOCK)
        self.pos = pos.value
        if done or self.stage >= 4:
            self.done = True

    def playhead(self):
        n = len(self.data)
        p = self.pos if self.loop else min(self.pos, self.L - 1)
        idx = self.e - 1 - p if self.reverse else self.s + p
        return idx / n


LOOP_TRACKS = 5
LOOP_STATES = ["EMPTY", "REC", "PLAY", "DUB", "STOP"]
LC_RECDUB, LC_STOP, LC_UNDO, LC_CLEAR, LC_ALL, LC_CLEARALL = 1, 2, 3, 4, 5, 6
LOOP_DEFAULTS = dict(src=0, mon=1, recdub=0, dubmode=0, met=0, bpm=120, sig=2, click=70, countin=1)
TIME_SIGS = ["2/4", "3/4", "4/4", "5/4", "6/8", "7/8"]
SIG_BEATS = [2, 3, 4, 5, 6, 7]


class State:
    def __init__(self):
        self.looper = dict(LOOP_DEFAULTS, vols=[1.0] * LOOP_TRACKS)
        self.loop_tracks = [False] * LOOP_TRACKS      # which tracks have a saved WAV (read from the project)
        self.loop_len = 0
        self.mode = "chromatic"
        self.slots = [Slot(False)] + [Slot(True) for _ in range(NUM_PADS)]
        self.fx = dict(FX_DEFAULTS)
        self.lfo = dict(LFO_DEFAULTS)
        self.rec = dict(gain=1.0, normalize=1, rate=REC_RATES.index(SR) if SR in REC_RATES else 2, bits=1)
        self.sel_pad = 0
        self.kb_octave = 4

    def cur_index(self):
        return 0 if self.mode == "chromatic" else 1 + self.sel_pad

    def cur_slot(self):
        return self.slots[self.cur_index()]

    def to_json(self):
        def rel(p):
            if not p:
                return None
            try:
                r = os.path.relpath(p, HERE)
                return p if r.startswith("..") else r
            except ValueError:
                return p
        return dict(version=1, mode=self.mode, sel_pad=self.sel_pad, kb_octave=self.kb_octave,
                    fx=self.fx, lfo=self.lfo, rec=self.rec, slots=[dict(path=rel(s.path), p=s.p) for s in self.slots],
                    looper=dict(self.looper, tracks=self.loop_tracks, length=self.loop_len))

    def from_json(self, d):
        self.mode = d.get("mode", "chromatic")
        self.sel_pad = int(d.get("sel_pad", 0))
        self.kb_octave = int(d.get("kb_octave", 4))
        self.fx.update({k: v for k, v in d.get("fx", {}).items() if k in self.fx})
        self.rec.update({k: v for k, v in d.get("rec", {}).items() if k in self.rec})
        self.lfo.update({k: v for k, v in d.get("lfo", {}).items() if k in self.lfo})
        lo = d.get("looper", {})
        self.looper.update({k: int(v) for k, v in lo.items() if k in LOOP_DEFAULTS})
        vols = lo.get("vols", [])
        self.looper["vols"] = [float(clamp(float(v), 0, 1.5)) for v in vols][:LOOP_TRACKS] + \
            [1.0] * (LOOP_TRACKS - len(vols[:LOOP_TRACKS]))
        tr = [bool(v) for v in lo.get("tracks", [])][:LOOP_TRACKS]
        self.loop_tracks = tr + [False] * (LOOP_TRACKS - len(tr))
        self.loop_len = int(lo.get("length", 0))
        loaded = {}                 # chopped pads share one file: load it once
        for slot, sd in zip(self.slots, d.get("slots", [])):
            slot.p.update({k: v for k, v in sd.get("p", {}).items() if k in slot.p})
            path = sd.get("path")
            if path:
                full = path if os.path.isabs(path) else os.path.join(HERE, path)
                try:
                    if full not in loaded:
                        loaded[full] = load_wav(full)
                    slot.set_data(loaded[full], full)
                except Exception as e:
                    print("could not load", full, e)


class Engine:
    def __init__(self, state):
        self.state = state
        self.events = collections.deque()
        self.voices = []
        self.native = native_lib()
        self.fx = NativeFX(self.native) if self.native else FXChain()
        self.bend = 0.0
        self.sustain = False
        self.fifo = np.zeros((0, 2), np.float32)
        self.ar = np.arange(BLOCK, dtype=np.float64)
        self.ramp = ((np.arange(BLOCK) + 1) / BLOCK).astype(np.float32)
        self.load = 0.0
        self.quiet_blocks = 0
        self.idle = False
        # Idle output is +-1 LSB at Nyquist (about -90 dB, inaudible) rather than exact digital silence:
        # the Brick's codec mutes itself on true silence and pops when the first note unmutes it.
        self.zero_block = np.full((BLOCK, 2), 1.01 / 32767.0, np.float32)   # -> exactly 1 LSB in 16-bit
        self.zero_block[1::2] *= -1
        self.guard = False          # overload guard, only for the SDL callback output
        self.slow_blocks = 0
        self.overloads = 0
        self.peak = [0.0, 0.0]
        self.playheads = []
        self.pad_hits = [0.0] * NUM_PADS
        self.last_pad = None
        self.midi_activity = 0.0
        self.error = None
        self.ccmap, self.learn, self.learned, self.cc_seen = dict(DEFAULT_CCMAP), None, None, 0.0
        self.lfo_val, self.lfo_scale = 0.0, 0.0
        self.looper = False             # the looper lives in the native engine only
        self.has_capture = False

    # thread-safe entry points (UI and MIDI threads)
    def note_on(self, note, vel=100):
        self.events.append(("on", note, vel))

    def note_off(self, note):
        self.events.append(("off", note, 0))

    def cc(self, num, val):
        self.events.append(("cc", num, val))

    def pitch_bend(self, val14):
        self.events.append(("bend", val14, 0))

    def all_off(self):
        self.events.append(("alloff", 0, 0))

    def _handle(self, ev):
        kind, a, b = ev
        st = self.state
        if kind == "on":
            if st.mode == "chromatic":
                idx, keytrack = 0, True
            else:
                pad = a - KIT_BASE_NOTE
                if not 0 <= pad < NUM_PADS:
                    return
                idx, keytrack = 1 + pad, False
                self.pad_hits[pad] = time.time()
                self.last_pad = pad
            slot = st.slots[idx]
            if slot.data is None:
                return
            if not keytrack:
                for v in self.voices:
                    if v.slot_idx == idx:
                        v.release(fast=True)
            live = [v for v in self.voices if v.stage < 3]
            if len(live) >= MAX_VOICES:
                live[0].release(fast=True)
            if len(self.voices) >= MAX_VOICES + 4:
                self.voices.pop(0)
            self.voices.append(Voice(idx, slot, a, b, keytrack))
        elif kind == "off":
            for v in self.voices:
                if v.note == a and not v.oneshot and v.stage < 3:
                    if self.sustain:
                        v.sustained = True
                    else:
                        v.release()
        elif kind == "cc":
            fx = st.fx
            if a == 64:
                self.sustain = b >= 64
                if not self.sustain:
                    for v in self.voices:
                        if v.sustained:
                            v.sustained = False
                            v.release()
            elif apply_cc(self, a, b):
                pass
            elif a in (120, 123):
                for v in self.voices:
                    v.release(fast=(a == 120))
        elif kind == "bend":
            self.bend = (a - 8192) / 8192.0 * BEND_RANGE
        elif kind == "alloff":
            for v in self.voices:
                v.release(fast=True)
            self.sustain = False

    def render_block(self):
        while self.events:
            self._handle(self.events.popleft())
        # Idle: nothing playing and the effect tails have died away -> skip all DSP.
        fx = fx_effective(self.state.fx)
        if not self.voices:
            tail = max(0.5, fx["dtime"] * 1.5 if fx["dmix"] > 0.0005 else 0.0)
            if self.quiet_blocks * BLOCK / SR > tail:
                if not self.idle:
                    self.fx.clear()
                    self.idle = True
                self.peak = [0.0, 0.0]
                self.playheads = []
                return self.zero_block
        self.idle = False
        mix = np.zeros((2, BLOCK), np.float32)
        if self.native:
            for v in self.voices:
                v.render_native(self.native, mix, self.bend)
        else:
            for v in self.voices:
                sig = v.render(BLOCK, self.ar, self.ramp, self.bend)
                mix[0] += sig * v.gl
                mix[1] += sig * v.gr
        self.voices = [v for v in self.voices if not v.done]
        self.playheads = [(v.slot_idx, v.playhead()) for v in self.voices]
        out = self.fx.process(mix, fx)
        pk = np.abs(out).max(axis=1)
        self.quiet_blocks = 0 if (self.voices or pk.max() > 2e-4) else self.quiet_blocks + 1
        self.peak = [max(self.peak[0] * 0.9, float(pk[0])), max(self.peak[1] * 0.9, float(pk[1]))]
        return out.T

    def fill(self, frames):
        """Return `frames` stereo frames as an (frames, 2) float32 array."""
        t0 = time.thread_time()
        try:
            parts = [self.fifo]
            have = len(self.fifo)
            while have < frames:
                b = self.render_block()
                parts.append(b)
                have += len(b)
            buf = np.concatenate(parts) if len(parts) > 1 else self.fifo
            out, self.fifo = buf[:frames], buf[frames:]
        except Exception:
            self.error = traceback.format_exc().strip().splitlines()[-1]
            traceback.print_exc()
            self.voices = []
            out = np.zeros((frames, 2), np.float32)
        dt = time.thread_time() - t0      # CPU time of this thread only, not time spent waiting
        ratio = dt / (frames / SR)
        self.load = self.load * 0.9 + 0.1 * ratio
        # Overload guard (SDL callback output only): there the audio thread could otherwise hold
        # Python's lock all the time and freeze the screen. With aplay the pipe paces itself.
        if self.guard and ratio > 0.9:
            self.slow_blocks += 1
            if self.slow_blocks >= 3:
                self.slow_blocks = 0
                self.overloads += 1
                live = [v for v in self.voices if v.stage < 3]
                for v in live[: max(1, len(live) // 2)]:
                    v.release(fast=True)
                time.sleep(frames / SR * 0.5)
        else:
            self.slow_blocks = 0
        return out


SLOT_KEYS = ["root", "tune", "fine", "gain", "pan", "start", "end", "loop", "reverse", "trigger",
             "attack", "decay", "sustain", "release", "velsens", "lstart", "lend", "xfade"]  # order of SP_* in bsdsp.c


def apply_cc(engine, num, val):
    """A CC from the MIDI thread: learn it, or move the parameter it is mapped to.
    Returns True when it changed an FX/LFO value (the engine then pushes it at once)."""
    if num in FIXED_CCS:
        return False
    if engine.learn is not None:
        key = engine.learn
        for c in [c for c, k in engine.ccmap.items() if k == key]:
            del engine.ccmap[c]                                  # one CC per parameter
        engine.ccmap[str(num)] = key
        engine.learn = None
        engine.learned = (num, key)
        return False
    key = engine.ccmap.get(str(num))
    if key is None:
        return False
    p = CC_TARGETS.get(key)
    if p is None:
        return False
    d = engine.state.lfo if p.scope == "lfo" else engine.state.fx
    k, x = p.key, val / 127.0
    if p.kind == "switch":
        d[k] = 1 if val >= 64 else 0
    elif p.kind == "enum":
        d[k] = min(len(p.opts) - 1, int(x * len(p.opts)))
    elif p.kind == "exp":
        d[k] = p.lo * (p.hi / p.lo) ** x
    else:
        v = p.lo + (p.hi - p.lo) * x
        d[k] = int(round(v)) if p.step >= 1 else v
    engine.cc_seen = time.time()
    return True
NATIVE_BLOCK = 128          # the C engine's own block (2.7 ms at 48 kHz)


class NativeEngine:
    """Voices, envelopes and FX run in native/bsdsp.c. On the Brick the C side also owns the audio
    thread, so Python (and the screen drawing) can never hold up a block. Python only sends notes and
    copies parameters across (sync(), once per UI loop) and reads back a small status array."""

    def __init__(self, state, lib):
        import ctypes as C
        self.C, self.lib, self.state = C, lib, state
        self.h = lib.eng_create(SR, NATIVE_BLOCK)
        self.native = lib
        self.block = NATIVE_BLOCK
        self.out_buf = np.zeros((NATIVE_BLOCK, 2), np.float32)
        self.status = np.zeros(12 + 2 * 16, np.float32)
        self.p_buf = np.zeros(len(SLOT_KEYS), np.float32)
        self.fx_buf = np.zeros(len(FX_KEYS) + len(NATIVE_FX_FLAGS), np.float32)
        self.held = [None] * len(state.slots)      # keeps sample arrays alive while C reads them
        self.sent_p = [None] * len(state.slots)
        self.sent_fx = None
        self.sent_lfo = None
        self.lfo_buf = np.zeros(len(LFO_KEYS), np.float32)
        self.ccmap, self.learn, self.learned, self.cc_seen = dict(DEFAULT_CCMAP), None, None, 0.0
        self.lfo_val, self.lfo_scale = 0.0, 0.0
        self.sent_cfg = None
        self.fifo = np.zeros((0, 2), np.float32)
        self.voices, self.playheads, self.peak = [], [], [0.0, 0.0]
        self.load, self.idle, self.underruns, self.buffer_frames, self.rt, self.pcm_open = 0.0, True, 0, 0, 0, False
        self.pad_hits = [0.0] * NUM_PADS
        self.last_pad = None
        self.midi_activity = 0.0
        self.error = None
        self.guard = False
        self.overloads = 0
        self.lock = threading.Lock()
        # looper (bsdsp v4)
        self.looper = lib.bsdsp_version() >= 4
        self.has_capture = self.looper and hasattr(lib, "eng_cap_open")
        self.comp = (0, 0)                  # (pads, input) delay compensation in frames, set by the app
        self.loop_buf = np.zeros(13 + LOOP_TRACKS, np.float32)
        self.sent_loop = None
        self.lstat = np.zeros(20 + 12 * LOOP_TRACKS, np.float64)
        self.loop = dict(beat=0.0, morigin=0, beats=4, length=0, origin=0, now=0, in_peak=[0.0, 0.0], cap_open=False, cap_period=0, mic_on=False,
                         ring_target=0, in_under=0, tracks=[dict(state=0, len=0, pos=0, phase=0, undo=0, fresh=0,
                                                                 dub=0, peak=0.0, ver=0, closing=0, master=0)
                                                            for _ in range(LOOP_TRACKS)])
        self.loop_views = []
        if self.looper:
            global VOLUME_IN_ENGINE
            VOLUME_IN_ENGINE = True
            self.maxf = int(lib.eng_loop_max(self.h))
            for t in range(LOOP_TRACKS):
                self.loop_views.append(np.ctypeslib.as_array(lib.eng_loop_data(self.h, t), shape=(self.maxf, 2)))
        self.tap_buf = np.zeros(8192, np.float32)
        self.sync()

    # ---- looper
    def push_loop(self):
        lo, st = self.state.looper, self.state
        t = (float(self.comp[0]), float(self.comp[1]), 0.0, float(lo["mon"]), float(st.rec["gain"]),
             float(OUTPUT_VOLUME), float(lo["recdub"])) + tuple(float(v) for v in lo["vols"]) + \
            (float(lo["met"]), float(lo["bpm"]), float(SIG_BEATS[lo["sig"]]), lo["click"] / 100.0, float(lo["dubmode"]), float(lo["countin"]))
        if t != self.sent_loop:
            self.loop_buf[:] = t
            self.lib.eng_loop_params(self.h, self.loop_buf.ctypes.data)
            self.sent_loop = t

    def loop_cmd(self, cmd, track=0):
        if self.looper:
            self.lib.eng_loop(self.h, int(cmd), int(track))

    def loop_reset(self):
        if self.looper:
            self.lib.eng_loop_reset(self.h)

    def loop_load(self, track, x, length):
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.lib.eng_loop_load(self.h, int(track), x.ctypes.data, len(x), int(length))

    def loop_audio(self, track):
        """The track's loop as (frames, 2), lined up so frame 0 is the start of the loop (a copy)."""
        tr = self.loop["tracks"][track]
        n = int(tr["len"])
        if n <= 0:
            return None
        x = self.loop_views[track][:n].copy()
        k = (int(self.loop["origin"]) - int(tr["phase"])) % n
        return np.roll(x, -k, axis=0) if k else x

    def set_capture(self, dev, mode):
        if self.has_capture:
            self.lib.eng_cap_open(self.h, (dev or "").encode(), int(mode))

    def cap_format(self, fmt, rate):
        """the format and rate for the next set_capture (the device's own; S16_LE / engine rate as fallback)"""
        if self.has_capture and hasattr(self.lib, "eng_cap_format"):
            self.lib.eng_cap_format(self.h, ALSA_FORMATS.get(fmt, 2), int(rate or 0))

    def cap_level(self):
        """(peak L, peak R, any non-zero sample) since the last call, raw from the card"""
        if not (self.has_capture and hasattr(self.lib, "eng_cap_level")):
            return None
        import ctypes as C
        o = (C.c_double * 5)()
        self.lib.eng_cap_level(self.h, o)
        return o[0], o[1], bool(o[2])

    def set_mic(self, on):
        if self.looper:
            self.lib.eng_mic(self.h, 1 if on else 0)

    def capture_error(self):
        return self.lib.eng_cap_error(self.h).decode(errors="replace") if self.has_capture else ""

    def tap(self, on):
        if self.looper:
            self.lib.eng_tap(self.h, 1 if on else 0)

    def tap_read(self):
        n = self.lib.eng_tap_read(self.h, self.tap_buf.ctypes.data, len(self.tap_buf))
        return self.tap_buf[:n].copy()

    def refresh_loop(self):
        if not self.looper:
            return
        o = self.lstat
        self.lib.eng_loop_status(self.h, o.ctypes.data, len(o))
        lp = self.loop
        lp["length"], lp["origin"], lp["now"] = int(o[0]), int(o[1]), int(o[2])
        lp["in_peak"] = [float(o[3]), float(o[4])]
        lp["cap_open"], lp["cap_period"], lp["in_under"], lp["ring_target"] = bool(o[5]), int(o[6]), int(o[8]), int(o[10])
        lp["mic_on"] = o[11] >= 1
        lp["cap_reads"], lp["cap_peak"], lp["cap_ch"] = int(o[12]), float(o[13]), int(o[14])
        lp["beat"], lp["morigin"], lp["beats"] = float(o[15]), int(o[16]), int(o[17])
        lp["drift"] = float(o[19])
        for i, tr in enumerate(lp["tracks"]):
            q = o[20 + 12 * i: 32 + 12 * i]
            tr.update(state=int(q[0]), len=int(q[1]), pos=int(q[2]), phase=int(q[3]), undo=int(q[4]), fresh=int(q[5]),
                      dub=int(q[6]), peak=float(q[7]), ver=int(q[8]), closing=int(q[10]), master=int(q[11]))

    # ---- parameters: copied to C when they change
    def sync(self):
        with self.lock:
            st, lib, h = self.state, self.lib, self.h
            cfg = (st.mode == "kit",)
            if cfg != self.sent_cfg:
                lib.eng_config(h, int(cfg[0]), MAX_VOICES, KIT_BASE_NOTE, BEND_RANGE)
                self.sent_cfg = cfg
            for i, slot in enumerate(st.slots):
                p = tuple(float(slot.p[k]) for k in SLOT_KEYS)
                if p != self.sent_p[i]:
                    self.p_buf[:] = p
                    lib.eng_set_slot_params(h, i, self.p_buf.ctypes.data)
                    self.sent_p[i] = p
                if slot.data is not self.held[i]:
                    d = slot.data
                    lib.eng_set_slot_data(h, i, d.ctypes.data if d is not None else None,
                                          len(d) if d is not None else 0)
                    self.held[i] = d
            self.push_fx()
            if self.looper:
                self.push_loop()
        self.refresh()
        self.refresh_loop()

    def push_fx(self):
        # raw settings plus the four on/off switches: the engine fades a section in or out (no click), and
        # delay / reverb switched off only close their input, so the echoes and the tail ring out
        f = self.state.fx
        t = tuple(float(f[k]) for k in FX_KEYS) + tuple(1.0 if f[k] else 0.0 for k in NATIVE_FX_FLAGS)
        if t != self.sent_fx:
            self.fx_buf[:] = t
            self.lib.eng_set_fx(self.h, self.fx_buf.ctypes.data)
            self.sent_fx = t
        t = tuple(float(self.state.lfo[k]) for k in LFO_KEYS)
        if t != self.sent_lfo:
            self.lfo_buf[:] = t
            self.lib.eng_set_lfo(self.h, self.lfo_buf.ctypes.data)
            self.sent_lfo = t

    def refresh(self):
        s = self.status
        n = self.lib.eng_status(self.h, s.ctypes.data, len(s))
        nv = int(s[0])
        self.peak = [float(s[1]), float(s[2])]
        self.load, self.idle, self.underruns = float(s[3]), bool(s[4]), int(s[5])
        self.buffer_frames, self.rt, self.pcm_open = int(s[6]), int(s[7]), bool(s[8])
        self.lfo_val, self.lfo_scale = float(s[10]), float(s[11])
        self.playheads = [(int(s[i]), float(s[i + 1])) for i in range(12, n, 2)]
        self.voices = [i for i, _ in self.playheads] if len(self.playheads) == nv else list(range(nv))

    # ---- events (any thread)
    def note_on(self, note, vel=100):
        if self.state.mode == "kit":
            pad = note - KIT_BASE_NOTE
            if 0 <= pad < NUM_PADS:
                self.pad_hits[pad] = time.time()
                self.last_pad = pad
        self.lib.eng_note_on(self.h, int(note), int(vel))

    def note_off(self, note):
        self.lib.eng_note_off(self.h, int(note))

    def cc(self, num, val):
        if apply_cc(self, num, val):
            with self.lock:
                self.push_fx()                       # filter sweeps reach the sound at once
        else:
            self.lib.eng_cc(self.h, int(num), int(val))

    def pitch_bend(self, val14):
        self.lib.eng_bend(self.h, int(val14))

    def all_off(self):
        self.lib.eng_all_off(self.h)

    # ---- pulled output (desktop SDL, aplay fallback, tests)
    def render_block(self):
        self.lib.eng_render(self.h, self.out_buf.ctypes.data)
        return self.out_buf.copy()

    def fill(self, frames):
        t0 = time.thread_time()
        parts, have = [self.fifo], len(self.fifo)
        while have < frames:
            b = self.render_block()
            parts.append(b)
            have += len(b)
        buf = np.concatenate(parts)
        out, self.fifo = buf[:frames], buf[frames:]
        self.load = self.load * 0.9 + 0.1 * (time.thread_time() - t0) / (frames / SR)
        return out


def make_engine(state):
    lib = native_lib()
    if lib is not None and lib.bsdsp_version() >= 3:
        return NativeEngine(state, lib)
    print("engine: Python fallback (no loop points, crossfade or LFO)", flush=True)
    return Engine(state)


class NativeAlsaOut:
    """The C engine's own writer thread (real-time priority) feeds ALSA directly."""
    FRAMES = [384, 768, 1536]            # Low / Mid / Safe buffer: 8 / 16 / 32 ms at 48 kHz

    def __init__(self, engine, latency=0):
        self.engine, self.lib, self.h = engine, engine.lib, engine.h
        self.mode = "alsa-native"
        self.latency = latency
        devices = ["plughw:0,0", "default"] if ALSA_DEVICE in ("default", "auto") else [ALSA_DEVICE, "default"]
        if not self.lib.eng_start(self.h, "\n".join(devices).encode(), self.FRAMES[latency]):
            raise OSError("libasound not usable")
        self.error = None
        self.paused = False
        self.t0 = time.time()
        self.reported = (None, 0)
        threading.Thread(target=self._watch, daemon=True).start()

    @property
    def ok(self):
        # The card is closed for a moment while the buffer size changes: only a gap of seconds is an error.
        now = time.time()
        if self.engine.pcm_open:
            self.t0 = now
        return self.paused or now - self.t0 < 3

    @property
    def buffer_ms(self):
        return self.engine.buffer_frames / SR * 1000

    def set_latency(self, idx):
        self.latency = idx
        self.lib.eng_set_buffer(self.h, self.FRAMES[idx])

    def suspend(self):
        self.paused = True
        ok = self.lib.eng_pause(self.h, 1)
        print("audio: output paused" + ("" if ok else " (timeout)"), flush=True)

    def resume(self):
        self.lib.eng_pause(self.h, 0)
        self.paused = False
        self.t0 = time.time()
        print("audio: output resumed", flush=True)

    def _watch(self):
        """Log what the C thread did: device and buffer, underruns, errors."""
        last_err, last_u, last_t = "", 0, 0.0
        while True:
            time.sleep(1.0)
            e = self.engine
            e.refresh()
            key = (self.lib.eng_device(self.h).decode(errors="replace"), e.buffer_frames)
            if e.pcm_open and key != self.reported[0]:
                self.reported = (key, 0)
                print(f"audio: ALSA {key[0]} (native thread, realtime {'yes' if e.rt else 'no'}), buffer "
                      f"{key[1]} frames ({key[1] / SR * 1000:.1f} ms), period {self.lib.eng_period(self.h)}, "
                      f"block {NATIVE_BLOCK}", flush=True)
            err = self.lib.eng_error(self.h).decode(errors="replace")
            self.error = (err or "no ALSA output device") if not self.ok else None
            if err and err != last_err:
                print("audio:", err, flush=True)
            last_err = err
            if e.underruns != last_u and time.time() - last_t > 10:
                print(f"audio: {e.underruns} underruns so far (buffer {self.buffer_ms:.1f} ms, cpu {e.load * 100:.0f}%)",
                      flush=True)
                last_u, last_t = e.underruns, time.time()


# ----------------------------------------------------------------------------- audio I/O

class AudioOut:
    """Uses the SDL mixer's post-mix hook as a plain audio callback (one device, no extra deps)."""

    def __init__(self, engine):
        self.engine = engine
        self.ok = False
        self.error = None
        self.mode = None
        try:
            pygame.mixer.pre_init(SR, 32, 2, DEVICE_CHUNK, allowedchanges=0)
            pygame.mixer.init()
            freq, fmt, ch = pygame.mixer.get_init()
            print(f"audio: mixer {freq} Hz, format {fmt}, {ch} ch, buffer {DEVICE_CHUNK}")
            if fmt != -32 or ch != 2 or freq != SR:
                raise RuntimeError(f"unexpected audio format {freq}/{fmt}/{ch}")
            try:
                from pygame._sdl2.mixer import set_post_mix
                set_post_mix(self._cb)
                engine.guard = True
                self.mode = "callback"
            except ImportError:
                # older pygame: stream blocks through a mixer channel instead (a little more latency)
                threading.Thread(target=self._stream, daemon=True).start()
                self.mode = "queue"
            print("audio: output mode", self.mode)
            self.ok = True
        except Exception as e:
            self.error = str(e)
            print("audio output unavailable:", e)

    def _stream(self):
        chan = pygame.mixer.Channel(0)
        while True:
            if chan.get_queue() is None:
                blk = np.ascontiguousarray(self.engine.fill(DEVICE_CHUNK), dtype=np.float32)
                snd = pygame.mixer.Sound(buffer=blk.tobytes())
                if chan.get_busy():
                    chan.queue(snd)
                else:
                    chan.play(snd)
            else:
                time.sleep(0.002)

    def _cb(self, _pm, mv):
        a = np.asarray(mv).view(np.float32)
        a[:] = self.engine.fill(len(a) // 2).reshape(-1)



# aplay/arecord need no special cleanup: if this app dies, their pipe breaks and they exit.


def use_alsa_tools():
    if AUDIO_BACKEND == "aplay":
        return True
    if AUDIO_BACKEND == "sdl":
        return False
    return ON_DEVICE and shutil.which("aplay") is not None


class AplayOut:
    """Streams 16-bit PCM to ALSA's `aplay` (the way GrainBrick does it on KNULLI). SDL audio is never opened."""

    def __init__(self, engine):
        self.engine = engine
        self.ok = False
        self.error = None
        self.mode = "aplay"
        self.proc = None
        self.chunk = DEVICE_CHUNK
        self.paused = threading.Event()
        self.lock = threading.Lock()
        try:
            self._start()
            threading.Thread(target=self._run, daemon=True).start()
            self.ok = True
        except Exception as e:
            self.error = str(e)
            print("audio output unavailable:", e, flush=True)

    def _start(self):
        exe = shutil.which("aplay")
        if not exe:
            raise RuntimeError("aplay not found")
        buf_us = int(self.chunk * 4 / SR * 1e6)     # 4 periods (~93 ms at 1024): headroom for screen redraws
        cmd = [exe, "-q", "-D", ALSA_DEVICE, "-t", "raw", "-f", "S16_LE", "-c", "2", "-r", str(SR),
               f"--buffer-time={buf_us}", f"--period-time={buf_us // 4}"]
        print("audio:", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)
        try:
            import fcntl
            fcntl.fcntl(self.proc.stdin.fileno(), 1031, max(4096, self.chunk * 4))   # F_SETPIPE_SZ: keep latency low
        except Exception as e:
            print("audio: pipe size unchanged:", e, flush=True)

    def suspend(self):
        """Stop playback and free the sound card (used when the mic can't run alongside it)."""
        self.paused.set()
        with self.lock:
            proc, self.proc = self.proc, None
        if proc:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        print("audio: output paused", flush=True)

    def resume(self):
        with self.lock:
            self._start()
        self.paused.clear()
        print("audio: output resumed", flush=True)

    def _run(self):
        restarts = 0
        period = self.chunk / SR
        last_log = 0.0
        while True:
            if self.paused.is_set():
                time.sleep(0.02)
                continue
            w0, c0, was_idle = time.perf_counter(), time.thread_time(), self.engine.idle
            out = self.engine.fill(self.chunk)
            wall, cpu = time.perf_counter() - w0, time.thread_time() - c0
            # Diagnostics: a block that takes most of its period risks an underrun. Say whether it was
            # our DSP (cpu ~ wall) or this thread waiting for Python's lock while the screen drew (wall >> cpu).
            if wall > period * 0.7 and time.time() - last_log > 0.5:
                last_log = time.time()
                print(f"audio: slow block {wall * 1000:.0f} ms of {period * 1000:.0f} (dsp {cpu * 1000:.0f} ms, "
                      f"voices {len(self.engine.voices)}, from_idle {was_idle and not self.engine.idle}, "
                      f"last draw {DRAW_MS[0]:.0f} ms)", flush=True)
            pcm = (np.clip(out, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            try:
                self.proc.stdin.write(pcm)
            except (BrokenPipeError, OSError, ValueError, AttributeError) as e:
                if self.paused.is_set():
                    continue
                restarts += 1
                print(f"audio: aplay stopped ({e}), exit {self.proc.poll()}", flush=True)
                if restarts > 5:
                    self.ok, self.error = False, "aplay keeps stopping"
                    return
                time.sleep(0.5)
                try:
                    self._start()
                except Exception as e2:
                    self.ok, self.error = False, str(e2)
                    return


# Output buffer sizes for SETTINGS > AUDIO > LATENCY, in DSP blocks (256 frames = 5.3 ms at 48 kHz)
LATENCY_NAMES = ["Low", "Mid", "Safe"]
LATENCY_BLOCKS = [3, 6, 12]


class AlsaOut:
    """Writes straight to ALSA through libasound (no aplay process, no pipe), with a small buffer so a
    note reaches the speaker a few milliseconds after the key. The buffer size can be changed live."""

    def __init__(self, engine, latency=0):
        import ctypes as C
        self.C = C
        lib = C.CDLL("libasound.so.2")
        vp = C.c_void_p
        lib.snd_pcm_open.argtypes = [C.POINTER(vp), C.c_char_p, C.c_int, C.c_int]
        lib.snd_pcm_set_params.argtypes = [vp, C.c_int, C.c_int, C.c_uint, C.c_uint, C.c_int, C.c_uint]
        lib.snd_pcm_get_params.argtypes = [vp, C.POINTER(C.c_ulong), C.POINTER(C.c_ulong)]
        lib.snd_pcm_writei.argtypes = [vp, vp, C.c_ulong]
        lib.snd_pcm_writei.restype = C.c_long
        lib.snd_pcm_recover.argtypes = [vp, C.c_int, C.c_int]
        lib.snd_pcm_close.argtypes = [vp]
        lib.snd_pcm_drop.argtypes = [vp]
        lib.snd_strerror.argtypes = [C.c_int]
        lib.snd_strerror.restype = C.c_char_p
        self.lib = lib
        self.engine = engine
        self.mode = "alsa"
        self.ok, self.error = True, None
        self.pcm = None
        self.device = None
        self.latency = latency
        self.buffer_frames = 0
        self.underruns = 0
        self.paused = threading.Event()
        self.closed = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def buffer_ms(self):
        return self.buffer_frames / SR * 1000

    def set_latency(self, idx):
        self.latency = idx           # the writer thread notices and reopens with the new buffer

    def _open(self):
        C, lib = self.C, self.lib
        want_us = int(LATENCY_BLOCKS[self.latency] * BLOCK / SR * 1e6)
        devices = ["plughw:0,0", "default"] if ALSA_DEVICE in ("default", "auto") else [ALSA_DEVICE, "default"]
        quiet = getattr(self, "_open_fails", 0) > 2              # don't flood the log if the card is gone
        for dev in devices:
            pcm = C.c_void_p()
            err = lib.snd_pcm_open(C.byref(pcm), dev.encode(), 0, 0)          # playback, blocking
            if err < 0:
                if not quiet:
                    print(f"audio: {dev}: {lib.snd_strerror(err).decode()}", flush=True)
                continue
            err = lib.snd_pcm_set_params(pcm, 2, 3, 2, SR, 1, want_us)       # S16_LE, interleaved, stereo
            if err < 0:
                print(f"audio: {dev} set_params: {lib.snd_strerror(err).decode()}", flush=True)
                lib.snd_pcm_close(pcm)
                continue
            buf, per = C.c_ulong(), C.c_ulong()
            lib.snd_pcm_get_params(pcm, C.byref(buf), C.byref(per))
            self.pcm, self.device, self.buffer_frames = pcm, dev, buf.value
            self.opened_latency = self.latency
            print(f"audio: ALSA {dev}, buffer {buf.value} frames ({buf.value / SR * 1000:.1f} ms), "
                  f"period {per.value}, block {BLOCK}", flush=True)
            return True
        return False

    def _close(self):
        if self.pcm is not None:
            self.lib.snd_pcm_drop(self.pcm)
            self.lib.snd_pcm_close(self.pcm)
            self.pcm = None

    def suspend(self):
        """Stop playback and free the sound card (the Brick can't record while it plays)."""
        self.closed.clear()
        self.paused.set()
        self.closed.wait(1.0)
        print("audio: output paused", flush=True)

    def resume(self):
        self.paused.clear()
        print("audio: output resumed", flush=True)

    def _run(self):
        failures, last_report = 0, time.time()
        period = BLOCK / SR
        while True:
            if self.paused.is_set():
                if self.pcm is not None:
                    self._close()
                self.closed.set()
                time.sleep(0.01)
                continue
            if self.pcm is not None and self.latency != self.opened_latency:
                self._close()
            if self.pcm is None:
                if not self._open():
                    failures += 1
                    self._open_fails = failures
                    self.ok, self.error = False, "no ALSA output device"
                    time.sleep(0.5)
                    continue
                self.ok, self.error = True, None
                failures = self._open_fails = 0
            w0, c0 = time.perf_counter(), time.thread_time()
            out = self.engine.fill(BLOCK)
            wall, cpu = time.perf_counter() - w0, time.thread_time() - c0
            pcm = np.ascontiguousarray((np.clip(out, -1.0, 1.0) * 32767.0).astype("<i2"))
            ptr, done = pcm.ctypes.data, 0
            while done < BLOCK:
                n = self.lib.snd_pcm_writei(self.pcm, ptr + done * 4, BLOCK - done)   # releases the GIL
                if n < 0:
                    self.underruns += 1
                    if self.lib.snd_pcm_recover(self.pcm, int(n), 1) < 0:
                        print(f"audio: write failed ({self.lib.snd_strerror(int(n)).decode()}), reopening", flush=True)
                        self._close()
                        break
                    continue
                done += n
            if wall > period * 0.8 and time.time() - last_report > 1:
                print(f"audio: slow block {wall * 1000:.1f} ms of {period * 1000:.1f} (dsp {cpu * 1000:.1f} ms, "
                      f"last draw {DRAW_MS[0]:.0f} ms)", flush=True)
                last_report = time.time()
            if self.underruns and time.time() - last_report > 10:
                print(f"audio: {self.underruns} underruns so far (buffer {self.buffer_ms:.1f} ms)", flush=True)
                last_report = time.time()


def make_audio_out(engine, latency=0):
    if use_alsa_tools():
        if AUDIO_BACKEND != "aplay" and isinstance(engine, NativeEngine) and hasattr(engine.lib, "eng_start"):
            try:
                return NativeAlsaOut(engine, latency)
            except OSError as e:
                print("audio: native ALSA thread not usable:", e, flush=True)
        if AUDIO_BACKEND != "aplay":
            try:
                return AlsaOut(engine, latency)
            except OSError as e:
                print("audio: libasound not usable, falling back to aplay:", e, flush=True)
        return AplayOut(engine)
    return AudioOut(engine)


ASOUND_CARDS = "/proc/asound/cards"
REC_CHANNELS = ["1", "2", "1+2"]


def card_index(cid):
    """ALSA card number of a card id (e.g. "Device" -> 1), or None."""
    try:
        with open(ASOUND_CARDS) as f:
            for ln in f.read().splitlines():
                m = re.match(r"\s*(\d+)\s+\[(\S+)\s*\]", ln)
                if m and m.group(2) == cid:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def capture_cards():
    """[(card id, name)] of the extra sound cards that can record (USB interfaces, USB mics, synths with
    USB audio), read from /proc/asound. Card 0 is the Brick's own codec: that one is the internal mic."""
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
        if n == 0:
            continue
        if not glob.glob(os.path.join(base, f"card{n}", "pcm*c")):
            continue                                  # playback only
        name = desc.split(" - ", 1)[1] if " - " in desc else desc
        out.append((cid, " ".join(name.split()) or cid))
    return out


def capture_devices(cid):
    """Recording devices of a USB card, as device numbers ([0], or [0, 1] for an adapter with a mic in and
    a line in...), from /proc/asound/cardN/pcmXc. [] when the card is not plugged in."""
    n = card_index(cid)
    if n is None:
        return []
    out = []
    for d in glob.glob(os.path.join(os.path.dirname(ASOUND_CARDS), f"card{n}", "pcm*c")):
        m = re.match(r"pcm(\d+)c$", os.path.basename(d))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def capture_stream(n, dev):
    """What a USB card says about recording on one device (/proc/asound/cardN/streamDEV):
    (formats, channels, rates). rates are (low, high) pairs; a fixed rate is (r, r). Empty when unknown."""
    fmts, chans, rates = [], [], []
    try:
        with open(os.path.join(os.path.dirname(ASOUND_CARDS), f"card{n}", f"stream{dev}")) as f:
            txt = f.read()
    except OSError:
        return fmts, chans, rates
    if "Capture:" not in txt:
        return fmts, chans, rates
    body = txt.split("Capture:", 1)[1].split("Playback:", 1)[0]
    for ln in body.splitlines():
        k, _, v = ln.strip().partition(":")
        if k == "Format":
            fmts += [x for x in re.split(r"[,\s]+", v.strip()) if x and x not in fmts]
        elif k == "Channels":
            chans += [int(x) for x in re.findall(r"\d+", v) if int(x) not in chans]
        elif k == "Rates":
            nums = [int(x) for x in re.findall(r"\d+", v)]
            if "continuous" in v and len(nums) >= 2:
                rates.append((nums[0], nums[1]))
            else:
                rates += [(r, r) for r in nums if (r, r) not in rates]
    return fmts, chans, rates


def pick_rate(rates, want=48000):
    """48000 when the device has it, else the first rate it lists (None when it lists none)."""
    if any(lo <= want <= hi for lo, hi in rates):
        return want
    return rates[0][0] if rates else None


ALSA_FORMATS = {"S16_LE": 2, "S24_LE": 6, "S32_LE": 10, "S24_3LE": 32}


def pick_format(fmts):
    """The device's own best capture format (some USB line inputs only send real audio in 24-bit)."""
    for f in ("S24_3LE", "S32_LE", "S24_LE", "S16_LE"):
        if f in fmts:
            return f
    return "S16_LE"


def usb_mixer_setup(n):
    """Log the USB card's mixer, then switch on any capture switch that is off and give a capture volume
    left at 0% a neutral gain (0 dB, or 80% without a dB scale). A computer often does this by itself; on
    the Brick nothing does, and the input then sends only zeros. Nothing else is touched."""
    exe = shutil.which("amixer")
    if not exe:
        print("loop: amixer not found, USB mixer left as it is", flush=True)
        return
    try:
        out = subprocess.run([exe, "-c", str(n), "scontents"], capture_output=True, text=True, timeout=3).stdout
    except Exception as ex:
        print(f"loop: amixer -c {n}: {ex}", flush=True)
        return
    print(f"loop: amixer -c {n} scontents:\n" + (out.rstrip() or "(no controls)"), flush=True)
    for blk in re.split(r"(?m)^Simple mixer control ", out)[1:]:
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
            todo.append(["capture", "0dB"] if any("dB]" in v for v in vals) else ["capture", "80%"])
        for args in todo:
            r = subprocess.run([exe, "-c", str(n), "-q", "sset", name] + args, capture_output=True, text=True, timeout=3)
            if r.returncode != 0 and args[-1] == "0dB":
                args = ["capture", "80%"]
                r = subprocess.run([exe, "-c", str(n), "-q", "sset", name] + args, capture_output=True, text=True,
                                   timeout=3)
            print(f"loop: mixer card {n} '{name}' {' '.join(args)}: " + ("ok" if r.returncode == 0 else r.stderr.strip()),
                  flush=True)


def read_hw_params(n, dev):
    """What the card really runs at once it is open (format, channels, rate, period, buffer), one line."""
    try:
        with open(os.path.join(os.path.dirname(ASOUND_CARDS), f"card{n}", f"pcm{dev}c", "sub0", "hw_params")) as f:
            d = dict(ln.split(":", 1) for ln in f.read().splitlines() if ":" in ln)
    except (OSError, ValueError):
        return ""
    d = {k.strip(): v.strip() for k, v in d.items()}
    rate = d.get("rate", "?").split()[0]
    return (f"format {d.get('format', '?')}, {d.get('channels', '?')} ch, {rate} Hz, "
            f"period {d.get('period_size', '?')}, buffer {d.get('buffer_size', '?')} frames")


class Recorder:
    def __init__(self, state):
        self.state = state
        self.dev = None
        self.recording = False
        self.chunks = []
        self.peaks = []
        self.frames = 0
        self.level = 0.0
        self.error = None
        self.device_name = None
        self.out = None             # audio output, set by the app (needed for half-duplex recording)
        self.half_duplex = False
        self.starting = False
        self.start_failed = False
        self.source = "internal"    # or the ALSA card id of a USB interface / mic / synth
        self.channel = 2            # USB: input 1, 2 or both mixed (index into REC_CHANNELS)
        self.engine = None          # set by the app: with the native engine, USB input comes from its capture

    @property
    def usb(self):
        return self.source != "internal"

    @property
    def tap_mode(self):
        """USB input read from the engine's own capture (the looper uses the same input, so the device is
        opened once, in C). Only the internal mic still goes through arecord."""
        return self.usb and self.engine is not None and getattr(self.engine, "has_capture", False)

    def _start_tap(self):
        self.engine.tap(True)
        self.dev = "tap"
        self.device_name = "engine capture"

        def reader():
            while self.dev == "tap":
                x = self.engine.tap_read()
                if len(x):
                    self._process(x)
                else:
                    time.sleep(0.01)
        threading.Thread(target=reader, daemon=True).start()

    def set_source(self, source, channel):
        """Switch input. A USB card stays open (live meter, speaker keeps playing); the internal mic
        is only opened while recording, as before."""
        self.channel = channel
        if source == self.source:
            return
        proc, self.dev = self.dev, None
        if proc == "tap":
            self.engine.tap(False)
        elif proc is not None and hasattr(proc, "terminate"):
            self._kill(proc)
        self.source, self.half_duplex, self.error, self.level = source, False, None, 0.0
        print(f"rec: source -> {source}", flush=True)

    def open_async(self):
        if self.starting:
            return
        self.starting = True

        def run():
            try:
                self.open()
            finally:
                self.starting = False
        threading.Thread(target=run, daemon=True).start()

    def _open_sdl(self):
        try:
            from pygame._sdl2 import audio as sdl_audio
            names = sdl_audio.get_audio_device_names(True)
            if not names:
                raise RuntimeError("no input device")
            self.device_name = names[0]
            self.dev = sdl_audio.AudioDevice(devicename=names[0], iscapture=True, frequency=SR,
                                             audioformat=sdl_audio.AUDIO_F32, numchannels=1,
                                             chunksize=512, allowed_changes=0, callback=self._cb)
            self.dev.pause(0)
            self.error = None
            return True
        except Exception as e:
            self.error = str(e)
            self.dev = None
            return False

    def _capture_candidates(self):
        """(device, channels) to try. hw:0,0 mono is what GrainBrick uses for the Brick's built-in mic."""
        # Never 'default' for capture on KNULLI: it goes through ALSA's dsnoop, which leaves a background
        # arecord holding the mic, and every later attempt then fails with "Device or resource busy".
        if self.usb:
            # plughw converts rate and format, and turns a mono mic into 2 channels
            d = f"CARD={self.source},DEV=0"
            return [(f"plughw:{d}", 2), (f"plughw:{d}", 1), (f"hw:{d}", 2)]
        c = [("hw:0,0", 1), ("plughw:0,0", 1), ("hw:0,0", 2)]
        if ALSA_CAPTURE not in ("default", "hw:0,0", "plughw:0,0"):
            c.insert(0, (ALSA_CAPTURE, 1))
        return c

    def _try_arecord(self, dev, ch, wait=1.0):
        """Start arecord on one device. Returns the process if audio arrives within `wait` s, else None."""
        exe = shutil.which("arecord")
        cmd = [exe, "-q", "-D", dev, "-t", "raw", "-f", "S16_LE", "-c", str(ch), "-r", str(SR)]
        print("mic:", " ".join(cmd), flush=True)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        except Exception as e:
            print("mic:", e, flush=True)
            return None
        got = threading.Event()
        frame = 2 * ch

        def reader():
            rest = b""
            while True:
                data = proc.stdout.read(4096)
                if not data:
                    if self.dev is proc:
                        self.error = f"mic stopped (arecord exit {proc.poll()})"
                        print(self.error, flush=True)
                        self.dev = None
                    got.set()
                    return
                if not got.is_set():
                    print(f"mic: audio arriving from {dev} ({ch} ch)", flush=True)
                    got.set()
                data = rest + data
                usable = len(data) - len(data) % frame
                rest = data[usable:]
                if usable:
                    x = np.frombuffer(data[:usable], "<i2").astype(np.float32) / 32768.0
                    if ch == 2:
                        x2 = x.reshape(-1, 2)
                        c = self.channel if self.usb else 2
                        x = x2[:, c] if c < 2 else x2.mean(axis=1)
                    self._process(x)

        self.dev = proc
        threading.Thread(target=reader, daemon=True).start()
        if got.wait(wait) and proc.poll() is None:
            return proc
        self.dev = None
        self._kill(proc)
        print(f"mic: {dev} ({ch} ch): " + ("no audio" if proc.returncode in (None, -9, -15) else f"exit {proc.returncode}"),
              flush=True)
        return None

    @staticmethod
    def _kill(proc):
        """Stop arecord and wait until it has really released the device (otherwise the next try is 'busy')."""
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    def _open_arecord(self, wait=1.0):
        if not shutil.which("arecord"):
            self.error = "arecord not found"
            return False
        for dev, ch in self._capture_candidates():
            proc = self._try_arecord(dev, ch, wait)
            if proc:
                self.dev, self.device_name, self.error = proc, f"{dev} {ch}ch", None
                return True
        return False

    def open(self):
        if self.dev is not None or self.half_duplex:
            return True
        if self.tap_mode:
            self.half_duplex = False
            self._start_tap()
            return True
        if use_alsa_tools():
            # The Brick can't record while it plays (the mic gives no audio with the speaker open), so the
            # mic is only opened while recording, with the output paused, the way GrainBrick does it.
            if not shutil.which("arecord"):
                self.error = "arecord not found"
                return False
            if self.usb:
                # A USB card is a separate device: it stays open, the speaker keeps playing.
                self.half_duplex = False
                if self.source not in [c for c, _ in capture_cards()]:
                    self.error = f"{self.source} not connected"
                    return False
                if self._open_arecord(wait=2.5):
                    return True
                self.error = self.error or f"no audio from {self.source} (see log.txt)"
                return False
            self.half_duplex = hasattr(self.out, "suspend")
            if not self.half_duplex:
                return self._open_arecord()
            self.error = None
            return True
        return self._open_sdl()

    @staticmethod
    def configure_builtin_mic():
        """Switch on the Brick's internal mic path in the codec mixer before capturing. With
        'ADCL Input MIC1 Boost' off the codec delivers no audio at all. These are the same baseline
        settings GrainBrick applies; they only touch capture controls, never playback."""
        exe = shutil.which("amixer")
        if not exe:
            print("mic: amixer not found, mixer left as is", flush=True)
            return
        for control, value in (("ADCL Input MIC1 Boost", "on"), ("MIC1 gain volume", "31"), ("ADC volume", "160,160")):
            try:
                r = subprocess.run([exe, "-c", "0", "sset", control, value], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, timeout=3)
                print(f"mic: amixer '{control}' {value}: " + ("ok" if r.returncode == 0 else r.stderr.strip()), flush=True)
            except Exception as e:
                print(f"mic: amixer '{control}' failed: {e}", flush=True)

    @staticmethod
    def free_capture():
        """Stop any arecord still holding the Brick's capture device (e.g. a dsnoop helper left over)."""
        if not sys.platform.startswith("linux"):
            return
        me = os.getpid()
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == me:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().split(b"\0")[0]
                if not cmd.endswith(b"arecord"):
                    continue
                holds = any(os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/snd/pcmC")
                            for fd in os.listdir(f"/proc/{pid}/fd"))
                if holds:
                    print(f"mic: stopping leftover arecord (pid {pid}) that holds the sound card", flush=True)
                    os.kill(int(pid), 9)
            except Exception:
                pass
        time.sleep(0.1)

    def _cb(self, _dev, mv):
        self._process(np.frombuffer(bytes(mv), dtype=np.float32))

    def _process(self, x):
        x = x * self.state.rec["gain"]
        pk = float(np.abs(x).max()) if len(x) else 0.0
        self.level = max(self.level * 0.85, pk)
        if self.recording:
            self.chunks.append(x)
            self.peaks.append(pk)
            self.frames += len(x)
            if self.frames >= MAX_REC_SECONDS * SR:
                self.recording = False

    def start(self):
        if not self.open():
            return False
        self.chunks, self.peaks, self.frames = [], [], 0
        if self.half_duplex:
            self.out.suspend()
            self.free_capture()
            self.configure_builtin_mic()
            if not self._open_arecord(wait=2.5):
                self.free_capture()
                self.out.resume()
                self.error = "no audio from the mic (see log.txt)"
                return False
        self.recording = True
        return True

    def start_async(self):
        """Start recording in the background so the screen doesn't freeze while the mic opens."""
        if self.starting or self.recording:
            return
        self.starting, self.start_failed = True, False

        def run():
            ok = False
            try:
                ok = self.start()
            except Exception as e:
                self.error = str(e)
            self.start_failed = not ok
            self.starting = False
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.recording = False
        if self.half_duplex:
            proc, self.dev = self.dev, None
            if proc:
                self._kill(proc)
            self.free_capture()
            self.out.resume()
        if not self.chunks:
            return None
        x = np.concatenate(self.chunks).astype(np.float32)
        self.chunks = []
        if self.state.rec["normalize"]:
            pk = np.abs(x).max()
            if pk > 1e-4:
                x = x / pk * 0.89
        return x


# ----------------------------------------------------------------------------- MIDI in

class MidiParser:
    def __init__(self, engine):
        self.engine = engine
        self.status = 0
        self.data = []
        self.sysex = False

    def feed(self, data):
        for b in data:
            if b >= 0xF8:
                continue
            if b >= 0x80:
                if b == 0xF0:
                    self.sysex = True
                elif b == 0xF7:
                    self.sysex = False
                elif b > 0xF0:
                    self.status = 0
                else:
                    self.sysex = False
                    self.status = b
                    self.data = []
                continue
            if self.sysex or not self.status:
                continue
            self.data.append(b)
            hi = self.status & 0xF0
            need = 1 if hi in (0xC0, 0xD0) else 2
            if len(self.data) == need:
                d1 = self.data[0]
                d2 = self.data[1] if need == 2 else 0
                self.data = []
                self.dispatch(hi, d1, d2)

    def dispatch(self, hi, d1, d2):
        e = self.engine
        e.midi_activity = time.time()
        if hi == 0xB0:              # BrickLooper only uses CCs (FX); notes and bend are ignored
            e.cc(d1, d2)


class MidiIn(threading.Thread):
    """Linux: reads raw ALSA MIDI devices (/dev/snd/midiC*D*) - no extra packages, works on KNULLI.
    Elsewhere: falls back to pygame.midi (PortMidi). Rescans for devices every 2 seconds."""

    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self.devices = []       # names, for the UI
        self.running = True

    def run(self):
        if sys.platform.startswith("linux") and glob.glob("/dev/snd"):
            self._run_alsa()
        else:
            self._run_portmidi()

    def _run_alsa(self):
        open_fds = {}
        parsers = {}
        last_scan = 0
        while self.running:
            now = time.time()
            if now - last_scan > 2:
                last_scan = now
                for path in glob.glob("/dev/snd/midiC*D*"):
                    if path not in open_fds:
                        try:
                            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                            open_fds[path] = fd
                            parsers[fd] = MidiParser(self.engine)
                        except OSError:
                            pass
                self.devices = [os.path.basename(p) for p in open_fds]
            if not open_fds:
                time.sleep(0.5)
                continue
            try:
                r, _, _ = select.select(list(open_fds.values()), [], [], 0.5)
            except (OSError, ValueError):
                r = []
            for fd in r:
                try:
                    data = os.read(fd, 512)
                    if data:
                        parsers[fd].feed(data)
                except BlockingIOError:
                    pass
                except OSError:
                    for p, f in list(open_fds.items()):
                        if f == fd:
                            del open_fds[p]
                    parsers.pop(fd, None)
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def _run_portmidi(self):
        try:
            import pygame.midi as pm
        except Exception as e:
            print("MIDI unavailable:", e)
            return
        inputs = {}
        last_scan = 0
        while self.running:
            now = time.time()
            if now - last_scan > 2 and not inputs:
                last_scan = now
                try:
                    if pm.get_init():
                        pm.quit()
                    pm.init()
                    for i in range(pm.get_count()):
                        interf, name, is_in, _, opened = pm.get_device_info(i)
                        if is_in and not opened:
                            try:
                                inputs[i] = (pm.Input(i), MidiParser(self.engine))
                                self.devices.append(name.decode(errors="ignore"))
                            except Exception:
                                pass
                except Exception as e:
                    print("MIDI scan failed:", e)
            got = False
            for i, (inp, parser) in list(inputs.items()):
                try:
                    while inp.poll():
                        for (msg, _ts) in inp.read(64):
                            if msg[0] < 0xF0:
                                parser.feed(bytes(msg[:3]))
                            got = True
                except Exception:
                    del inputs[i]
            if not got:
                time.sleep(0.001)


# ----------------------------------------------------------------------------- parameters

def f_pct(v): return f"{v * 100:.0f}%"
def f_time(v): return f"{v * 1000:.1f} ms" if v < 0.01 else f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"
def f_hz(v): return f"{v:.0f} Hz" if v < 1000 else f"{v / 1000:.2f} kHz"
def f_st(v): return f"{int(v):+d} st"
def f_ct(v): return f"{int(v):+d} ct"
def f_pan(v): return "C" if abs(v) < 0.01 else (f"L{abs(v) * 100:.0f}" if v < 0 else f"R{v * 100:.0f}")
def f_bits(v): return "Off" if v >= 16 else f"{int(v)} bit"
def f_ds(v): return "Off" if v <= 1 else f"1/{int(v)}"
def f_gain(v): return "-inf dB" if v <= 0.0001 else f"{20 * math.log10(v):+.1f} dB"


class Param:
    def __init__(self, key, label, kind, lo=0.0, hi=1.0, step=0.01, fmt=f_pct, opts=None, scope="slot"):
        self.key, self.label, self.kind, self.lo, self.hi = key, label, kind, lo, hi
        self.step, self.fmt, self.opts, self.scope = step, fmt, opts, scope

    def frac(self, v):
        if self.kind == "exp":
            return math.log(v / self.lo) / math.log(self.hi / self.lo)
        if self.kind in ("lin", "note"):
            return (v - self.lo) / (self.hi - self.lo)
        return None


OFFON = ["Off", "On"]
P_FILE = Param("file", "File", "file")
P_PAD = Param("pad", "Pad", "pad")          # kit mode: which pad the page edits
P_ROOT = Param("root", "Root note", "note", 0, 127, 1, note_name)
SAMPLE_PARAMS_TAIL = [
    Param("tune", "Tune", "lin", -24, 24, 1, f_st),
    Param("fine", "Fine", "lin", -100, 100, 5, f_ct),
    Param("gain", "Gain", "lin", 0, 1.5, 0.02, f_gain),
    Param("pan", "Pan", "lin", -1, 1, 0.05, f_pan),
    Param("start", "Start", "lin", 0, 1, 0.005, lambda v: f"{v * 100:.1f}%"),
    Param("end", "End", "lin", 0, 1, 0.005, lambda v: f"{v * 100:.1f}%"),
    Param("loop", "Loop", "enum", opts=OFFON),
    Param("reverse", "Reverse", "enum", opts=OFFON),
    Param("trigger", "Trigger", "enum", opts=["Gate", "One-shot"]),
]
AMP_PARAMS = [
    Param("attack", "Attack", "exp", 0.0005, 5.0, 0.12, f_time),
    Param("decay", "Decay", "exp", 0.005, 10.0, 0.12, f_time),
    Param("sustain", "Sustain", "lin", 0, 1, 0.02),
    Param("release", "Release", "exp", 0.002, 10.0, 0.12, f_time),
    Param("velsens", "Velocity", "lin", 0, 1, 0.05),
]
FX_PARAMS = [
    Param("drive", "Drive", "lin", 0, 1, 0.02, scope="fx"),
    Param("bits", "Bitcrush", "lin", 2, 16, 1, f_bits, scope="fx"),
    Param("downsample", "Downsample", "lin", 1, 32, 1, f_ds, scope="fx"),
    Param("ftype", "Type", "enum", opts=["Off", "Low-pass", "High-pass", "Band-pass"], scope="fx"),
    Param("cutoff", "Cutoff", "exp", 30, 18000, 0.05, f_hz, scope="fx"),
    Param("reso", "Resonance", "lin", 0, 1, 0.02, scope="fx"),
    Param("master", "Master", "lin", 0, 1.5, 0.02, f_gain, scope="fx"),
    Param("dtime", "Time", "exp", 0.025, 2.0, 0.04, f_time, scope="fx"),
    Param("dfb", "Feedback", "lin", 0, 0.95, 0.02, scope="fx"),
    Param("dtone", "Tone", "lin", 0, 1, 0.05, scope="fx"),
    Param("dping", "Ping-pong", "enum", opts=OFFON, scope="fx"),
    Param("dmix", "Mix", "lin", 0, 1, 0.02, scope="fx"),
    Param("vsize", "Size", "lin", 0, 1, 0.02, scope="fx"),
    Param("vdamp", "Damping", "lin", 0, 1, 0.05, scope="fx"),
    Param("vmix", "Mix", "lin", 0, 1, 0.02, scope="fx"),
]
def f_pos(v): return f"{v * 100:.1f}%"
def f_rate(v): return f"{v:.2f} Hz" if v < 10 else f"{v:.1f} Hz"
def f_delay(v): return "Off" if v < 0.001 else f"{v:.2f} s"
def f_semi(v): return "Off" if v < 0.001 else f"{v:.2f} st"


LOOP_PARAMS = [
    Param("loop", "Loop", "enum", opts=OFFON),
    Param("lstart", "Loop start", "lin", 0, 1, 0.002, f_pos),
    Param("lend", "Loop end", "lin", 0, 1, 0.002, f_pos),
    Param("xfade", "Crossfade", "lin", 0, 0.5, 0.01),
]
TOOL_PARAMS = [Param("chop", "Chop", "action"), Param("stretch", "Stretch", "action"),
               Param("snap", "Zero snap", "action")]
LFO_PARAMS = [
    Param("wave", "Wave", "enum", opts=["Sine", "Triangle", "Saw", "Square", "S&H"], scope="lfo"),
    Param("rate", "Rate", "exp", 0.05, 20.0, 0.06, f_rate, scope="lfo"),
    Param("delay", "Delay", "lin", 0, 4, 0.05, f_delay, scope="lfo"),
    Param("modwheel", "Mod wheel", "enum", opts=["Off", "Depth"], scope="lfo"),
    Param("pitch", "Pitch", "lin", 0, 12, 0.05, f_semi, scope="lfo"),
    Param("volume", "Volume", "lin", 0, 1, 0.02, scope="lfo"),
    Param("cutoff", "Cutoff", "lin", 0, 1, 0.02, scope="lfo"),
]
LFO_WAVES = ["SINE", "TRIANGLE", "SAW", "SQUARE", "S&H"]
P_CCMAP = Param("ccmap", "CC map", "action")
P_CCRESET = Param("ccreset", "Reset map", "action")
P_RECORD = Param("record", "Record", "action", scope="rec")
REC_RATES = [32000, 44100, 48000]
REC_BITS = [8, 16, 24]
PROJECT_PARAMS = [Param("pname", "Name", "action"), Param("pload", "Load", "action"),
                  Param("psaveas", "Save as", "action"), Param("pnew", "New", "action")]
SETTINGS_PARAMS = [Param("rate", "Sample rate", "enum", opts=["32 kHz", "44.1 kHz", "48 kHz"], scope="rec"),
                   Param("bits", "Bit depth", "enum", opts=["8 bit", "16 bit", "24 bit"], scope="rec")]
# What a CC can move: FX values, FX section switches, LFO values ("lfo:" prefix, cutoff exists in both)
CC_TARGETS = {p.key: p for p in FX_PARAMS}
CC_TARGETS.update({f"lfo:{p.key}": p for p in LFO_PARAMS})
CC_TARGETS.update({flag: Param(flag, title.capitalize() + " on/off", "switch", scope="fx")
                   for title, flag in FX_SECTION_SWITCH.items()})


def cc_key(p, page):
    return f"lfo:{p.key}" if p.scope == "lfo" else p.key


def cc_label(key):
    p = CC_TARGETS.get(key)
    if p is None:
        return key.upper()
    return ("LFO " if p.scope == "lfo" else "") + p.label.upper()


def f_ms(v): return f"{int(v):+d} ms" if int(v) else "0 ms"


P_MONITOR = Param("mon", "Monitor", "enum", opts=OFFON, scope="loop")
LOOPER_PARAMS = [Param("recdub", "Rec end", "enum", opts=["Play", "Dub"], scope="loop"),
                 Param("dubmode", "Dub mode", "enum", opts=["Add", "Replace"], scope="loop")]
METRO_PARAMS = [Param("met", "Click", "enum", opts=OFFON, scope="loop"),
                Param("bpm", "BPM", "lin", 40, 240, 1, lambda v: f"{int(v)}", scope="loop"),
                Param("sig", "Time", "enum", opts=TIME_SIGS, scope="loop"),
                Param("countin", "Count-in", "enum", opts=["Off", "1 bar"], scope="loop")]
P_CLICKVOL = Param("click", "Click vol", "lin", 0, 100, 5, lambda v: f"{int(v)}%", scope="loop")
P_OFFSET = Param("loop_offset", "Offset", "lin", -100, 200, 1, f_ms, scope="ui")
TRACK_PARAMS = [Param(f"track{i}", f"Track {i + 1}", "track") for i in range(LOOP_TRACKS)]
P_SOURCE = Param("rec_source", "Source", "source", scope="ui")
P_USBIN = Param("usb_in", "USB input", "usbin", scope="ui")      # which recording device of the USB card
P_CHANNEL = Param("rec_channel", "Channel", "enum", opts=REC_CHANNELS, scope="ui")
REC_PARAMS = [P_RECORD, P_SOURCE, P_CHANNEL,
              Param("gain", "Input gain", "lin", 0, 4, 0.05, f_gain, scope="rec"),
              Param("normalize", "Normalize", "enum", opts=OFFON, scope="rec")]


# ----------------------------------------------------------------------------- UI

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
· ..... ..... ..... ..#.. ..... ..... .....
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


PAGES = ["LOOP", "FX", "SETTINGS"]
TAB_LABEL = {"SETTINGS": "SET"}
P_DISPLAY = Param("display", "Colour", "enum", opts=PALETTE_NAMES, scope="ui")
P_LATENCY = Param("latency", "Latency", "enum", opts=LATENCY_NAMES, scope="ui")
PIANO_SCANCODES = ["A", "W", "S", "E", "D", "F", "T", "G", "Y", "H", "U", "J", "K", "O", "L", "P", "SEMICOLON", "APOSTROPHE"]


class App:
    def __init__(self, args):
        self.args = args
        if sys.platform == "win32":
            # Without this, Windows display scaling (125%, 150%) blows the window up past the screen and blurs it.
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
        pygame.display.init()
        pygame.font.init()
        pygame.joystick.init()
        self.controllers = []
        self.joysticks = []
        self.ctrl_mod = None
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
        flags = pygame.FULLSCREEN if fullscreen else 0
        size = (SCREEN_W, SCREEN_H) if fullscreen else (int(SCREEN_W * self.scale), int(SCREEN_H * self.scale))
        self.window = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("BrickLooper")
        if fullscreen:
            pygame.mouse.set_visible(False)
        self.surf = pygame.Surface((SCREEN_W, SCREEN_H)) if self.scale != 1.0 else self.window
        print(f"video: driver {pygame.display.get_driver()}, window {self.window.get_size()}", flush=True)

        self._lcds = {}
        self.load_settings()
        self.lcd = self.make_lcd()              # halo around lit dots only on the PC; it costs a few ms
        self.dialog = None
        self.wave_key = None
        self.fcurve_key = None
        self.lfo_key = None

        self.splash("1/5  Starting")
        self.splash("2/5  Loading project")
        self.state = State()
        self.load_project()

        self.splash("3/5  Starting audio")
        if ON_DEVICE:
            Recorder.free_capture()     # a crashed earlier run may have left arecord holding the mic
        self.engine = make_engine(self.state)
        self.engine.ccmap = self.ui["ccmap"]            # learning edits the saved map directly
        self.job = None                                 # background work (time-stretch)
        self.audio = make_audio_out(self.engine, self.ui["latency"])
        self.splash("4/5  Starting MIDI")
        self.recorder = Recorder(self.state)
        self.recorder.out = self.audio
        self.recorder.engine = self.engine
        self.recorder.set_source(self.ui["rec_source"], self.ui["rec_channel"])
        self.loop_track = 0
        self.loop_wave = {}
        self.loop_saved = {}
        self.loop_writer = None
        self.update_capture()
        self.load_loops()
        print("rec: extra input cards:", capture_cards() or "none", flush=True)
        self.midi = MidiIn(self.engine)
        self.midi.start()
        self.splash("5/5  Ready")
        time.sleep(max(0.0, 1.2 - (time.time() - self._splash_t0)))   # keep the splash up long enough to read

        self.page = 0
        self.sel = [0] * len(PAGES)
        self.held = {}
        self.browser = None
        self.toast = ("", 0.0)
        self.piano_held = {}
        self.audition_note = None
        self.select_combo = False
        self.shown_overloads = 0
        self.logged_inputs = 0
        self.dirty = True
        self.fcurve_key = None
        self.running = True
        self.last_input = "pad" if ON_DEVICE else "keyboard"
        self.samples_cache = list_samples()
        self.wave_surf_key = None
        self.wave_surf = None
        self.last_rec_peaks = []
        self.scan = {name: getattr(pygame, "KSCAN_" + name) for name in PIANO_SCANCODES}
        self.piano_map = {}                     # no sampler: the A-K row doesn't play notes
        self.key_map = {
            pygame.KSCAN_UP: "up", pygame.KSCAN_DOWN: "down", pygame.KSCAN_LEFT: "left", pygame.KSCAN_RIGHT: "right",
            pygame.KSCAN_Z: "a", pygame.KSCAN_RETURN: "a", pygame.KSCAN_X: "b", pygame.KSCAN_ESCAPE: "b",
            pygame.KSCAN_BACKSPACE: "b", pygame.KSCAN_SPACE: "x", pygame.KSCAN_LSHIFT: "y", pygame.KSCAN_RSHIFT: "y",
            pygame.KSCAN_LEFTBRACKET: "l1", pygame.KSCAN_RIGHTBRACKET: "r1", pygame.KSCAN_COMMA: "l2",
            pygame.KSCAN_PERIOD: "r2", pygame.KSCAN_TAB: "select",
        }
        self.show_toast("Hold SELECT to see the controls" if ON_DEVICE else "Hold Tab to see the keyboard controls", 4)
        if self.audio.error:
            self.show_toast("Audio: " + self.audio.error, 6)

    # ---------- setup helpers
    def splash(self, step):
        """Startup screen. The last step shown also says where it stopped if startup hangs."""
        print("startup:", step, flush=True)
        if not hasattr(self, "_splash_t0"):
            self._splash_t0 = time.time()
        try:
            n, total = (int(v) for v in step.split()[0].split("/"))
            label = step.split(None, 1)[1]
        except Exception:
            n, total, label = 0, 5, step
        L = self.lcd
        L.clear()
        L.frame(14, 22, 228, 104)
        L.frame(16, 24, 224, 100)
        L.ctext(128, 34, "BRICKLOOPER", scale=3)
        L.ctext(128, 62, "LOOPER - SAMPLER - FX - MIDI")
        pts = []
        for i in range(201):
            t = i / 200
            v = math.sin(t * 2 * math.pi * 11) * math.exp(-t * 3.2) * (1 - math.exp(-t * 60))
            pts.append((28 + i, 98 - v * 18))
        L.hline(28, 98, 201, dotted=True)
        L.polyline(pts)
        seg_w = (228 - 4 * (total - 1)) / total
        for i in range(total):
            x = 14 + int(i * (seg_w + 4))
            if i < n:
                L.fill(x, 138, int(seg_w), 9)
            else:
                L.frame(x, 138, int(seg_w), 9)
        L.ctext(128, 156, label)
        L.text(2, 183, "TRIMUI BRICK / KNULLI" if ON_DEVICE else "DESKTOP TEST BUILD")
        L.rtext(254, 183, f"V{VERSION}")
        L.compose(self.surf, full=True)
        if self.surf is not self.window:
            pygame.transform.smoothscale(self.surf, self.window.get_size(), self.window)
        pygame.display.flip()
        pygame.event.pump()

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
        for c in self.controllers:
            print("input: game controller", repr(c.name))
        for j in self.joysticks:
            print(f"input: joystick {j.get_name()!r} buttons={j.get_numbuttons()} hats={j.get_numhats()} axes={j.get_numaxes()}"
                  f" (not recognised as a game controller: using controls.json joy_buttons)")

    # ---------- settings (app-wide, not per project)
    def load_settings(self):
        self.ui = dict(display=0, project="DEFAULT", latency=0, ccmap=dict(DEFAULT_CCMAP),
                       rec_source="internal", rec_channel=2, volume=1.0, loop_offset=0, usb_inputs={}, usb_dev=0)
        try:
            with open(SETTINGS_FILE) as f:
                d = json.load(f)
            self.ui.update({k: v for k, v in d.items() if k in self.ui})
        except Exception:
            pass
        self.ui["display"] = int(clamp(int(self.ui["display"]), 0, len(PALETTE_NAMES) - 1))
        self.ui["latency"] = int(clamp(int(self.ui["latency"]), 0, len(LATENCY_NAMES) - 1))
        self.ui["project"] = clean_name(str(self.ui["project"])) or "DEFAULT"
        global OUTPUT_VOLUME
        self.ui["volume"] = float(clamp(float(self.ui["volume"]), 0.0, 1.0))
        OUTPUT_VOLUME = self.ui["volume"] ** 2
        self.ui["rec_source"] = str(self.ui["rec_source"]) or "internal"
        self.ui["rec_channel"] = int(clamp(int(self.ui["rec_channel"]), 0, 2))
        self.ui["loop_offset"] = int(clamp(int(self.ui["loop_offset"]), -100, 200))
        ui_in = self.ui["usb_inputs"] if isinstance(self.ui["usb_inputs"], dict) else {}
        self.ui["usb_inputs"] = {str(c): int(d) for c, d in ui_in.items() if str(d).isdigit()}   # card -> device
        self.ui["usb_dev"] = int(self.ui["usb_dev"]) if str(self.ui["usb_dev"]).isdigit() else 0
        cm = self.ui["ccmap"] if isinstance(self.ui["ccmap"], dict) else {}
        self.ui["ccmap"] = {str(int(c)): k for c, k in cm.items()
                            if str(c).isdigit() and 0 <= int(c) < 128 and k in CC_TARGETS}

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self.ui, f, indent=1)
        except Exception as e:
            print("settings not saved:", e, flush=True)

    def make_lcd(self):
        name = PALETTE_NAMES[self.ui["display"]]
        if name not in self._lcds:
            self._lcds[name] = LCD(glow=not ON_DEVICE, palette=name)
        lcd = self._lcds[name]
        lcd.prev = None                     # new colours: redraw the whole display
        return lcd

    # ---------- projects
    @staticmethod
    def project_path(name):
        return os.path.join(PROJECTS_DIR, name + ".json")

    @staticmethod
    def list_projects():
        try:
            return sorted(os.path.splitext(f)[0] for f in os.listdir(PROJECTS_DIR) if f.lower().endswith(".json"))
        except OSError:
            return []

    def migrate_projects(self):
        """1.1 kept a single projects/project.json; it becomes the project called DEFAULT."""
        old = os.path.join(PROJECTS_DIR, "project.json")
        if os.path.exists(old) and not os.path.exists(self.project_path("DEFAULT")):
            try:
                os.replace(old, self.project_path("DEFAULT"))
                print("projects: project.json -> DEFAULT.json", flush=True)
            except OSError as e:
                print("projects: migration failed:", e, flush=True)

    def load_defaults(self):
        pass                                    # a new project starts with empty tracks

    def reset_state(self):
        self.state.__dict__.update(State().__dict__)
        self.sel = [0] * len(PAGES)
        self.wave_key = None
        self.fcurve_key = None
        self.lfo_key = None

    def load_project(self):
        self.migrate_projects()
        path = self.project_path(self.ui["project"])
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.state.from_json(json.load(f))
                return
            except Exception as e:
                print("project load failed:", e, flush=True)
        self.load_defaults()

    @staticmethod
    def loops_dir(name):
        return os.path.join(PROJECTS_DIR, name + "_LOOPS")

    def save_loops(self):
        """Each track that changed since it was last saved goes to projects/<NAME>_LOOPS/T<n>.wav. The audio
        is copied here (fast) and written to the SD card in the background, so the screen doesn't stall."""
        e = self.engine
        if not e.looper:
            return
        e.refresh_loop()
        name, st = self.ui["project"], self.state
        folder = self.loops_dir(name)
        jobs = []
        st.loop_len = e.loop["length"]
        for t, tr in enumerate(e.loop["tracks"]):
            path = os.path.join(folder, f"T{t + 1}.wav")
            has = tr["state"] != 0 and tr["len"] > 0 and not tr["fresh"]
            key = (name, t)
            if has and (self.loop_saved.get(key) != tr["ver"] or not os.path.exists(path)):
                jobs.append((path, e.loop_audio(t)))
                self.loop_saved[key] = tr["ver"]
            elif not has:
                if st.loop_tracks[t] or os.path.exists(path):
                    jobs.append((path, None))
                self.loop_saved.pop(key, None)
            st.loop_tracks[t] = has
        if not jobs:
            return
        prev = self.loop_writer

        def write():
            if prev is not None:
                prev.join()
            for path, x in jobs:
                try:
                    if x is None:
                        if os.path.exists(path):
                            os.remove(path)
                    else:
                        save_wav_stereo_float(path, x)
                except Exception as ex:
                    print("loop save failed:", path, ex, flush=True)
            print(f"loops: saved {sum(1 for _, x in jobs if x is not None)} track(s) to {folder}", flush=True)
        self.loop_writer = threading.Thread(target=write, daemon=True)
        self.loop_writer.start()

    def load_loops(self):
        e = self.engine
        if not e.looper:
            return
        if self.loop_writer is not None:
            self.loop_writer.join()                 # the files of this project may still be being written
        e.loop_reset()
        st, name = self.state, self.ui["project"]
        folder = self.loops_dir(name)
        for t in range(LOOP_TRACKS):
            path = os.path.join(folder, f"T{t + 1}.wav")
            if st.loop_tracks[t] and os.path.exists(path):
                try:
                    x = load_wav_stereo(path)
                    e.loop_load(t, x, st.loop_len or len(x))
                except Exception as ex:
                    print("loop load failed:", path, ex, flush=True)
        e.refresh_loop()
        self.loop_wave = {}
        for t, tr in enumerate(e.loop["tracks"]):
            if tr["len"] > 0 and tr["state"] != 0:
                self.loop_saved[(name, t)] = tr["ver"]

    def save_project(self, toast=True):
        try:
            self.save_loops()
            os.makedirs(PROJECTS_DIR, exist_ok=True)
            with open(self.project_path(self.ui["project"]), "w") as f:
                json.dump(self.state.to_json(), f, indent=1)
            self.save_settings()
            if toast:
                self.show_toast(f"Saved {self.ui['project']}")
        except Exception as e:
            self.show_toast(f"Save failed: {e}", 4)

    def switch_project(self, name, fresh=False):
        self.save_project(toast=False)
        self.engine.all_off()
        self.reset_state()
        self.ui["project"] = name
        if fresh:
            self.load_defaults()
        else:
            self.load_project()
        self.load_loops()
        self.save_project(toast=False)
        self.show_toast(("New project " if fresh else "Loaded ") + name)

    def next_name(self, prefix, existing):
        nums = [int(m.group(1)) for n in existing for m in [re.match(rf"{prefix}_(\d+)$", n)] if m]
        return f"{prefix}_{(max(nums) + 1) if nums else 1:03d}"

    def do_action(self, key):
        if key == "record":
            self.toggle_record()
        elif key == "chop":
            self.chop_menu()
        elif key == "stretch":
            self.stretch_menu()
        elif key == "snap":
            self.zero_snap()
        elif key == "ccmap":
            self.ccmap_menu()
        elif key == "ccreset":
            def reset():
                self.ui["ccmap"].clear()
                self.ui["ccmap"].update(DEFAULT_CCMAP)
                self.save_settings()
                self.show_toast("CC map reset: 74 CUTOFF, 71 RESO, 7 MASTER", 3)
            self.confirm("RESET MIDI CC MAP?", ["BACK TO CC74 CUTOFF,", "CC71 RESONANCE, CC7 MASTER."], reset)
        elif key == "pname":
            cur = self.ui["project"]
            self.ask_name("RENAME PROJECT", cur, lambda n: self.rename_project(cur, n))
        elif key == "pload":
            self.open_browser("projects")
        elif key == "psaveas":
            def save_as(n):
                if n in self.list_projects():
                    self.show_toast(f"{n} already exists", 3)
                    return
                self.ui["project"] = n
                self.save_project(toast=False)
                self.show_toast(f"Saved as {n}")
            self.ask_name("SAVE PROJECT AS", self.next_name("PROJECT", self.list_projects()), save_as)
        elif key == "pnew":
            def new(n):
                if n in self.list_projects():
                    self.show_toast(f"{n} already exists", 3)
                    return
                self.switch_project(n, fresh=True)
            self.ask_name("NEW PROJECT", self.next_name("PROJECT", self.list_projects()), new)

    # ---------- MIDI learn
    def learn_menu(self, p):
        key = cc_key(p, PAGES[self.page])
        items = [(f"LEARN {p.label.upper()}", lambda: self.start_learn(key))]
        flag = FX_SWITCH.get(p.key) if PAGES[self.page] == "FX" else None
        if flag:
            items.append((f"LEARN {cc_label(flag)}", lambda: self.start_learn(flag)))
        for c, k in sorted(self.ui["ccmap"].items(), key=lambda t: int(t[0])):
            if k == key or (flag and k == flag):
                items.append((f"FORGET CC{c} {cc_label(k)}", lambda c=c: self.forget_cc(c)))
        self.menu("MIDI CC", items)

    def start_learn(self, key):
        self.engine.learn = key
        self.show_toast(f"MOVE A KNOB FOR {cc_label(key)}  (B CANCEL)", 30)

    def forget_cc(self, c):
        k = self.ui["ccmap"].pop(c, None)
        self.save_settings()
        self.show_toast(f"CC{c} {cc_label(k)} removed")

    def ccmap_menu(self):
        m = sorted(self.ui["ccmap"].items(), key=lambda t: int(t[0]))
        if not m:
            self.show_toast("No CC mapped: press B on an FX or LFO value", 3)
            return

        def ask(c, k):
            self.confirm(f"REMOVE CC{c}?", [cc_label(k)], lambda: self.forget_cc(c), yes="REMOVE")
        self.menu("MIDI CC MAP  (A REMOVE)", [(f"CC{c:<4}{cc_label(k)}", lambda c=c, k=k: ask(c, k)) for c, k in m[:9]])

    # ---------- sample tools: chop, time-stretch, zero snap
    def chop_menu(self):
        if self.state.cur_slot().data is None:
            self.show_toast("Load a sample first")
            return
        self.menu("CHOP REGION ONTO PADS", [("4 SLICES", lambda: self.chop(4)), ("8 SLICES", lambda: self.chop(8)),
                                            ("16 SLICES", lambda: self.chop(16)),
                                            ("AUTO (AT THE ATTACKS)", lambda: self.chop(0))])

    def chop(self, n):
        src = self.state.cur_slot()
        d, path, N = src.data, src.path, len(src.data)
        s, e = int(src.p["start"] * N), int(src.p["end"] * N)
        if n:
            bounds = [int(round(s + (e - s) * i / n)) for i in range(n + 1)]
        else:
            bounds = [s + b for b in detect_onsets(d[s:e], SR)]
        k = len(bounds) - 1
        if k < 2:
            self.show_toast("No attacks found: try equal slices", 3)
            return

        def do():
            self.engine.all_off()
            for i in range(k):
                pad = self.state.slots[1 + i]
                pad.set_data(d, path)
                pad.p = dict(PAD_DEFAULTS, start=bounds[i] / N, end=bounds[i + 1] / N,
                             lstart=bounds[i] / N, lend=bounds[i + 1] / N)
            self.state.mode = "kit"
            self.state.sel_pad = 0
            self.wave_key = None
            self.save_project(toast=False)
            self.show_toast(f"Chopped into {k} pads (KIT mode)", 3)
        busy = [i for i in range(k) if self.state.slots[1 + i].data is not None and self.state.slots[1 + i] is not src]
        if busy:
            self.confirm(f"CHOP INTO {k} PADS?", [f"PADS 1-{k} ARE REPLACED."], do, yes="CHOP")
        else:
            do()

    def stretch_menu(self):
        slot = self.state.cur_slot()
        if slot.data is None:
            self.show_toast("Load a sample first")
            return
        if self.job is not None:
            self.show_toast("Still working on the last one")
            return
        items = [(f"LENGTH {pct}%", lambda r=pct / 100: self.ask_stretch(r)) for pct in (50, 67, 75, 90, 110, 125, 150, 200)]
        self.menu("TIME-STRETCH (SAME PITCH)", items)

    def ask_stretch(self, ratio):
        slot = self.state.cur_slot()
        base = os.path.splitext(slot.name)[0] if slot.path else "TAKE"
        default = clean_name(f"{base}_T{round(ratio * 100)}")[:NAME_MAX]
        self.ask_name("NAME THE NEW SAMPLE", default, lambda n: self.start_stretch(ratio, n))

    def start_stretch(self, ratio, name):
        slot = self.state.cur_slot()
        job = dict(kind="stretch", slot=slot, name=name, ratio=ratio, progress=0.0, done=False, result=None, error=None)

        def run():
            try:
                y = time_stretch(slot.data, ratio, SR, progress=lambda f: job.__setitem__("progress", f))
                folder = os.path.dirname(slot.path) if slot.path else SAMPLES_DIR
                path, i = os.path.join(folder, name + ".wav"), 2
                while os.path.exists(path):
                    path = os.path.join(folder, f"{name}_{i}.wav")
                    i += 1
                pk = float(np.abs(y).max()) if len(y) else 0.0
                if pk > 0.999:
                    y = y / pk * 0.999
                save_wav(path, y, SR, 24)
                job["result"] = path
            except Exception as e:
                traceback.print_exc()
                job["error"] = str(e)
            job["done"] = True
        self.job = job
        self.show_toast("Stretching...", 2)
        threading.Thread(target=run, daemon=True).start()

    def finish_job(self):
        job, self.job = self.job, None
        if job["error"] or not job["result"]:
            self.show_toast("Stretch failed: " + (job["error"] or "?"), 4)
            return
        try:
            job["slot"].load(job["result"])          # keeps start/end/loop as fractions of the length
        except Exception as e:
            self.show_toast(f"Can't load: {e}", 4)
            return
        self.wave_key = None
        self.samples_cache = list_samples()
        self.save_project(toast=False)
        self.show_toast(f"Saved {os.path.basename(job['result'])}", 3)

    def zero_snap(self):
        slot = self.state.cur_slot()
        if slot.data is None:
            self.show_toast("Load a sample first")
            return
        d, p, N = slot.data, slot.p, len(slot.data)
        ls, le = loop_bounds(p, N)
        r = int(SR * 0.01)
        ls2, le2 = zero_cross(d, ls, r), zero_cross(d, le, r)
        if le2 - ls2 < 2:
            ls2, le2 = ls, le
        p["lstart"], p["lend"] = (ls2 + 0.5) / N, (le2 + 0.5) / N
        self.show_toast("Loop moved to zero crossings" if (ls2, le2) != (ls, le) else "Already on zero crossings")

    def rename_project(self, old, new):
        if new == old:
            return
        if new in self.list_projects():
            self.show_toast(f"{new} already exists", 3)
            return
        try:
            if os.path.exists(self.project_path(old)):
                os.replace(self.project_path(old), self.project_path(new))
            if self.loop_writer is not None:
                self.loop_writer.join()
            if os.path.isdir(self.loops_dir(old)):
                os.replace(self.loops_dir(old), self.loops_dir(new))
            for t in range(LOOP_TRACKS):
                if (old, t) in self.loop_saved:
                    self.loop_saved[(new, t)] = self.loop_saved.pop((old, t))
        except OSError as e:
            self.show_toast(f"Rename failed: {e}", 4)
            return
        if self.ui["project"] == old:
            self.ui["project"] = new
            self.save_settings()
        self.show_toast(f"Renamed to {new}")

    def delete_project(self, name):
        try:
            os.remove(self.project_path(name))
            shutil.rmtree(self.loops_dir(name), ignore_errors=True)
            self.show_toast(f"Deleted {name}")
        except OSError as e:
            self.show_toast(f"Delete failed: {e}", 4)

    # ---------- samples on disk
    def slots_using(self, path):
        out = []
        for i, s in enumerate(self.state.slots):
            if s.path and os.path.abspath(s.path) == os.path.abspath(path):
                out.append("CHROMATIC" if i == 0 else f"PAD {i}")
        return out

    def rename_sample(self, path, new):
        folder, ext = os.path.dirname(path), os.path.splitext(path)[1] or ".wav"
        target = os.path.join(folder, new + ext)
        if os.path.abspath(target) == os.path.abspath(path):
            return path
        if os.path.exists(target):
            self.show_toast(f"{new}{ext.upper()} already exists", 3)
            return None
        try:
            os.replace(path, target)
        except OSError as e:
            self.show_toast(f"Rename failed: {e}", 4)
            return None
        for s in self.state.slots:
            if s.path and os.path.abspath(s.path) == os.path.abspath(path):
                s.path, s.name = target, os.path.basename(target)
        self.samples_cache = list_samples()
        self.save_project(toast=False)
        self.show_toast(f"Renamed to {os.path.basename(target)}")
        return target

    def delete_sample(self, path):
        try:
            os.remove(path)
        except OSError as e:
            self.show_toast(f"Delete failed: {e}", 4)
            return
        self.engine.all_off()
        for s in self.state.slots:
            if s.path and os.path.abspath(s.path) == os.path.abspath(path):
                s.data, s.path, s.name, s.peaks = None, None, "(empty)", None
        self.wave_key = None
        self.samples_cache = list_samples()
        self.save_project(toast=False)
        self.show_toast(f"Deleted {os.path.basename(path)}")

    # ---------- recording: every take gets a name before it is saved
    def finish_take(self, x, peaks):
        existing = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(REC_DIR, "*.wav"))]
        default = self.next_name("REC", existing)

        def discard():
            def back():
                self.ask_name("NAME THE TAKE", default, lambda n: self.save_take(x, peaks, n),
                              cancel=("DISCARD", discard))
            self.confirm("DISCARD THIS TAKE?", ["THE RECORDING IS LOST."], lambda: self.show_toast("Take discarded"),
                         on_no=back, yes="DISCARD")
        self.ask_name("NAME THE TAKE", default, lambda n: self.save_take(x, peaks, n), cancel=("DISCARD", discard))

    def save_take(self, x, peaks, name):
        rate = REC_RATES[int(self.state.rec["rate"])]
        bits = REC_BITS[int(self.state.rec["bits"])]
        os.makedirs(REC_DIR, exist_ok=True)
        path, n = os.path.join(REC_DIR, name + ".wav"), 2
        while os.path.exists(path):
            path = os.path.join(REC_DIR, f"{name}_{n}.wav")
            n += 1
        # Captured at the engine rate; converted to the chosen rate / bit depth when saved, and the
        # saved file is what gets played, so the slot really sounds like 32k or 8 bit.
        save_wav(path, resample_fft(x, SR, rate), rate, bits)
        slot = self.state.cur_slot()
        try:
            slot.load(path)
        except Exception as e:
            print("take reload failed:", e, flush=True)
            slot.set_data(x, path)
        slot.p["start"], slot.p["end"] = 0.0, 1.0
        self.last_rec_peaks = list(peaks)
        self.samples_cache = list_samples()
        self.save_project(toast=False)
        target = "chromatic slot" if self.state.mode == "chromatic" else f"pad {self.state.sel_pad + 1}"
        self.show_toast(f"Saved {os.path.basename(path)} to {target}", 3)

    # ---------- dialogs: on-screen keyboard, yes/no, small menu
    def ask_name(self, title, text, on_ok, cancel=("CANCEL", None)):
        self.dialog = dict(kind="keyboard", title=title, text=clean_name(text), on_ok=on_ok,
                           cancel_label=cancel[0], on_cancel=cancel[1], row=0, col=0)

    def confirm(self, title, lines, on_yes, on_no=None, yes="YES"):
        self.dialog = dict(kind="confirm", title=title, lines=lines, on_yes=on_yes, on_no=on_no, yes=yes)

    def menu(self, title, items):
        self.dialog = dict(kind="menu", title=title, items=items, idx=0)

    def _close_dialog_and(self, fn, *args):
        self.dialog = None
        if fn:
            fn(*args)

    def kb_rows(self, d):
        return KB_ROWS + [["DEL", "CLEAR", d["cancel_label"], "OK"]]

    def dialog_button(self, btn):
        d = self.dialog
        self.dirty = True
        if d["kind"] == "keyboard":
            rows = self.kb_rows(d)
            if btn in ("up", "down"):
                old = rows[d["row"]]
                d["row"] = (d["row"] + (1 if btn == "down" else -1)) % len(rows)
                new = rows[d["row"]]
                # keep roughly the same horizontal position between rows of different length
                d["col"] = int(round(d["col"] / max(1, len(old) - 1) * (len(new) - 1)))
            elif btn in ("left", "right"):
                d["col"] = (d["col"] + (1 if btn == "right" else -1)) % len(rows[d["row"]])
            elif btn == "b":
                d["text"] = d["text"][:-1]
            elif btn == "start":
                self.keyboard_ok()
            elif btn == "a":
                key = rows[d["row"]][d["col"]]
                if key == "DEL":
                    d["text"] = d["text"][:-1]
                elif key == "CLEAR":
                    d["text"] = ""
                elif key == "OK":
                    self.keyboard_ok()
                elif key == d["cancel_label"]:
                    self._close_dialog_and(d["on_cancel"])
                elif len(d["text"]) < NAME_MAX:
                    d["text"] += key
        elif d["kind"] == "confirm":
            if btn == "a":
                self._close_dialog_and(d["on_yes"])
            elif btn == "b":
                self._close_dialog_and(d["on_no"])
        elif d["kind"] == "menu":
            if btn in ("up", "down"):
                d["idx"] = (d["idx"] + (1 if btn == "down" else -1)) % len(d["items"])
            elif btn == "a":
                self._close_dialog_and(d["items"][d["idx"]][1])
            elif btn == "b":
                self.dialog = None

    def keyboard_ok(self):
        d = self.dialog
        name = clean_name(d["text"])
        if not name:
            self.show_toast("Type a name first")
            return
        self._close_dialog_and(d["on_ok"], name)

    def dialog_key(self, ev):
        """Computer keyboard while a dialog is open: type the name directly."""
        d = self.dialog
        k = ev.key
        if d["kind"] == "keyboard":
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.keyboard_ok()
            elif k == pygame.K_ESCAPE:
                self._close_dialog_and(d["on_cancel"])
            elif k == pygame.K_BACKSPACE:
                d["text"] = d["text"][:-1]
            elif k in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                self.dialog_button({pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left",
                                    pygame.K_RIGHT: "right"}[k])
            else:
                ch = (ev.unicode or "").upper().replace(" ", "_")
                if ch and ch in NAME_CHARS and len(d["text"]) < NAME_MAX:
                    d["text"] += ch
        else:
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_z):
                self.dialog_button("a")
            elif k in (pygame.K_ESCAPE, pygame.K_x, pygame.K_BACKSPACE):
                self.dialog_button("b")
            elif k in (pygame.K_UP, pygame.K_DOWN):
                self.dialog_button("up" if k == pygame.K_UP else "down")
        self.dirty = True

    def draw_dialog(self):
        L = self.lcd
        d = self.dialog
        if d["kind"] == "keyboard":
            L.erase(0, 15, LCD_W, LCD_H - 15)
            L.text(2, 19, d["title"])
            L.rtext(254, 19, f"{len(d['text'])}/{NAME_MAX}")
            L.frame(2, 30, 252, 22)
            w = L.text(6, 34, d["text"], scale=2)
            if int(time.time() * 2) % 2 == 0 and len(d["text"]) < NAME_MAX:
                L.fill(6 + w + (3 if d["text"] else 0), 46, 10, 2)          # blinking cursor
            rows = self.kb_rows(d)
            for r, keys in enumerate(rows):
                y = 60 + r * 19
                action = r == len(rows) - 1
                cw = 252 // len(keys) if action else 19
                for c, key in enumerate(keys):
                    x = 2 + c * cw
                    sel = r == d["row"] and c == d["col"]
                    if sel:
                        L.bar(x, y - 3, cw - 2, 13)
                        L.ctext(x + (cw - 2) // 2, y, key, inv=True)
                    else:
                        if action:
                            L.frame(x, y - 3, cw - 2, 13)
                        L.ctext(x + (cw - 2) // 2, y, key)
            L.hline(0, 140, LCD_W, dotted=True)
            if self.last_input == "pad":
                hints = ["D-PAD MOVE   A TYPE   B DELETE", "START OK"]
            else:
                hints = ["TYPE THE NAME   BACKSPACE DELETE", "ENTER OK   ESC " + d["cancel_label"]]
            for i, h in enumerate(hints):
                L.ctext(128, 147 + i * 12, h)
            return
        if d["kind"] == "confirm":
            lines = [d["title"]] + list(d["lines"])
        else:
            lines = [d["title"]]
        w = 220
        h = (len(lines) * 12 + 30) if d["kind"] == "confirm" else (len(d["items"]) * 13 + 26)
        x, y = (LCD_W - w) // 2, max(18, (LCD_H - h) // 2)
        L.erase(0, y - 3, LCD_W, h + 6)                   # clear the whole band so nothing peeks out at the sides
        L.frame(x, y, w, h)
        L.frame(x + 1, y + 1, w - 2, h - 2)
        if d["kind"] == "confirm":
            for i, line in enumerate(lines):
                L.ctext(128, y + 6 + i * 12, L.fit(line, w - 10))
            yy = y + h - 15
            L.bar(x + 10, yy - 2, 96, 11)
            L.ctext(x + 58, yy, f"A {d['yes']}", inv=True)
            L.ctext(x + w - 58, yy, "B CANCEL")
        else:
            L.ctext(128, y + 6, L.fit(d["title"], w - 10))
            L.hline(x + 4, y + 16, w - 8, dotted=True)
            for i, (label, _) in enumerate(d["items"]):
                yy = y + 22 + i * 13
                if i == d["idx"]:
                    L.bar(x + 4, yy - 2, w - 8, 11)
                    L.text(x + 8, yy, label, inv=True)
                else:
                    L.text(x + 8, yy, label)

    def show_toast(self, msg, secs=2.0):
        self.dirty = True
        self.toast = (msg, time.time() + secs)

    # ---------- params
    def page_params(self):
        params = [p for col in self.page_layout() for _, ps in col for p in ps]
        return TRACK_PARAMS + params if PAGES[self.page] == "LOOP" else params

    def target(self, p):
        if p.scope == "ui":
            return self.ui
        if p.scope == "loop":
            return self.state.looper
        if p.scope == "fx":
            return self.state.fx
        if p.scope == "rec":
            return self.state.rec
        if p.scope == "lfo":
            return self.state.lfo
        return self.state.cur_slot().p

    def value_text(self, p):
        if p.key == "bpm" and p.scope == "loop" and self.engine.looper and self.engine.loop["length"] \
                and self.engine.loop["beat"] > 0:
            return f"{60 * SR / self.engine.loop['beat']:.1f}"
        if p.kind == "track":
            return f"{round(self.state.looper['vols'][int(p.key[5:])] * 100)}%"
        if p.kind == "pad":
            i = self.state.sel_pad
            return f"{i + 1}  {note_name(KIT_BASE_NOTE + i)}"
        if p.kind == "file":
            return self.state.cur_slot().name
        if p.kind == "action":
            if p.key == "record":
                return f"STOP  {self.recorder.frames / SR:.1f}s" if self.recorder.recording else "Press A"
            if p.key == "pname":
                return self.ui["project"]
            if p.key == "ccmap":
                return f"{len(self.ui['ccmap'])} CC"
            if p.key == "chop" and self.job is not None:
                return "WORKING"
            if p.key == "stretch" and self.job is not None:
                return f"{self.job.get('progress', 0) * 100:.0f}%"
            return "Press A"
        if p.kind == "source":
            return self.source_label(self.ui["rec_source"])
        if p.kind == "usbin":
            k = self.usb_input()
            return f"Input {k[1] + 1}" if k else "Input 1"
        if p.scope == "ui":
            if p.key == "latency":
                idx = int(self.ui["latency"])
                frames = self.audio.FRAMES[idx] if hasattr(self.audio, "FRAMES") else LATENCY_BLOCKS[idx] * BLOCK
                return f"{p.opts[int(self.ui['latency'])]}  {frames / SR * 1000:.0f} ms"
            if p.kind == "lin":
                return p.fmt(self.ui[p.key])
            return p.opts[int(self.ui[p.key])]
        v = self.target(p)[p.key]
        if p.scope == "slot" and p.key in ("lstart", "lend"):
            sp = self.target(p)
            v = clamp(v, sp["start"], sp["end"])          # what really plays
        if p.kind == "enum":
            return p.opts[int(v)]
        return p.fmt(v)

    def adjust(self, p, direction, fine):
        if p.key == "bpm" and p.scope == "loop" and self.engine.looper and self.engine.loop["length"]:
            self.show_toast("The loop sets the tempo now: clear all to change it", 2.5)
            return
        if p.kind == "track":
            vols = self.state.looper["vols"]
            i = int(p.key[5:])
            vols[i] = round(clamp(vols[i] + direction * (0.01 if fine else 0.05), 0.0, 1.5), 2)
            return
        if p.kind == "pad":
            self.state.sel_pad = (self.state.sel_pad + direction) % NUM_PADS
            return
        if p.kind == "file":
            self.cycle_sample(direction)
            return
        if p.kind == "action":
            return
        if p.kind == "source":
            ids = ["internal"] + [c for c, _ in capture_cards()]
            cur = self.ui["rec_source"]
            i = ids.index(cur) if cur in ids else 0
            self.ui["rec_source"] = ids[clamp(i + direction, 0, len(ids) - 1)]
            return
        if p.kind == "usbin":
            k = self.usb_input()
            if k is None:
                return
            devs = capture_devices(self.ui["rec_source"])
            i = clamp(k[1] + direction, 0, len(devs) - 1)      # one device only: stays on Input 1
            self.ui["usb_inputs"][self.ui["rec_source"]] = devs[i]
            return
        d = self.target(p)
        v = d[p.key]
        if p.kind == "enum":
            v = clamp(int(v) + direction, 0, len(p.opts) - 1)
        elif p.kind == "note":
            v = int(clamp(v + direction * (12 if fine else 1), p.lo, p.hi))
        elif p.kind == "exp":
            v = clamp(v * (1 + p.step) ** (direction * (0.15 if fine else 1)), p.lo, p.hi)
        else:
            step = p.step if p.step >= 1 else p.step * (0.1 if fine else 1)
            v = round(clamp(v + direction * step, p.lo, p.hi), 6)
            if p.step >= 1:
                v = int(v)
        if p.scope == "slot":
            if p.key == "start":
                v = min(v, d["end"] - 0.002)
            elif p.key == "end":
                v = max(v, d["start"] + 0.002)
            elif p.key == "lstart":
                v = clamp(max(v, d["start"]), d["start"], min(d["lend"], d["end"]) - 0.002)
            elif p.key == "lend":
                v = clamp(min(v, d["end"]), max(d["lstart"], d["start"]) + 0.002, d["end"])
        d[p.key] = v
        if p.scope == "slot" and p.key in ("start", "end"):      # loop points stay inside the region
            d["lstart"] = clamp(d["lstart"], d["start"], d["end"])
            d["lend"] = clamp(d["lend"], d["start"], d["end"])

    def change_volume(self, direction):
        global OUTPUT_VOLUME
        v = round(clamp(self.ui["volume"] + 0.05 * direction, 0.0, 1.0), 2)
        self.ui["volume"] = v
        OUTPUT_VOLUME = v ** 2
        if hasattr(self.engine, "push_fx"):
            with self.engine.lock:
                self.engine.push_fx()
        self.save_settings()
        self.show_toast(f"VOLUME {round(v * 100)}%", 1.2)

    def usb_input(self, log=False):
        """(device number, position) of the chosen recording device of the USB source, or None (built-in
        mic, or the card is not plugged in). A device that is gone falls back to the first one (DEV=0)."""
        src = self.ui["rec_source"]
        if src == "internal":
            return None
        devs = capture_devices(src)
        if not devs:
            return None
        want = self.ui["usb_inputs"].get(src, devs[0])
        if want not in devs:
            if log:
                print(f"loop: card {src} has no recording device {want} now (it has {devs}): "
                      f"using device {devs[0]} (Input 1)", flush=True)
            want = devs[0]
        return want, devs.index(want)

    def source_label(self, cid):
        """Built-in mic, or USB: any USB interface / mic / synth (numbered when more than one is plugged in)."""
        if cid == "internal":
            return "Internal mic"
        ids = [c for c, _ in capture_cards()]
        if cid not in ids:
            return "USB (off)"
        return "USB" if len(ids) == 1 else f"USB {ids.index(cid) + 1}"

    def cycle_sample(self, direction):
        files = self.samples_cache = list_samples()
        if not files:
            return
        slot = self.state.cur_slot()
        try:
            i = files.index(slot.path)
        except ValueError:
            i = -1 if direction > 0 else 0
        self.load_into_slot(files[(i + direction) % len(files)])

    def load_into_slot(self, path):
        try:
            self.state.cur_slot().load(path)
            self.show_toast("Loaded " + os.path.basename(path))
        except Exception as e:
            self.show_toast(f"Can't load: {e}", 4)

    # ---------- input
    def press(self, btn):
        now = time.time()
        self.held[btn] = (now, now + 0.32)
        self.on_button(btn)

    def release_btn(self, btn):
        t0 = self.held.pop(btn, (time.time(), 0))[0]
        self.dirty = True
        if btn == "select" and time.time() - t0 > 0.45:
            self.select_combo = True     # long press only showed the help screen
        if btn == "x" and self.audition_note is not None:
            self.engine.note_off(self.audition_note)
            self.audition_note = None
        if btn == "select":
            pass                                # hold SELECT: controls help; tap: nothing (no kit mode)
            self.select_combo = False

    def toggle_mode(self):
        self.engine.all_off()
        self.state.mode = "kit" if self.state.mode == "chromatic" else "chromatic"
        self.show_toast("Mode: " + self.state.mode.upper())

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
            self.change_volume(1 if btn == "vol_up" else -1)
            return
        self.dirty = True
        fine = "y" in self.held
        if self.dialog is not None:
            self.dialog_button(btn)
            return
        if btn == "start":
            if "select" in self.held:
                self.select_combo = True
                self.running = False
            else:
                self.save_project()
            return
        if self.browser is not None:
            self.browser_button(btn)
            return
        if self.engine.learn is not None and btn == "b":
            self.engine.learn = None
            self.show_toast("MIDI learn cancelled")
            return
        if PAGES[self.page] == "LOOP" and self.loop_button(btn, repeat):
            return
        params = self.page_params()
        sel = self.sel[self.page] = clamp(self.sel[self.page], 0, len(params) - 1)
        p = params[sel]
        if btn in ("up", "down"):
            self.sel[self.page] = (sel + (1 if btn == "down" else -1)) % len(params)
            if PAGES[self.page] == "LOOP" and self.sel[self.page] < LOOP_TRACKS:
                self.loop_track = self.sel[self.page]
        elif btn in ("left", "right"):
            self.adjust(p, -1 if btn == "left" else 1, fine)
            if p.key == "display":
                self.lcd = self.make_lcd()
                self.save_settings()
            elif p.key == "latency":
                if hasattr(self.audio, "set_latency"):
                    self.audio.set_latency(self.ui["latency"])
                self.save_settings()
            elif p.key == "loop_offset":
                self.save_settings()
            elif p.key in ("rec_source", "rec_channel", "usb_in") and not self.recorder.recording:
                self.recorder.set_source(self.ui["rec_source"], self.ui["rec_channel"])
                self.update_capture()
                self.save_settings()
                if self.recorder.usb and self.recorder.dev is None:
                    self.recorder.open_async()
        elif btn == "l1" and not repeat:
            self.set_page(self.page - 1)
        elif btn == "r1" and not repeat:
            self.set_page(self.page + 1)
        elif btn in ("l2", "r2") and self.state.mode == "kit":
            self.state.sel_pad = (self.state.sel_pad + (1 if btn == "r2" else -1)) % NUM_PADS
        elif btn == "a":
            if p.kind == "file":
                self.open_browser()
            elif p.kind == "action":
                self.do_action(p.key)
            elif PAGES[self.page] == "FX" and p.key in FX_SWITCH:
                flag = FX_SWITCH[p.key]
                self.state.fx[flag] = 0 if self.state.fx[flag] else 1
                name = [t for t, f in FX_SECTION_SWITCH.items() if f == flag][0]
                self.show_toast(f"{name} {'ON' if self.state.fx[flag] else 'OFF'}")
        elif btn == "b" and PAGES[self.page] in ("FX", "LFO") and p.kind != "action":
            self.learn_menu(p)

    # ---------- looper
    def loop_button(self, btn, repeat):
        """A / X / B / L2 / R2 on the LOOP page. They act on the current track wherever the cursor is."""
        if btn not in ("a", "x", "b", "l2", "r2") or repeat:
            return False
        e = self.engine
        if btn in ("l2", "r2"):
            self.loop_track = (self.loop_track + (1 if btn == "r2" else -1)) % LOOP_TRACKS
            if self.sel[self.page] < LOOP_TRACKS:
                self.sel[self.page] = self.loop_track
            return True
        if not e.looper:
            self.show_toast("The looper needs the native engine (see log.txt)", 3)
            return True
        t, shift = self.loop_track, "y" in self.held
        tr = e.loop["tracks"][t]
        if btn == "a" and shift:
            if tr["fresh"] or tr["dub"]:
                self.show_toast("Wait for the take to finish")
            elif tr["undo"] == 1:
                e.loop_cmd(LC_UNDO, t)
                self.show_toast(f"UNDO TRACK {t + 1}")
            elif tr["undo"] == 2:
                e.loop_cmd(LC_UNDO, t)
                self.show_toast(f"REDO TRACK {t + 1}")
            else:
                self.show_toast("Nothing to undo")
        elif btn == "a":
            if tr["state"] == 0:
                self.warn_loop_input()
            e.loop_cmd(LC_RECDUB, t)
        elif btn == "x" and shift:
            if tr["state"] != 0 or tr["fresh"]:
                e.loop_cmd(LC_CLEAR, t)
                self.show_toast(f"TRACK {t + 1} CLEARED")
        elif btn == "x":
            e.loop_cmd(LC_STOP, t)
        elif btn == "b" and shift:
            if any(x["state"] != 0 for x in e.loop["tracks"]):
                self.confirm("CLEAR ALL TRACKS?", ["THE LOOPS ARE LOST", "(SAVED FILES TOO, ON SAVE)."],
                             lambda: (e.loop_cmd(LC_CLEARALL), self.show_toast("ALL TRACKS CLEARED")), yes="CLEAR")
        elif btn == "b":
            e.loop_cmd(LC_ALL)
        return True

    def warn_loop_input(self):
        """Say what will happen with the input this take records from."""
        if self.ui["rec_source"] == "internal":
            if ON_DEVICE:
                self.show_toast("Mic: the sound stops while you record", 2.5)
        elif not self.engine.loop["cap_open"]:
            self.show_toast("USB input not connected: " + (self.engine.capture_error() or "see log.txt"), 3)

    def log_usb_cards(self):
        """log.txt: the recording devices of each USB card"""
        cards = capture_cards()
        if not cards:
            print("loop: no USB recording device plugged in", flush=True)
        for i, (cid, _) in enumerate(cards):
            n = card_index(cid)
            print(f"loop: USB {i + 1} = card {n} ({cid}): recording devices {capture_devices(cid)}", flush=True)

    def update_capture(self):
        """The USB input (interface, mic, synth) is opened by the engine and shared by the looper and REC."""
        e = self.engine
        if not getattr(self, "_cards_logged", False) and use_alsa_tools():
            self._cards_logged = True
            self.log_usb_cards()
        if not getattr(e, "has_capture", False):
            return
        src = self.ui["rec_source"]
        if src != "internal" and use_alsa_tools():
            e.set_mic(False)
            n = card_index(src)
            self.cap_present = n is not None
            if n is None:
                self._mixer_done = {x for x in getattr(self, "_mixer_done", set()) if x[0] != src}   # set again on plug-in
                e.set_capture("", 0)
                print(f"loop: input {src} is not plugged in (waiting for it)", flush=True)
                return
            k = self.usb_input(log=True)
            d, pos = k if k else (0, 0)
            fmts, chans, rates = capture_stream(n, d)
            rate = pick_rate(rates, 48000) or SR
            fmt = pick_format(fmts)
            if (src, n) not in getattr(self, "_mixer_done", set()):
                self._mixer_done = getattr(self, "_mixer_done", set()) | {(src, n)}
                usb_mixer_setup(n)
            e.cap_format(fmt, rate)
            self.ui["usb_dev"] = d
            devs = [f"plughw:CARD={src},DEV={d}", f"hw:CARD={src},DEV={d}", f"plughw:{n},{d}", f"hw:{n},{d}"]
            e.set_capture("\n".join(devs), self.ui["rec_channel"])
            self._hw_logged = None
            self._cap_dev = (n, d)
            rtxt = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in rates) or "?"
            print(f"loop: input {src} (card {n}), device {d} = Input {pos + 1} of {len(capture_devices(src))}, "
                  f"channel {REC_CHANNELS[self.ui['rec_channel']]}; the card says: formats "
                  f"{'/'.join(fmts) or '?'}, channels {'/'.join(map(str, chans)) or '?'}, rates {rtxt}", flush=True)
            print(f"loop: opening {devs[0]} as {fmt}, {rate} Hz"
                  + (f" (converted to {SR} Hz in the code)" if rate != SR else ""), flush=True)
        else:
            e.set_capture("", 0)
            on = src == "internal" and ON_DEVICE
            e.set_mic(on)
            if on:
                threading.Thread(target=Recorder.configure_builtin_mic, daemon=True).start()
            print("loop: input built-in mic (the speaker goes quiet while it records)", flush=True)

    def watch_input(self):
        """Every 2 s: open the USB input when it gets plugged in. Every 10 s: a status line in log.txt."""
        now = time.time()
        if now < getattr(self, "_watch_t", 0):
            return
        self._watch_t = now + 2.0
        e, src = self.engine, self.ui["rec_source"]
        if not e.looper:
            return
        if src != "internal" and use_alsa_tools():
            present = card_index(src) is not None
            if present != getattr(self, "cap_present", None):
                if present:
                    self.log_usb_cards()
                self.update_capture()
        cd = getattr(self, "_cap_dev", None)
        if src != "internal" and cd and e.loop["cap_open"] and getattr(self, "_hw_logged", None) != cd:
            self._hw_logged = cd
            hw = read_hw_params(*cd)
            print(f"loop: open {src} card {cd[0]} device {cd[1]}: {hw or 'card details not available'}"
                  f" (engine period {e.loop['cap_period']})", flush=True)
            e.cap_level()                                  # start measuring now
            self._lvl_due = (cd, now + 1.9)
        due = getattr(self, "_lvl_due", None)
        if due and now >= due[1]:
            self._lvl_due = None
            lv = e.cap_level() if src != "internal" and e.loop["cap_open"] else None
            if lv:
                ch = int(e.loop["cap_ch"] or 2)
                if not lv[2]:
                    txt = "silent (all 0)"
                else:
                    txt = ", ".join(f"{nm} {20 * math.log10(max(x, 1e-9)):.1f} dBFS"
                                    for nm, x in (("L", lv[0]), ("R", lv[1]))[:ch])
                print(f"loop: level of card {due[0][0]} device {due[0][1]} after 2 s: {txt}", flush=True)
        if now >= getattr(self, "_log_t", 0):
            self._log_t = now + 10.0
            lp = e.loop
            db = lambda x: f"{20 * math.log10(max(x, 1e-6)):.0f}dB"
            print(f"loop: input={src} dev={cd[1] if cd and src != 'internal' else '-'} open={int(lp['cap_open'])} ch={lp['cap_ch']} reads={lp['cap_reads']} "
                  f"usb_peak={db(lp['cap_peak'])} in_peak={db(max(lp['in_peak']))} under={lp['in_under']} "
                  f"drift={lp.get('drift', 0):+.0f}ppm "
                  f"mic_on={int(lp['mic_on'])} err='{e.capture_error()}' out_peak={db(max(e.peak))} "
                  f"tracks={[LOOP_STATES[t['state']] for t in lp['tracks']]}", flush=True)

    def input_status(self):
        lp = self.engine.loop if self.engine.looper else {}
        if self.ui["rec_source"] == "internal":
            return "MIC (REC)" if lp.get("mic_on") else "BUILT-IN MIC"
        if lp.get("cap_open"):
            ids = [c for c, _ in capture_cards()]
            src, k = self.ui["rec_source"], self.usb_input()
            if src in ids and k:
                return f"USB {ids.index(src) + 1} IN {k[1] + 1}"
            return self.source_label(src).upper()
        return "NOT CONNECTED"

    def update_comp(self):
        """How late sounds reach the speaker, and the input reaches the engine, so takes land in time."""
        e = self.engine
        if not e.looper:
            return
        a = self.audio
        if isinstance(a, NativeAlsaOut):
            out = e.buffer_frames or a.FRAMES[a.latency]
        elif getattr(a, "buffer_frames", 0):
            out = a.buffer_frames
        elif isinstance(a, AplayOut):
            out = a.chunk * 4
        else:
            out = DEVICE_CHUNK * 2
        lp = e.loop
        inp = lp["cap_period"] + lp["ring_target"] + NATIVE_BLOCK
        off = int(self.ui["loop_offset"] * SR / 1000)
        e.comp = (int(out), int(max(out, out + inp + off)))

    def loop_columns(self, t, w):
        """Waveform of a track, w columns (cached until its audio changes)."""
        tr = self.engine.loop["tracks"][t]
        key = (tr["ver"], tr["len"], w)
        c = self.loop_wave.get(t)
        if c is None or c[0] != key:
            n = tr["len"]
            x = np.abs(self.engine.loop_views[t][:n]).max(axis=1)
            edges = np.minimum(np.linspace(0, n, w + 1).astype(np.int64)[:-1], n - 1)
            pk = np.maximum.reduceat(x, edges)
            pk = pk / max(0.05, float(pk.max()))
            c = self.loop_wave[t] = (key, pk)
        return c[1]

    def set_page(self, i):
        self.page = i % len(PAGES)
        # The mic stays open once opened: closing SDL devices while their Python callback
        # is waiting for the GIL can deadlock, so devices are only released at process exit.
        if PAGES[self.page] == "REC" and self.recorder.dev is None:
            if self.recorder.usb and use_alsa_tools():
                self.recorder.open_async()
            elif not self.recorder.open():
                self.show_toast("Mic: " + (self.recorder.error or "unavailable"), 4)

    def toggle_record(self):
        rec = self.recorder
        if rec.starting:
            return
        if rec.recording:
            x = rec.stop()
            if x is None or len(x) < SR * 0.05:
                self.show_toast("No audio from the mic: " + rec.error if rec.error else
                                "No audio from the mic" if x is None else "Recording too short", 4)
                return
            self.finish_take(x, list(rec.peaks))
        elif not rec.starting:
            rec.start_async()

    def open_browser(self, kind="samples"):
        if kind == "projects":
            items = self.list_projects()
            if not items:
                self.show_toast("No saved projects yet")
                return
            cur = self.ui["project"]
            idx = items.index(cur) if cur in items else 0
        else:
            items = self.samples_cache = list_samples()
            if not items:
                self.show_toast("No WAV files in samples/")
                return
            cur = self.state.cur_slot().path
            idx = items.index(cur) if cur in items else 0
        self.browser = dict(kind=kind, files=items, idx=idx)

    def refresh_browser(self, select=None):
        b = self.browser
        if b is None:
            return
        b["files"] = self.list_projects() if b["kind"] == "projects" else list_samples()
        if not b["files"]:
            self.browser = None
            return
        if select in b["files"]:
            b["idx"] = b["files"].index(select)
        b["idx"] = clamp(b["idx"], 0, len(b["files"]) - 1)

    def browser_button(self, btn):
        b = self.browser
        n = len(b["files"])
        item = b["files"][b["idx"]]
        if btn in ("up", "down"):
            b["idx"] = (b["idx"] + (1 if btn == "down" else -1)) % n
        elif btn in ("l1", "r1"):
            b["idx"] = clamp(b["idx"] + (8 if btn == "r1" else -8), 0, n - 1)
        elif btn == "b":
            self.browser = None
        elif b["kind"] == "projects":
            if btn == "a":
                self.browser = None
                if item != self.ui["project"]:
                    self.switch_project(item)
            elif btn == "y":
                self.project_menu(item)
        else:
            if btn == "a":
                self.load_into_slot(item)
                self.browser = None
            elif btn == "x":
                self.load_into_slot(item)
                note = self.state.cur_slot().p["root"] if self.state.mode == "chromatic" else KIT_BASE_NOTE + self.state.sel_pad
                self.audition_note = note
                self.engine.note_on(note, 110)
            elif btn == "y":
                self.sample_menu(item)

    def sample_menu(self, path):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]

        def rename():
            def done(new):
                target = self.rename_sample(path, new)
                self.refresh_browser(target)
            self.ask_name("RENAME SAMPLE", stem, done)

        def delete():
            used = self.slots_using(path)
            lines = ["THE FILE IS REMOVED FROM THE SD CARD."]
            if used:
                lines.append("USED BY " + ", ".join(used[:3]) + ("..." if len(used) > 3 else ""))

            def really():
                self.confirm("ARE YOU SURE?", ["THIS CAN'T BE UNDONE."],
                             lambda: (self.delete_sample(path), self.refresh_browser()), yes="DELETE")
            self.confirm(f"DELETE {base}?", lines, really, yes="YES")
        self.menu(base, [("RENAME", rename), ("DELETE", delete), ("CANCEL", None)])

    def project_menu(self, name):
        def rename():
            self.ask_name("RENAME PROJECT", name, lambda n: (self.rename_project(name, n), self.refresh_browser(n)))

        def delete():
            if name == self.ui["project"]:
                self.show_toast("Can't delete the open project", 3)
                return

            def really():
                self.confirm("ARE YOU SURE?", ["THIS CAN'T BE UNDONE."],
                             lambda: (self.delete_project(name), self.refresh_browser()), yes="DELETE")
            self.confirm(f"DELETE PROJECT {name}?", ["ITS LOOPS GO TOO.", "SAMPLES ARE KEPT."], really, yes="YES")
        self.menu(name, [("LOAD", lambda: (setattr(self, "browser", None), self.switch_project(name))
                          if name != self.ui["project"] else None),
                         ("RENAME", rename), ("DELETE", delete), ("CANCEL", None)])

    def handle_events(self):
        for ev in pygame.event.get():
            t = ev.type
            if t == pygame.QUIT:
                self.running = False
            elif t == pygame.KEYDOWN:
                self.last_input = "keyboard"
                if self.dialog is not None:
                    self.dialog_key(ev)
                    continue
                mods = pygame.key.get_mods()
                if mods & (pygame.KMOD_CTRL | pygame.KMOD_META):
                    if ev.key == pygame.K_s:
                        self.save_project()
                    elif ev.key == pygame.K_q:
                        self.running = False
                    continue
                sc = ev.scancode
                if sc in self.piano_map:
                    if sc not in self.piano_held:
                        base = KIT_BASE_NOTE if self.state.mode == "kit" else 12 * (self.state.kb_octave + 1)
                        note = base + self.piano_map[sc]
                        self.piano_held[sc] = note
                        self.engine.note_on(note, 100)
                elif sc in self.key_map and self.key_map[sc] not in self.held:
                    self.press(self.key_map[sc])
            elif t == pygame.KEYUP:
                sc = ev.scancode
                if sc in self.piano_held:
                    self.engine.note_off(self.piano_held.pop(sc))
                elif sc in self.key_map:
                    self.release_btn(self.key_map[sc])
            elif t == getattr(pygame, "CONTROLLERBUTTONDOWN", -1):
                self.last_input = "pad"
                btn = self.ctrl_button(ev.button)
                if self.args.debug_input:
                    self.show_toast(f"controller button {ev.button} -> {btn}")
                    print(f"input: controller button {ev.button} -> {btn}", flush=True)
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
                            self.last_input = "pad"
                            self.press(btn)
                        elif ev.value < 8000 and btn in self.held:
                            self.release_btn(btn)
            elif t in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
                if self.controllers:
                    continue   # handled as controller events
                btn = JOY_BUTTONS.get(ev.button)
                if t == pygame.JOYBUTTONDOWN and self.logged_inputs < 40:
                    self.logged_inputs += 1
                    print(f"input: joystick button {ev.button} -> {btn}", flush=True)
                if self.args.debug_input and t == pygame.JOYBUTTONDOWN:
                    self.show_toast(f"joystick button {ev.button} -> {btn}")
                    print(f"input: joystick button {ev.button} -> {btn}", flush=True)
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
            elif t == pygame.JOYAXISMOTION and not self.controllers and ev.axis not in (0, 1):
                if abs(ev.value) > 0.5 and self.logged_inputs < 40:
                    self.logged_inputs += 1
                    print(f"input: joystick axis {ev.axis} = {ev.value:+.2f} (not mapped)", flush=True)
            elif t == pygame.JOYAXISMOTION and not self.controllers and ev.axis in (0, 1):
                neg, pos = ("left", "right") if ev.axis == 0 else ("up", "down")
                for btn, on in ((neg, ev.value < -0.6), (pos, ev.value > 0.6)):
                    if on and btn not in self.held:
                        self.last_input = "pad"
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

    # ---------- LCD drawing (all coordinates in dots, 256 x 192)
    def page_layout(self):
        """Columns of (section title, params). Up/down walks the left column, then the right."""
        name = PAGES[self.page]
        if name == "LOOP":
            return [[("LOOPER", [P_SOURCE, P_USBIN] + LOOPER_PARAMS)], [("METRONOME", METRO_PARAMS)]]
        if name == "SAMPLE":
            left = ([P_FILE, P_ROOT] if self.state.mode == "chromatic" else [P_PAD, P_FILE]) + SAMPLE_PARAMS_TAIL[:4]
            rest = SAMPLE_PARAMS_TAIL[4:]
            return [[("SAMPLE", left)], [("REGION", rest[:2]), ("PLAYBACK", rest[2:])]]
        if name == "EDIT":
            pad = [P_PAD] if self.state.mode == "kit" else []
            return [[("LOOP", pad + LOOP_PARAMS)], [("TOOLS", TOOL_PARAMS)]]
        if name == "LFO":
            return [[("LFO", LFO_PARAMS[:4])], [("DEPTH", LFO_PARAMS[4:])]]
        if name == "AMP":
            pad = [P_PAD] if self.state.mode == "kit" else []
            return [[("ENVELOPE", pad + AMP_PARAMS[:4])], [("VELOCITY", AMP_PARAMS[4:])]]
        if name == "FX":
            f = FX_PARAMS
            return [[("DRIVE", f[0:3]), ("FILTER", f[3:6]), ("OUTPUT", f[6:7])], [("DELAY", f[7:12]), ("REVERB", f[12:15])]]
        if name == "REC":
            pad = [P_PAD] if self.state.mode == "kit" else []
            return [[("RECORD", pad + REC_PARAMS)], []]
        return [[("DISPLAY", [P_DISPLAY]), ("AUDIO", [P_LATENCY, P_CLICKVOL]), ("INPUT", [P_MONITOR, P_CHANNEL, P_OFFSET]),
                 ("MIDI CC", [P_CCMAP, P_CCRESET])],
                [("PROJECT", PROJECT_PARAMS)]]

    def draw_sections(self, y0, row_h=13, title_h=12, gap=3, offset=0):
        """Two columns of sections; the selected parameter is a solid bar with the text punched out."""
        L = self.lcd
        sel = self.sel[self.page]
        col_w = 123
        i = offset
        for c, col in enumerate(self.page_layout()):
            x, y = 2 + c * (col_w + 6), y0
            for title, params in col:
                L.text(x, y, title)
                tw = L.width(title)
                flag = FX_SECTION_SWITCH.get(title) if PAGES[self.page] == "FX" else None
                if flag and not self.state.fx[flag]:
                    L.bar(x + tw + 3, y - 2, 21, 11)          # switched-off section: OFF tag
                    L.text(x + tw + 6, y, "OFF", inv=True)
                    tw += 24
                L.hline(x + tw + 2, y + 6, col_w - tw - 2, dotted=True)
                y += title_h
                for p in params:
                    is_sel = i == sel and self.browser is None
                    if is_sel:
                        L.bar(x - 1, y - 2, col_w + 2, 11)
                    lw = L.text(x + 1, y, p.label, inv=is_sel)
                    val = L.fit(self.value_text(p), col_w - lw - 10)
                    L.rtext(x + col_w - 2, y, val, inv=is_sel)
                    i += 1
                    y += row_h
                y += gap

    def draw_header(self):
        L = self.lcd
        x = 1
        for i, name in enumerate(PAGES):
            name = TAB_LABEL.get(name, name)
            w = L.width(name) + 6
            if i == self.page:
                L.bar(x, 1, w, 11)
                L.text(x + 3, 3, name, inv=True)
            else:
                L.text(x + 3, 3, name)
            x += w + 2
        tabs_end = x
        xr = 254
        cpu_warn = self.engine.load > 0.85
        if cpu_warn:                         # CPU and voices live on SETTINGS; only warn up here
            xr -= L.rtext(xr, 3, "CPU!") + 6
        has_midi = bool(self.midi.devices)
        active = time.time() - self.engine.midi_activity < 0.12
        if has_midi or active:
            xr -= L.rtext(xr, 3, "MIDI") + 2
            if active:
                L.fill(xr - 5, 4, 5, 5)
            else:
                L.frame(xr - 5, 4, 5, 5)
            xr -= 11
        _ = tabs_end
        L.hline(0, 14, LCD_W)
        if self.engine.error or not self.audio.ok:
            self.toast = ("AUDIO ERROR - SEE LOG.TXT", time.time() + 1)

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
        if PAGES[self.page] == "LOOP" and self.last_input == "pad":
            title, rows = "LOOPER", [("A", "REC / DUB / PLAY"), ("X", "STOP / PLAY TRACK"), ("B", "START / STOP ALL"),
                                      ("Y + A", "UNDO / REDO"), ("Y + X", "CLEAR TRACK"), ("Y + B", "CLEAR ALL"),
                                      ("UP / DOWN", "TRACK / SETTING"), ("LEFT / RIGHT", "VOLUME / VALUE"),
                                      ("L2 / R2", "TRACK"), ("L1 / R1", "PAGE"), ("START", "SAVE"),
                                      ("SELECT+START", "SAVE AND QUIT")]
        elif PAGES[self.page] == "LOOP":
            title, rows = "LOOPER KEYS", [("Z / ENTER", "REC / DUB / PLAY"), ("SPACE", "STOP / PLAY TRACK"),
                                           ("X / ESC", "START / STOP ALL"), ("SHIFT+Z", "UNDO / REDO"),
                                           ("SHIFT+SPACE", "CLEAR TRACK"), ("SHIFT+X", "CLEAR ALL"),
                                           ("UP / DOWN", "TRACK / SETTING"), ("LEFT / RIGHT", "VOLUME / VALUE"),
                                           (", .", "TRACK"), ("[ ]", "PAGE"), ("CTRL+S / Q", "SAVE / QUIT")]
        elif self.last_input == "pad":
            title, rows = "CONTROLS", [("UP / DOWN", "SELECT"), ("LEFT / RIGHT", "CHANGE VALUE"), ("HOLD Y", "FINE ADJUST"),
                                        ("A", "SWITCH / OPEN"), ("B", "BACK / MIDI LEARN"), ("L1 / R1", "PAGE"),
                                        ("VOLUME +/-", "OUTPUT LEVEL"), ("START", "SAVE"), ("SELECT+START", "SAVE AND QUIT")]
        else:
            title, rows = "KEYBOARD", [("UP / DOWN", "SELECT"), ("LEFT / RIGHT", "VALUE (SHIFT FINE)"),
                                        ("Z / ENTER", "SWITCH / OPEN"), ("X / ESC", "BACK / MIDI LEARN"),
                                        ("[ ]", "PAGE"), ("CTRL+S / Q", "SAVE / QUIT")]
        L.text(2, 18, title)
        L.rtext(254, 18, "RELEASE TO CLOSE")
        L.hline(0, 28, LCD_W, dotted=True)
        for i, (k, v) in enumerate(rows):
            y = 33 + i * 13
            L.text(4, y, k)
            L.text(100, y, v)

    def slot_title(self):
        st = self.state
        if st.mode == "chromatic":
            return "CHROMATIC"
        return f"PAD {st.sel_pad + 1}  {note_name(KIT_BASE_NOTE + st.sel_pad)}"

    def draw_pads(self, y):
        L = self.lcd
        st = self.state
        now = time.time()
        for i in range(NUM_PADS):
            x = i * 16
            loaded = st.slots[1 + i].data is not None
            hit = now - self.engine.pad_hits[i] < 0.15
            label = str(i + 1)
            if i == st.sel_pad or hit:
                L.bar(x, y, 15, 11)
                L.ctext(x + 8, y + 2, label, inv=True)
            else:
                if loaded:
                    L.frame(x, y, 15, 11)
                L.ctext(x + 8, y + 2, label)

    def content_top(self):
        """Top of the page body; in kit mode the 16 pads sit above it."""
        y = 17
        if self.state.mode == "kit" and PAGES[self.page] in ("SAMPLE", "EDIT", "AMP", "REC"):
            self.draw_pads(y)
            y += 14
        return y

    def wave_columns(self, slot, w):
        key = (id(slot.data), w)
        if self.wave_key != key:
            d = slot.data
            edges = np.minimum(np.linspace(0, len(d), w + 1).astype(np.int64)[:-1], len(d) - 1)
            mx = np.maximum.reduceat(d, edges)
            mn = np.minimum.reduceat(d, edges)
            norm = 1.0 / max(1e-6, float(np.abs(d).max()))
            self.wave_key, self.wave_cols = key, (mn * norm, mx * norm)
        return self.wave_cols

    def draw_waveform(self, slot, x, y, w, h, playheads, loop_marks=False):
        L = self.lcd
        L.frame(x, y, w, h)
        if slot.data is None:
            L.ctext(x + w // 2, y + h // 2 - 3, "EMPTY - SELECT FILE, PRESS A")
            return
        p = slot.p
        iw, ih = w - 2, h - 2
        mid = y + 1 + ih // 2
        mn, mx = self.wave_columns(slot, iw)
        half = ih // 2 - 2
        top = (mid - np.rint(mx * half)).astype(int)
        bot = (mid - np.rint(mn * half)).astype(int)
        s_px, e_px = int(p["start"] * iw), int(p["end"] * iw)
        inside = np.zeros(iw, bool)
        inside[s_px:e_px + 1] = True
        cols = np.arange(iw)
        # inside the region: solid; outside: every other dot
        L.columns(x + 1, np.where(inside, top, LCD_H), np.where(inside, bot, -1))
        L.columns(x + 1, np.where(~inside & (cols % 2 == 0), top, LCD_H), np.where(~inside & (cols % 2 == 0), bot, -1), dotted=True)
        for m in (s_px, min(e_px, iw - 1)):
            L.vline(x + 1 + m, y + 1, ih)
            L.fill(x + m, y + h - 4, 3, 3)
        if loop_marks:
            n = len(slot.data)
            ls, le = loop_bounds(p, n)
            for m in (int(ls / n * iw), min(int(le / n * iw), iw - 1)):
                L.vline(x + 1 + m, y + 1, ih, dotted=True)
                L.fill(x + m, y + 1, 3, 3)
            if p["xfade"] > 0 and p["loop"]:
                xf = max(1, int(p["xfade"] * (le - ls) / n * iw))
                L.hline(x + 1 + int(le / n * iw) - xf, y + 2, xf)
        for fr in playheads:
            col = x + 1 + int(fr * iw)
            if x < col < x + w - 1:
                L.lit[y + 1:y + h - 1, col] ^= True

    def draw_sample(self):
        L = self.lcd
        st = self.state
        slot = st.cur_slot()
        y = self.content_top()
        L.text(2, y + 1, self.slot_title())
        L.rtext(254, y + 1, L.fit(slot.name, 150))
        wh = 56 if st.mode == "chromatic" else 44
        idx = st.cur_index()
        self.draw_waveform(slot, 2, y + 11, 252, wh, [f for i, f in self.engine.playheads if i == idx])
        yi = y + 11 + wh + 2
        if slot.data is not None:
            p = slot.p
            tags = (["LOOP"] if p["loop"] and not p["trigger"] else []) + (["REV"] if p["reverse"] else []) + \
                   (["ONE-SHOT"] if p["trigger"] else ["GATE"])
            L.text(4, yi, "  ".join(tags))
            length = len(slot.data) / SR
            L.rtext(253, yi, f"{length * (p['end'] - p['start']):.2f}S / {length:.2f}S")
        self.draw_sections(yi + 11)

    def draw_edit(self):
        L = self.lcd
        st = self.state
        slot = st.cur_slot()
        y = self.content_top()
        L.text(2, y + 1, self.slot_title())
        L.rtext(254, y + 1, L.fit(slot.name, 150))
        kit = st.mode == "kit"
        oh, sh = (30, 26) if kit else (36, 30)
        idx = st.cur_index()
        self.draw_waveform(slot, 2, y + 11, 252, oh, [f for i, f in self.engine.playheads if i == idx], loop_marks=True)
        self.draw_splice(slot, 2, y + 13 + oh, 252, sh)
        self.draw_sections(y + 17 + oh + sh)

    def draw_splice(self, slot, x, y, w, h):
        """The loop seam up close: the last samples before LOOP END (left) joined to the first samples
        from LOOP START (right), one sample per dot, in the order the sound plays them."""
        L = self.lcd
        L.frame(x, y, w, h)
        cx = x + w // 2
        if slot.data is None:
            return
        p = slot.p
        if not p["loop"] or p["trigger"]:
            L.ctext(cx, y + h // 2 - 3, "LOOP OFF" if not p["loop"] else "ONE-SHOT: NO LOOP")
            return
        d, n = slot.data, len(slot.data)
        ls, le = loop_bounds(p, n)
        k = (w - 4) // 2
        if p["reverse"]:          # backwards: the loop ends at LOOP START and jumps back to LOOP END
            left = d[np.clip(ls + k - 1 - np.arange(k), 0, n - 1)]
            right = d[np.clip(le - 1 - np.arange(k), 0, n - 1)]
        else:
            left = d[np.clip(np.arange(le - k, le), 0, n - 1)]
            right = d[np.clip(np.arange(ls, ls + k), 0, n - 1)]
        sig = np.concatenate([left, right])
        peak = max(float(np.abs(sig).max()), 1e-4)
        mid = y + h // 2
        half = h // 2 - 3
        ys = np.rint(mid - sig / peak * half).astype(int)
        nxt = np.concatenate([ys[1:], ys[-1:]])
        L.hline(x + 2, mid, w - 4, dotted=True)
        L.vline(cx, y + 1, h - 2, dotted=True)
        L.columns(x + 2, np.minimum(ys, nxt), np.maximum(ys, nxt))

    def draw_loop(self):
        L = self.lcd
        e = self.engine
        if not e.looper:
            for i, line in enumerate(["THE LOOPER NEEDS THE NATIVE ENGINE.", "NATIVE/ LIBRARY NOT LOADED:",
                                      "SEE LOG.TXT"]):
                L.ctext(128, 70 + i * 12, line)
            return
        lp, lo = e.loop, self.state.looper
        L0 = lp["length"]
        y = 18
        # transport: PLAY while any track runs (B starts / stops everything)
        playing = any(t["state"] in (1, 2, 3) for t in lp["tracks"])
        if playing:
            L.bar(1, y - 2, 31, 11)
            L.text(4, y, "PLAY", inv=True)
        else:
            L.frame(1, y - 2, 31, 11)
            L.text(4, y, "STOP")
        if lo["met"] and lp["beat"] > 0:
            L.text(37, y, f"{60 * SR / lp['beat']:.0f} BPM {TIME_SIGS[lo['sig']]}")
        else:
            L.text(37, y, f"LOOP {L0 / SR:.2f}S" if L0 else "NO LOOP")
        mx = 132
        if self.ui["rec_source"] == "internal":
            L.text(110, y, "MIC")
            if not lp["mic_on"]:
                L.text(mx, y, "QUIET WHILE REC")
                mx = None
        else:
            L.text(116, y, "IN")
            if not lp["cap_open"]:
                L.text(mx, y, "NOT CONNECTED")
                mx = None
        if mx is not None:
            L.frame(mx, y, 254 - mx, 7)
            for c, pk in enumerate(lp["in_peak"]):
                fr = clamp((20 * math.log10(max(pk, 1e-5)) + 60) / 60, 0, 1)
                if fr > 0:
                    L.fill(mx + 2, y + 2 + c * 2, int((250 - mx) * fr), 1)
        # where we are: the beats of the bar with the click on, else the loop
        if lo["met"] and lp["beat"] > 0:
            nb = max(1, lp["beats"])
            cur = int((lp["now"] - lp["morigin"]) // lp["beat"]) % nb
            w = 252 // nb
            for b in range(nb):
                if b == cur:
                    L.fill(2 + b * w, 26, w - 3, 3)
                else:
                    L.hline(2 + b * w, 27, w - 3, dotted=True)
        elif L0:
            fr = ((lp["now"] - lp["origin"]) % L0) / L0
            L.fill(2, 27, max(1, int(252 * fr)), 2)
        else:
            L.hline(2, 28, 252, dotted=True)
        cursor = self.sel[self.page]
        wx, ww = 46, 160
        for i, tr in enumerate(lp["tracks"]):
            ty, h = 31 + i * 18, 16
            st = tr["state"]
            if i == self.loop_track:
                L.bar(2, ty, 13, h)
                L.ctext(8, ty + 5, str(i + 1), inv=True)
            else:
                L.frame(2, ty, 13, h)
                L.ctext(8, ty + 5, str(i + 1))
            blink = int(time.time() * 4) % 2 == 0
            label = LOOP_STATES[st] if st else "--"
            if st == 3 and lo["dubmode"]:
                label = "REPL"
            counting = tr["fresh"] and st == 1 and tr["pos"] < 0
            if counting:
                label = "CNT"
            elif st == 1 and not blink:
                label = ""
            L.text(18, ty + 1, label)
            if tr["dub"] and st != 3 or tr["closing"]:
                L.text(18, ty + 9, "...")
            elif st and tr["len"] and L0:
                k = tr["len"] / L0
                L.text(18, ty + 9, f"X{k:g}" if abs(k - round(k)) < 1e-6 else f"{tr['len'] / SR:.1f}S")
            L.frame(wx, ty, ww, h)
            iw, mid, half = ww - 2, ty + h // 2, h // 2 - 2
            if counting:
                left = -tr["pos"] / lp["beat"] if lp["beat"] > 0 else 0
                L.ctext(wx + ww // 2, ty + 5, f"COUNT IN  {max(1, math.ceil(left))}")
            elif tr["fresh"] and st == 1:
                done = tr["pos"]
                if L0 and not tr["master"]:
                    n = int(done // L0) + 1
                    fr = (done % L0) / L0
                    if int(iw * fr) > 0:
                        L.columns(wx + 1, np.full(int(iw * fr), mid - 1), np.full(int(iw * fr), mid + 1))
                    L.rtext(wx + ww - 3, ty + 5, f"LOOP {n}")
                else:
                    L.ctext(wx + ww // 2, ty + 5, f"RECORDING {max(0, done) / SR:.1f}S")
            elif st and tr["len"] > 0:
                pk = self.loop_columns(i, iw)
                a = np.rint(pk * half).astype(int)
                if st == 4:
                    L.columns(wx + 1, mid - a, mid + a, dotted=True)
                else:
                    L.columns(wx + 1, mid - a, mid + a)
                    col = wx + 1 + int(tr["pos"] / tr["len"] * iw)
                    if wx < col < wx + ww - 1:
                        L.lit[ty + 1:ty + h - 1, col] ^= True
            # volume (edited with left / right when the cursor is on this row) and level
            vt = f"{round(lo['vols'][i] * 100)}%"
            if cursor == i and self.browser is None:
                L.bar(209, ty - 1, 46, 9)
                L.rtext(253, ty + 1, vt, inv=True)
            else:
                L.rtext(253, ty + 1, vt)
            fr = clamp((20 * math.log10(max(tr["peak"], 1e-5)) + 48) / 48, 0, 1)
            L.frame(210, ty + 10, 44, 5)
            if fr > 0:
                L.fill(212, ty + 12, int(40 * fr), 1)
        self.draw_sections(124, row_h=10, title_h=10, gap=2, offset=LOOP_TRACKS)
        L.hline(0, 174, LCD_W, dotted=True)
        hint = ("A REC/DUB  X STOP  B START/STOP  Y+A UNDO" if self.last_input == "pad"
                else "Z REC SPACE STOP X START/STOP SHIFT+Z UNDO")
        L.ctext(128, 179, hint)

    def draw_lfo(self):
        L = self.lcd
        lf = self.state.lfo
        y = 17
        gw, gh = 252, 54
        L.frame(2, y, gw, gh)
        wave = int(lf["wave"])
        L.text(5, y + 3, LFO_WAVES[wave])
        L.rtext(gw - 24, y + 3, f_rate(lf["rate"]).upper())
        lx, rx = 6, gw - 24
        top, bot = y + 14, y + gh - 5
        if self.lfo_key != wave:          # two cycles, rasterised once per wave shape
            ph = np.linspace(0, 2, rx - lx, endpoint=False) % 1.0
            if wave == 1:
                v = 1 - 4 * np.abs((ph + 0.25) % 1.0 - 0.5)
            elif wave == 2:
                v = 2 * ph - 1
            elif wave == 3:
                v = np.where(ph < 0.5, 1.0, -1.0)
            elif wave == 4:
                steps = np.array([0.3, -0.7, 0.9, -0.1, 0.6, -0.9, 0.1, -0.4])
                v = steps[(np.linspace(0, 2, rx - lx, endpoint=False) * 4).astype(int) % 8]
            else:
                v = np.sin(2 * np.pi * ph)
            ys = top + (1 - v) / 2 * (bot - top)
            before = L.lit.copy()
            L.lit[:] = False
            L.polyline(list(zip((lx + np.arange(len(ys))).tolist(), ys.tolist())))
            self.lfo_px = np.nonzero(L.lit)
            L.lit[:] = before
            self.lfo_key = wave
        L.hline(lx, (top + bot) // 2, rx - lx, dotted=True)
        L.lit[self.lfo_px] = True
        # live value (after the delay fade-in and the mod wheel): a small meter on the right
        mx = gw - 16
        L.frame(mx, top - 2, 14, bot - top + 5)
        if self.engine.voices:
            v = self.engine.lfo_val * self.engine.lfo_scale
            yy = int(round((top + bot) / 2 - v * (bot - top) / 2))
            L.fill(mx + 3, yy - 1, 8, 3)
        targets = [t for t, k in (("PITCH", "pitch"), ("VOLUME", "volume"), ("CUTOFF", "cutoff")) if lf[k] > 0.0005]
        L.text(2, y + gh + 3, "TO " + (" + ".join(targets) if targets else "NOTHING (SET A DEPTH)"))
        if lf["modwheel"]:
            L.rtext(254, y + gh + 3, "WHEEL")
        self.draw_sections(y + gh + 16)

    def draw_amp(self):
        L = self.lcd
        st = self.state
        slot = st.cur_slot()
        p = slot.p
        y = self.content_top()
        L.text(2, y + 1, self.slot_title())
        L.rtext(254, y + 1, "ONE-SHOT" if p["trigger"] else "GATE")
        gh = 76 if st.mode == "chromatic" else 62
        gx, gy, gw = 2, y + 11, 252
        L.frame(gx, gy, gw, gh)
        wa, wd, wr, ws = math.sqrt(p["attack"]), math.sqrt(p["decay"]), math.sqrt(p["release"]), (0.6 if p["trigger"] else 0.9)
        total = wa + wd + ws + wr
        lx, rx = gx + 6, gx + gw - 7
        top, bot = gy + 5, gy + gh - 13
        sx, hh = (rx - lx) / total, bot - top
        pts = [(lx + t * sx, bot - v * hh) for t, v in
               [(0, 0), (wa, 1), (wa + wd, p["sustain"]), (wa + wd + ws, p["sustain"]), (total, 0)]]
        L.hline(lx, bot, rx - lx + 1, dotted=True)
        L.polyline(pts)
        for px, py in pts[1:4]:
            L.fill(int(px) - 1, int(py) - 1, 3, 3)
        for name, t in (("A", wa / 2), ("D", wa + wd / 2), ("S", wa + wd + ws / 2), ("R", wa + wd + ws + wr / 2)):
            L.ctext(int(clamp(lx + t * sx, lx + 3, rx - 3)), gy + gh - 10, name)
        self.draw_sections(gy + gh + 5)

    def draw_fx(self):
        L = self.lcd
        fx = fx_effective(self.state.fx)
        y = 17
        gw, gh = 196, 54
        L.frame(2, y, gw, gh)
        ft = int(fx["ftype"])
        L.text(5, y + 3, ["FILTER OFF", "LOW-PASS", "HIGH-PASS", "BAND-PASS"][ft])
        if ft:
            L.rtext(2 + gw - 4, y + 3, f_hz(fx["cutoff"]).upper())
        lx, rx = 5, 2 + gw - 4
        top, bot = y + 12, y + gh - 4
        key = (ft, fx["cutoff"], fx["reso"])
        if key != self.fcurve_key:
            freqs = np.geomspace(20, 20000, rx - lx)
            if ft:
                b, a = filter_coeffs(ft, fx["cutoff"], fx["reso"])
                z = np.exp(-1j * 2 * np.pi * freqs / SR)
                H = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
                db = np.clip(20 * np.log10(np.maximum(np.abs(H), 1e-6)), -36, 18)
            else:
                db = np.zeros_like(freqs)
            self.fcurve_key = key
            pts = list(zip((lx + np.arange(len(freqs))).tolist(), (top + (18 - db) / 54 * (bot - top)).tolist()))
            before = L.lit.copy()
            L.lit[:] = False
            L.polyline(pts)                         # rasterise once, keep the dot positions
            self.fcurve_px = np.nonzero(L.lit)
            L.lit[:] = before
        zero = int(top + 18 / 54 * (bot - top))
        L.hline(lx, zero, rx - lx, dotted=True)
        for fm in (100, 1000, 10000):
            xm = lx + int(math.log(fm / 20) / math.log(1000) * (rx - lx))
            L.vline(xm, bot - 2, 3)
        L.lit[self.fcurve_px] = True
        # output meters
        mx = 2 + gw + 4
        mw = 254 - mx
        L.frame(mx, y, mw, gh)
        L.ctext(mx + mw // 2, y + 3, "OUT")
        for i, pk in enumerate(self.engine.peak):
            db_ = 20 * math.log10(max(pk, 1e-5))
            fr = clamp((db_ + 48) / 48, 0, 1)
            bx, bw, btop, bbot = mx + 8 + i * 22, 16, y + 13, y + gh - 4
            L.frame(bx, btop, bw, bbot - btop)
            hh = int((bbot - btop - 4) * fr)
            if hh:
                L.fill(bx + 2, bbot - 2 - hh, bw - 4, hh)
        self.draw_sections(y + gh + 5, row_h=11, title_h=11, gap=2)

    def draw_rec(self):
        L = self.lcd
        rec = self.recorder
        y = self.content_top()
        target = "CHROMATIC" if self.state.mode == "chromatic" else f"PAD {self.state.sel_pad + 1}"
        bh = 82 if self.state.mode == "chromatic" else 68
        L.frame(2, y, 252, bh)
        if rec.recording:
            L.fill(7, y + 5, 12, 12)
            L.text(25, y + 4, f"REC {rec.frames / SR:4.1f}S", scale=2)
        else:
            tap_wait = rec.dev == "tap" and not self.engine.loop["cap_open"]
            status = ("CONNECTING..." if rec.starting and rec.usb else "STARTING MIC..." if rec.starting
                      else (self.engine.capture_error() or "CONNECTING...") if tap_wait
                      else "READY" if rec.dev is not None
                      else "READY - SOUND PAUSES WHILE RECORDING" if rec.half_duplex
                      else (rec.error or "INPUT UNAVAILABLE") if rec.error else "INPUT CLOSED")
            L.frame(7, y + 5, 12, 12)
            L.text(25, y + 8, L.fit(status, 225))
        src = self.source_label(rec.source)
        if rec.usb:
            src += " IN " + REC_CHANNELS[rec.channel]
        L.text(25, y + 20, L.fit("FROM " + src, 150))
        L.rtext(251, y + 20, f"INTO {target}")
        # input level
        db_ = 20 * math.log10(max(rec.level, 1e-5))
        fr = clamp((db_ + 60) / 60, 0, 1)
        L.frame(6, y + 29, 244, 7)
        L.fill(8, y + 31, int(240 * fr), 3)
        # current or last take
        peaks = rec.peaks if rec.recording else self.last_rec_peaks
        wy, wh, ww = y + 40, bh - 44, 244
        mid = wy + wh // 2
        L.hline(6, mid, ww, dotted=True)
        if peaks:
            arr = np.array(peaks)
            total = max(len(arr), int(4 * SR / 512)) if rec.recording else len(arr)
            pad = np.zeros(max(total, ww))
            pad[:len(arr)] = arr
            vals = np.maximum.reduceat(pad, np.linspace(0, len(pad), ww + 1).astype(int)[:-1])
            if not rec.recording:
                vals = vals / max(1e-4, float(arr.max()))
            a = np.rint(np.minimum(vals, 1.0) * (wh // 2 - 1)).astype(int)
            L.columns(6, mid - a, mid + a)
        self.draw_sections(y + bh + 5)
        nx = 131
        for i, line in enumerate(["TAKES ARE SAVED TO", "SAMPLES/RECORDINGS", "", "USB INTERFACES, MICS",
                                  "AND SYNTHS SHOW UP", "AS SOURCES"]):
            L.text(nx, y + bh + 17 + i * 11, line)

    def draw_settings(self):
        L = self.lcd
        y = 17
        self.draw_sections(y + 2)
        # engine status, right column under PROJECT
        x, w, ey = 131, 123, y + 76
        L.text(x, ey, "ENGINE")
        L.hline(x + L.width("ENGINE") + 2, ey + 6, w - L.width("ENGINE") - 2, dotted=True)
        midi = ", ".join(self.midi.devices) if self.midi.devices else "NONE"
        rows = [("OUTPUT", f"{SR / 1000:g} KHZ"),
                ("DSP", "NATIVE" if self.engine.native else "NUMPY"),
                ("CPU", f"{round(self.engine.load * 100)}%"),
                ("INPUT", self.input_status()),
                ("MIDI", midi),
                ("BUFFER", f"{self.audio.buffer_ms:.1f} MS" if hasattr(self.audio, "buffer_ms") else "APLAY")]
        for i, (k, v) in enumerate(rows):
            yy = ey + 12 + i * 13
            lw = L.text(x + 1, yy, k)
            L.rtext(x + w - 2, yy, L.fit(v, w - lw - 10))

    def draw_browser(self):
        L = self.lcd
        b = self.browser
        projects = b["kind"] == "projects"
        L.text(2, 18, "PROJECTS" if projects else L.fit(f"LOAD  {self.slot_title()}", 180))
        L.rtext(254, 18, f"{b['idx'] + 1}/{len(b['files'])}")
        L.hline(0, 28, LCD_W, dotted=True)
        row_h, top_y = 13, 33
        rows = (LCD_H - top_y - 14) // row_h
        top = clamp(b["idx"] - rows // 2, 0, max(0, len(b["files"]) - rows))
        for i in range(top, min(len(b["files"]), top + rows)):
            y = top_y + (i - top) * row_h
            if projects:
                name = b["files"][i] + ("  (OPEN)" if b["files"][i] == self.ui["project"] else "")
            else:
                name = os.path.relpath(b["files"][i], SAMPLES_DIR).replace(os.sep, "/")
            name = L.fit(name, 246)
            if i == b["idx"]:
                L.bar(1, y - 2, 254, 11)
                L.text(4, y, name, inv=True)
            else:
                L.text(4, y, name)
        L.hline(0, 179, LCD_W, dotted=True)
        if self.last_input == "pad":
            hint = "A LOAD  Y RENAME/DELETE  B BACK" if projects else "A LOAD  X PLAY  Y RENAME/DELETE  B BACK"
        else:
            hint = "Z LOAD  SHIFT MENU  X BACK" if projects else "Z LOAD  SPACE PLAY  SHIFT MENU  X BACK"
        L.ctext(128, 183, hint)

    def draw(self):
        t0 = time.perf_counter()
        self._draw()
        DRAW_MS[0] = (time.perf_counter() - t0) * 1000

    def _draw(self):
        self.lcd.clear()
        self.draw_header()
        if self.browser is not None:
            self.draw_browser()
        else:
            dict(LOOP=self.draw_loop, SAMPLE=self.draw_sample, EDIT=self.draw_edit, AMP=self.draw_amp, LFO=self.draw_lfo, FX=self.draw_fx,
                 REC=self.draw_rec, SETTINGS=self.draw_settings)[PAGES[self.page]]()
        if self.dialog is not None:
            self.draw_dialog()
        self.draw_overlays()
        rects = self.lcd.compose(self.surf)
        if not rects:
            return                                  # nothing changed on the display
        if self.surf is not self.window:
            pygame.transform.smoothscale(self.surf, self.window.get_size(), self.window)
            pygame.display.flip()
        else:
            pygame.display.update(rects)

    # ---------- main loop
    def run(self):
        clock = pygame.time.Clock()
        # A full redraw takes ~25 ms on the Brick, so animation (playheads, meters) runs at 12 fps there;
        # a button press always redraws immediately.
        fps = self.args.fps or (12 if ON_DEVICE else 60)
        last_draw = 0.0
        while self.running:
            n_ev = len(self.held)
            self.handle_events()
            self.repeat_buttons()
            if hasattr(self.engine, "sync"):
                self.update_comp()
                self.engine.sync()
                self.watch_input()
            lp = self.engine.last_pad
            if lp is not None:
                self.engine.last_pad = None
                if self.state.mode == "kit" and PAGES[self.page] in ("SAMPLE", "EDIT", "AMP") and self.browser is None:
                    self.state.sel_pad = lp
            # Redraw only when something changed or is moving; the screen otherwise idles at 4 fps.
            now = time.time()
            if self.recorder.start_failed:
                self.recorder.start_failed = False
                self.show_toast("Mic: " + (self.recorder.error or "unavailable"), 4)
            looping = self.engine.looper and (any(t["state"] in (1, 2, 3) or t["peak"] > 1e-4
                                                  for t in self.engine.loop["tracks"])
                                              or (PAGES[self.page] == "LOOP" and self.engine.loop["cap_open"]))
            moving = (looping or self.engine.voices or self.recorder.recording or self.recorder.starting or now < self.toast[1]
                      or (self.dialog is not None and self.dialog["kind"] == "keyboard")
                      or now - self.engine.midi_activity < 0.3 or ("select" in self.held)
                      or (PAGES[self.page] == "FX" and max(self.engine.peak) > 1e-3))
            if self.dirty or n_ev != len(self.held) or (moving and now - last_draw >= 1 / fps) or now - last_draw >= 0.25:
                self.draw()
                self.dirty = False
                last_draw = now
            if not STARTED.is_set():
                STARTED.set()
                faulthandler.cancel_dump_traceback_later()
                print("startup: first frame drawn", flush=True)
            if self.engine.learned is not None:
                num, key = self.engine.learned
                self.engine.learned = None
                self.save_settings()
                self.show_toast(f"CC{num} -> {cc_label(key)}", 3)
            if self.job is not None and self.job["done"]:
                self.finish_job()
            if self.job is not None and now - last_draw >= 0.25:
                self.dirty = True
            if self.engine.overloads != self.shown_overloads:
                self.shown_overloads = self.engine.overloads
                self.show_toast("CPU overload: notes dropped. Try audio_buffer 2048", 3)
            clock.tick(60)
        self.shutdown()

    def shutdown(self):
        self.midi.running = False
        self.recorder.recording = False
        self.save_project()
        if self.loop_writer is not None:
            self.show_toast("Saving loops...", 10)
            self.draw()
            self.loop_writer.join(timeout=60)
        try:
            pygame.display.quit()
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(0)   # skip audio teardown (see set_page); the OS releases the devices


# Fallback for pads SDL doesn't recognise as game controllers. Run with --debug-input to see indexes.
JOY_BUTTONS = {0: "a", 1: "b", 2: "x", 3: "y", 4: "l1", 5: "r1", 6: "select", 7: "start", 9: "l2", 10: "r2"}
# The Brick's volume rocker reaches SDL as joystick buttons 0 and 1: its keys (KEY_VOLUMEDOWN 114,
# KEY_VOLUMEUP 115) have lower codes than the game buttons, so SDL numbers them first (seen in log.txt).
VOLUME_BUTTONS = {0: "vol_down", 1: "vol_up"}


STARTED = threading.Event()
DRAW_MS = [0.0]     # duration of the last screen redraw, for the audio diagnostics


AUDIO_BACKEND = "auto"      # auto: aplay on the Brick, SDL elsewhere
ALSA_DEVICE = "default"
ALSA_CAPTURE = "default"
RAW_JOYSTICK = None         # None: raw on the Brick, SDL game controller elsewhere


def load_config():
    global SWAP_AB, DEVICE_CHUNK, JOY_BUTTONS, AUDIO_BACKEND, ALSA_DEVICE, ALSA_CAPTURE, RAW_JOYSTICK
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        SWAP_AB = bool(cfg.get("swap_ab", SWAP_AB))
        DEVICE_CHUNK = int(clamp(int(cfg.get("audio_buffer", DEVICE_CHUNK)), 256, 4096))
        global NATIVE_DSP
        NATIVE_DSP = bool(cfg.get("native_dsp", NATIVE_DSP))
        AUDIO_BACKEND = cfg.get("audio_backend", AUDIO_BACKEND)
        ALSA_DEVICE = cfg.get("alsa_device", ALSA_DEVICE)
        ALSA_CAPTURE = cfg.get("alsa_capture", ALSA_CAPTURE)
        if "raw_joystick" in cfg:
            RAW_JOYSTICK = bool(cfg["raw_joystick"])
        jb = cfg.get("joy_buttons")
        if isinstance(jb, dict):
            JOY_BUTTONS = {int(k): v for k, v in jb.items()}
        for k, v in VOLUME_BUTTONS.items():
            JOY_BUTTONS.setdefault(k, v)
        print("config:", cfg)
    except Exception as e:
        print("controls.json ignored:", e)


def main():
    print(f"BrickLooper starting: python {sys.version.split()[0]}, pygame {pygame.version.ver}, "
          f"SDL {'.'.join(map(str, pygame.get_sdl_version()))}, numpy {np.__version__}, device={ON_DEVICE}", flush=True)
    load_config()
    sys.setswitchinterval(0.001)   # let the audio callback get the GIL quickly while the UI draws
    ap = argparse.ArgumentParser(description="BrickLooper")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale on desktop, e.g. 0.8")
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--debug-input", action="store_true", help="show button numbers as you press them")
    args = ap.parse_args()
    # Hang diagnostics: if no frame is drawn within 30 s, write every thread's stack to the log
    # and quit, so the console never stays stuck on a black screen. Runs in C, so it works even
    # when a native call is holding Python's lock.
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, exit=True)
    App(args).run()


if __name__ == "__main__":
    main()
