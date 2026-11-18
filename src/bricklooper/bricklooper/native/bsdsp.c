/*
 * bsdsp - native audio engine for BrickLooper (5-track looper, USB / mic input, FX, metronome, ALSA out).
 * Build: see build.sh (zig cc cross-compiles for the Brick, Windows and Linux).
 *
 * BrickLooper - Copyright (C) 2026 Alessandro Pietrosanti / BrickWorld (brickworldhub.com)
 * This program is free software: you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
 * LICENSE file or <https://www.gnu.org/licenses/>.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#define DSIZE (1 << 18)            /* delay line: ~5.4 s at 48 kHz */
#define DMASK (DSIZE - 1)
#define NCOMB 8
#define NAP 4

enum { P_DRIVE, P_BITS, P_DS, P_FTYPE, P_CUTOFF, P_RESO, P_DTIME, P_DFB, P_DTONE, P_DPING, P_DMIX,
       P_VSIZE, P_VDAMP, P_VMIX, P_MASTER, P_DRVON, P_FLTON, P_DLYON, P_REVON, P_COUNT };
#define FXMAXN 1024

typedef struct {
    float *buf;
    int len, pos;
    float store;
} Comb;

typedef struct {
    float *buf;
    int len, pos;
} Allpass;

typedef struct {
    float sr;
    /* downsample hold */
    int ds_count;
    float ds_hold[2];
    /* biquad (transposed direct form II) */
    int f_on, f_type;
    float f_cut, f_reso;
    float b0, b1, b2, a1, a2;
    float z1[2], z2[2];
    /* delay */
    float *dbuf[2];
    int dw;
    float dcur;                     /* current delay in samples, glides toward the target */
    float tone[2];
    /* reverb */
    Comb comb[2][NCOMB];
    Allpass ap[2][NAP];
    /* switching without clicks (as BrickFX 0.3.1 / 0.3.2) */
    float g_drv, g_flt;             /* drive and filter sections: 0 = dry, 1 = full effect */
    int flt_cur;                    /* filter type running now (0 = none) */
    float gin_d, gin_v;             /* delay / reverb input gates: switching off only closes the input */
    int d_run, v_run;               /* still running (on, or its tail ringing out) */
    float d_quiet, v_quiet, d_off, v_off;
    float dmix_c, vmix_c;           /* smoothed mix levels */
    float wL[FXMAXN], wR[FXMAXN];
} FX;

static const int COMB_44[NCOMB] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
static const int AP_44[NAP] = {556, 441, 341, 277};
#define SPREAD 23

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

API FX *fx_create(int sr) {
    FX *f = (FX *)calloc(1, sizeof(FX));
    if (!f) return NULL;
    f->sr = (float)sr;
    float k = sr / 44100.0f;
    for (int c = 0; c < 2; c++) {
        f->dbuf[c] = (float *)calloc(DSIZE, sizeof(float));
        for (int i = 0; i < NCOMB; i++) {
            int len = (int)((COMB_44[i] + (c ? SPREAD : 0)) * k);
            f->comb[c][i].len = len;
            f->comb[c][i].buf = (float *)calloc(len, sizeof(float));
        }
        for (int i = 0; i < NAP; i++) {
            int len = (int)((AP_44[i] + (c ? SPREAD : 0)) * k);
            f->ap[c][i].len = len;
            f->ap[c][i].buf = (float *)calloc(len, sizeof(float));
        }
    }
    f->dcur = -1;
    f->f_type = -1;
    return f;
}

API void fx_destroy(FX *f) {
    if (!f) return;
    for (int c = 0; c < 2; c++) {
        free(f->dbuf[c]);
        for (int i = 0; i < NCOMB; i++) free(f->comb[c][i].buf);
        for (int i = 0; i < NAP; i++) free(f->ap[c][i].buf);
    }
    free(f);
}

API void fx_clear(FX *f) {
    for (int c = 0; c < 2; c++) {
        memset(f->dbuf[c], 0, DSIZE * sizeof(float));
        for (int i = 0; i < NCOMB; i++) { memset(f->comb[c][i].buf, 0, f->comb[c][i].len * sizeof(float)); f->comb[c][i].store = 0; }
        for (int i = 0; i < NAP; i++) memset(f->ap[c][i].buf, 0, f->ap[c][i].len * sizeof(float));
        f->z1[c] = f->z2[c] = 0;
        f->tone[c] = 0;
    }
    f->d_run = f->v_run = 0; f->dcur = -1; f->d_quiet = f->v_quiet = 0;
}

/* RBJ biquad, same formulas as filter_coeffs() in bricksampler.py (incl. resonance gain compensation). */
static void set_filter(FX *f, int type, float fc, float reso) {
    fc = clampf(fc, 20.0f, f->sr * 0.45f);
    float q = 0.707f * powf(2.0f, reso * 4.5f);
    float w0 = 2.0f * 3.14159265f * fc / f->sr;
    float c = cosf(w0), s = sinf(w0), al = s / (2 * q), g = sqrtf(0.707f / q);
    float b0, b1, b2;
    if (type == 1) { b0 = (1 - c) / 2 * g; b1 = (1 - c) * g; b2 = b0; }
    else if (type == 2) { b0 = (1 + c) / 2 * g; b1 = -(1 + c) * g; b2 = b0; }
    else { b0 = al; b1 = 0; b2 = -al; }
    float a0 = 1 + al;
    f->b0 = b0 / a0; f->b1 = b1 / a0; f->b2 = b2 / a0;
    f->a1 = -2 * c / a0; f->a2 = (1 - al) / a0;
    f->f_type = type; f->f_cut = fc; f->f_reso = reso;
}

static float scurve(float g) { return g * g * (3.0f - 2.0f * g); }

/* Moves g toward target in `ms` milliseconds; returns the value at the end of this block. */
static float ramp(float g, float target, int n, float sr, float ms) {
    float step = (float)n / (ms * 0.001f * sr);
    return target > g ? fminf(target, g + step) : fmaxf(target, g - step);
}

static void drive_block(FX *f, float *L, float *R, int n, const float *p) {
    float *ch[2] = {L, R};
    float dr = p[P_DRIVE];
    if (dr > 0.001f) {
        float g = 1 + 30 * dr * dr, comp = 0.4f + 0.6f / g;
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < n; i++) ch[c][i] = tanhf(ch[c][i] * g) * comp;
    }
    int bits = (int)p[P_BITS];
    if (bits < 16 && bits >= 1) {
        float q = (float)(1 << (bits - 1));
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < n; i++) ch[c][i] = roundf(ch[c][i] * q) / q;
    }
    int k = (int)p[P_DS];
    if (k > 1) {
        int cnt = f->ds_count;
        for (int i = 0; i < n; i++) {
            if (cnt == 0) { f->ds_hold[0] = L[i]; f->ds_hold[1] = R[i]; }
            L[i] = f->ds_hold[0]; R[i] = f->ds_hold[1];
            if (++cnt >= k) cnt = 0;
        }
        f->ds_count = cnt;
    }
}

/* out = dry + (wet - dry) * S(g), g going from g0 to g1 over the block */
static void blend(float *L, float *R, const float *wL, const float *wR, int n, float g0, float g1) {
    for (int i = 0; i < n; i++) {
        float s = scurve(g0 + (g1 - g0) * (float)(i + 1) / (float)n);
        L[i] += (wL[i] - L[i]) * s;
        R[i] += (wR[i] - R[i]) * s;
    }
}

static void fx_chunk(FX *f, float *L, float *R, int n, const float *p);

/* Process one stereo block in place. p holds P_COUNT parameters (see enum). */
API void fx_process(FX *f, float *L, float *R, int n, const float *p) {
    for (int o = 0; o < n; o += FXMAXN) {
        int m = n - o < FXMAXN ? n - o : FXMAXN;
        fx_chunk(f, L + o, R + o, m, p);
    }
}

static void fx_chunk(FX *f, float *L, float *R, int n, const float *p) {
    float *ch[2] = {L, R};
    float sr = f->sr;
    /* DRIVE (drive, bitcrush, downsample): fades in / out over 10 ms */
    {
        float g0 = f->g_drv, g1 = ramp(g0, p[P_DRVON] > 0.5f ? 1.0f : 0.0f, n, sr, 10);
        if (g0 > 0 || g1 > 0) {
            memcpy(f->wL, L, sizeof(float) * n); memcpy(f->wR, R, sizeof(float) * n);
            drive_block(f, f->wL, f->wR, n, p);
            blend(L, R, f->wL, f->wR, n, g0, g1);
        }
        f->g_drv = g1;
    }
    /* FILTER: a type change fades out, changes over while silent, fades in (10 ms) */
    {
        int want = p[P_FLTON] > 0.5f ? (int)p[P_FTYPE] : 0;
        float target = (want != 0 && want == f->flt_cur) ? 1.0f : 0.0f;
        float g0 = f->g_flt, g1 = ramp(g0, target, n, sr, 20);
        if (f->flt_cur != 0 && (g0 > 0 || g1 > 0)) {
            if (f->flt_cur != f->f_type || fabsf(p[P_CUTOFF] - f->f_cut) > 0.05f || fabsf(p[P_RESO] - f->f_reso) > 0.0005f)
                set_filter(f, f->flt_cur, p[P_CUTOFF], p[P_RESO]);
            memcpy(f->wL, L, sizeof(float) * n); memcpy(f->wR, R, sizeof(float) * n);
            float *w[2] = {f->wL, f->wR};
            for (int c = 0; c < 2; c++) {
                float z1 = f->z1[c], z2 = f->z2[c], *x = w[c];
                for (int i = 0; i < n; i++) {
                    float in = x[i], out = f->b0 * in + z1;
                    z1 = f->b1 * in - f->a1 * out + z2;
                    z2 = f->b2 * in - f->a2 * out;
                    x[i] = out;
                }
                f->z1[c] = z1; f->z2[c] = z2;
            }
            blend(L, R, f->wL, f->wR, n, g0, g1);
        }
        f->g_flt = g1;
        if (g1 <= 0.0f && f->flt_cur != want) {          /* silent: change over */
            f->flt_cur = want;
            f->z1[0] = f->z1[1] = f->z2[0] = f->z2[1] = 0;
            f->f_type = -1;
        }
    }
    /* DELAY: off only closes the input (30 ms); the echoes ring out on their own */
    {
        int on = p[P_DLYON] > 0.5f;
        float gi0 = f->gin_d, gi1 = ramp(gi0, on ? 1.0f : 0.0f, n, sr, 30);
        if (on) { f->d_run = 1; f->d_quiet = 0; f->d_off = 0; }
        float m0 = f->dmix_c, m1 = ramp(m0, p[P_DMIX], n, sr, 30);
        f->dmix_c = m1;
        if (f->d_run) {
            float target = clampf(p[P_DTIME] * sr, 16.0f, (float)(DSIZE - 4));
            if (f->dcur < 0) f->dcur = target;
            float c = 0.85f * (1 - p[P_DTONE]), fb = fminf(p[P_DFB], 0.95f);
            int ping = p[P_DPING] > 0.5f;
            float *b0 = f->dbuf[0], *b1 = f->dbuf[1];
            int w = f->dw;
            float pk = 0, tailg = 1.0f;
            if (!on) {
                f->d_off += (float)n / sr;
                if (f->d_off > 20.0f) tailg = fmaxf(0.0f, 1.0f - (f->d_off - 20.0f));   /* never-ending tail */
            }
            for (int i = 0; i < n; i++) {
                float fr_ = (float)(i + 1) / (float)n;
                float gin = scurve(gi0 + (gi1 - gi0) * fr_), mix = (m0 + (m1 - m0) * fr_) * tailg;
                f->dcur += (target - f->dcur) * 0.0004f;
                float rp = (float)w - f->dcur;
                int r0 = (int)floorf(rp);
                float fr = rp - r0;
                int ia = r0 & DMASK, ib = (r0 + 1) & DMASK;
                float rl = b0[ia] + (b0[ib] - b0[ia]) * fr;
                float rr = b1[ia] + (b1[ib] - b1[ia]) * fr;
                f->tone[0] = c * f->tone[0] + (1 - c) * rl;
                f->tone[1] = c * f->tone[1] + (1 - c) * rr;
                float wl = f->tone[0], wr = f->tone[1];
                float il = L[i] * gin, ir = R[i] * gin;
                if (ping) { b0[w] = 0.5f * (il + ir) + fb * wr; b1[w] = fb * wl; }
                else { b0[w] = il + fb * wl; b1[w] = ir + fb * wr; }
                L[i] += mix * wl; R[i] += mix * wr;
                float a = fabsf(wl) > fabsf(wr) ? fabsf(wl) : fabsf(wr);
                if (a > pk) pk = a;
                w = (w + 1) & DMASK;
            }
            f->dw = w;
            if (!on && gi1 <= 0.0f) {
                f->d_quiet = (pk < 3.2e-5f || tailg <= 0.0f) ? f->d_quiet + (float)n / sr : 0.0f;
                if (f->d_quiet > p[P_DTIME] + 0.3f || tailg <= 0.0f) {
                    f->d_run = 0; f->dcur = -1;
                    memset(f->dbuf[0], 0, DSIZE * sizeof(float)); memset(f->dbuf[1], 0, DSIZE * sizeof(float));
                    f->tone[0] = f->tone[1] = 0;
                }
            }
        }
        f->gin_d = gi1;
    }
    /* REVERB: same, input closes over 80 ms, the tail rings out */
    {
        int on = p[P_REVON] > 0.5f;
        float gi0 = f->gin_v, gi1 = ramp(gi0, on ? 1.0f : 0.0f, n, sr, 80);
        if (on) { f->v_run = 1; f->v_quiet = 0; f->v_off = 0; }
        float m0 = f->vmix_c, m1 = ramp(m0, p[P_VMIX], n, sr, 30);
        f->vmix_c = m1;
        if (f->v_run) {
            float fb = p[P_VSIZE] * 0.28f + 0.7f, d1 = p[P_VDAMP] * 0.4f, d0 = 1 - d1;
            float pk = 0, tailg = 1.0f;
            if (!on) {
                f->v_off += (float)n / sr;
                if (f->v_off > 20.0f) tailg = fmaxf(0.0f, 1.0f - (f->v_off - 20.0f));
            }
            for (int i = 0; i < n; i++) {
                float fr_ = (float)(i + 1) / (float)n;
                float gin = scurve(gi0 + (gi1 - gi0) * fr_), wet = (m0 + (m1 - m0) * fr_) * 3.0f * tailg;
                float in = (L[i] + R[i]) * 0.015f * gin;
                for (int c = 0; c < 2; c++) {
                    float out = 0;
                    for (int j = 0; j < NCOMB; j++) {
                        Comb *cb = &f->comb[c][j];
                        float y = cb->buf[cb->pos];
                        out += y;
                        cb->store = d0 * y + d1 * cb->store;
                        cb->buf[cb->pos] = in + fb * cb->store;
                        if (++cb->pos >= cb->len) cb->pos = 0;
                    }
                    for (int j = 0; j < NAP; j++) {
                        Allpass *a = &f->ap[c][j];
                        float bo = a->buf[a->pos];
                        a->buf[a->pos] = out + bo * 0.5f;
                        out = bo - out;
                        if (++a->pos >= a->len) a->pos = 0;
                    }
                    ch[c][i] += wet * out;
                    if (fabsf(out) > pk) pk = fabsf(out);
                }
            }
            if (!on && gi1 <= 0.0f) {
                f->v_quiet = (pk * 3.0f < 3.2e-5f || tailg <= 0.0f) ? f->v_quiet + (float)n / sr : 0.0f;
                if (f->v_quiet > 0.4f || tailg <= 0.0f) {
                    f->v_run = 0;
                    for (int c = 0; c < 2; c++) {
                        for (int j = 0; j < NCOMB; j++) { memset(f->comb[c][j].buf, 0, f->comb[c][j].len * sizeof(float)); f->comb[c][j].store = 0; }
                        for (int j = 0; j < NAP; j++) memset(f->ap[c][j].buf, 0, f->ap[c][j].len * sizeof(float));
                    }
                }
            }
        }
        f->gin_v = gi1;
    }
    /* master + soft clip above 0.8 */
    float m = p[P_MASTER];
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < n; i++) {
            float x = ch[c][i] * m, ax = fabsf(x);
            if (ax > 0.8f) x = copysignf(0.8f + 0.2f * tanhf((ax - 0.8f) / 0.2f), x);
            ch[c][i] = x;
        }
}

/*
 * Render one voice into L/R (added). Linear interpolation, same indexing as Voice.render():
 * region [s, e), pos counts from 0 inside the region, inc = playback rate.
 * Envelope ramps linearly from e0 to e1 over the block (sample k gets e0 + (e1-e0)*(k+1)/n).
 * Returns 1 when a non-looping voice has reached the end of its region.
 */
API int voice_render(const float *d, int s, int e, double *pos, double inc, int loop, int reverse,
                     float e0, float e1, float amp, float gl, float gr, float *L, float *R, int n) {
    int len = e - s;
    if (len < 2) return 1;
    double ps = *pos;
    float de = (e1 - e0) / n;
    int done = 0;
    for (int k = 0; k < n; k++) {
        double x = ps + inc * k;
        int i0, i1;
        if (loop) {
            x = fmod(x, (double)len);
            if (x < 0) x += len;
            i0 = (int)x;
            i1 = i0 + 1;
            if (i1 >= len) i1 = 0;
        } else {
            if (x >= len - 1) { done = 1; break; }
            i0 = (int)x;
            i1 = i0 + 1;
        }
        float fr = (float)(x - floor(x));
        int a = reverse ? e - 1 - i0 : s + i0;
        int b = reverse ? e - 1 - i1 : s + i1;
        float v = (d[a] + (d[b] - d[a]) * fr) * (e0 + de * (k + 1)) * amp;
        L[k] += v * gl;
        R[k] += v * gr;
    }
    ps += inc * n;
    if (loop) { ps = fmod(ps, (double)len); }
    *pos = ps;
    return done;
}

API int bsdsp_version(void) { return 5; }


/* ===========================================================================================
 * Engine (version 2): voices, envelopes, FX and - on Linux - the ALSA writer thread, all in C.
 * The audio thread never touches Python, so drawing the screen can't delay a block.
 * Python sends notes and parameters; it reads back a small status array.
 * =========================================================================================== */
#define E_SLOTS 17
#define E_MAXV 12
#define E_QLEN 512

enum { SP_ROOT, SP_TUNE, SP_FINE, SP_GAIN, SP_PAN, SP_START, SP_END, SP_LOOP, SP_REVERSE, SP_TRIGGER,
       SP_ATTACK, SP_DECAY, SP_SUSTAIN, SP_RELEASE, SP_VELSENS, SP_LSTART, SP_LEND, SP_XFADE, SP_COUNT };
enum { LP_WAVE, LP_RATE, LP_DELAY, LP_PITCH, LP_VOLUME, LP_CUTOFF, LP_MODWHEEL, LP_COUNT };
enum { EV_ON = 1, EV_OFF, EV_CC, EV_BEND, EV_ALLOFF, EV_LOOP };

typedef struct { const float *d; int n; float p[SP_COUNT]; } ESlot;
typedef struct {
    int slot, note, keytrack, s, e, reverse, oneshot, loop, stage, done, sustained;
    double pos;
    int ls, le;                     /* loop in playback coordinates (0 = region start in play direction) */
    double xf;                      /* crossfade length in samples */
    float amp, gl, gr, level, rel_rate;
} EVoice;
typedef struct { int kind, a, b; } EEv;

#if !defined(_WIN32)
#include <pthread.h>
#include <dlfcn.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
typedef pthread_mutex_t elock_t;
#define LOCK_INIT(m) pthread_mutex_init(&(m), NULL)
#define LOCK(m) pthread_mutex_lock(&(m))
#define UNLOCK(m) pthread_mutex_unlock(&(m))
#else
#include <windows.h>
typedef CRITICAL_SECTION elock_t;
#define LOCK_INIT(m) InitializeCriticalSection(&(m))
#define LOCK(m) EnterCriticalSection(&(m))
#define UNLOCK(m) LeaveCriticalSection(&(m))
#endif

/* ---- looper (version 4): 5 stereo tracks, synced to the first loop's length.
 * Time is counted in output frames (G). "Index time" t is where a sound belongs in the loop: the frame
 * the player was hearing when they played it. Sounds made inside the engine (pads, notes) belong at
 * G - comp_int (the output buffer delay); sounds from the USB input at G - comp_ext (output + input delay). */
#define NTRK 5
#define T_EMPTY 0
#define T_REC 1
#define T_PLAY 2
#define T_DUB 3
#define T_STOP 4
#define TFOREVER ((long long)1 << 62)
#define RING (1 << 15)               /* input ring, stereo frames */
#define TAPN (1 << 16)               /* mono tap for the sampler's REC page */
enum { LC_RECDUB = 1, LC_STOP, LC_UNDO, LC_CLEAR, LC_ALL, LC_CLEARALL };
enum { LP2_COMPI, LP2_COMPE, LP2_SRC, LP2_MON, LP2_INGAIN, LP2_OUTVOL, LP2_RECDUB, LP2_V1,
       LP2_MET = LP2_V1 + NTRK, LP2_BPM, LP2_BEATS, LP2_CLICK, LP2_REPLACE, LP2_COUNTIN, LP2_COUNT };

typedef struct {
    float *d, *ly;                  /* loop audio and the last overdub (for undo), interleaved stereo */
    long long len, ph;              /* length; read index = (t - ph) mod len */
    int st;
    int fresh, master;              /* a new loop is being recorded (length not known yet) */
    long long rs, re, base, cap;    /* its window [rs, re) in index time, written at d[t - base] */
    long long zhi;                  /* d[0..zhi) has been zeroed for this take */
    int after;                      /* state when the new loop's window closes */
    int dub;                        /* overdub window open (or its input tail still arriving) */
    int replace;                    /* this overdub replaces what was there instead of adding */
    long long unit;                 /* a new take is rounded to whole units (the loop, or a bar of the click) */
    long long ds, de;
    int undo, ukind;                /* undo: 1 = can undo, 2 = can redo; kind 0 = overdub, 1 = whole take */
    float g;                        /* fade gain (stop/start without clicks) */
    float pk;
    int ver;                        /* bumped whenever the audio changes (the screen redraws the wave) */
} Track;

typedef struct {
    int sr, block, kit, max_voices, kit_base, sustain;
    float bend, bend_range;
    ESlot slot[E_SLOTS];
    EVoice v[E_MAXV];
    int nv;
    float fxp[P_COUNT], fxl[P_COUNT];
    float lfop[LP_COUNT];
    double lfo_phase, lfo_t;
    float lfo_sh, lfo_val, lfo_scale, mw;
    unsigned rng;
    FX *fx;
    elock_t lock, qlock;
    EEv q[E_QLEN];
    int qh, qt;
    float *L, *R;
    short *pcm;
    float peak[2], load;
    int quiet_blocks, idle, underruns, blocks;
    /* looper */
    Track t[NTRK];
    long long G, L0, om, maxf;
    long long mo;                   /* metronome origin before there is a loop */
    float lp[LP2_COUNT];
    float *inL, *inR, *pL, *pR;
    float in_pk[2];
    float *ring;                    /* input frames from the capture thread (or eng_in_push) */
    volatile unsigned rh, rt_;      /* write / read counters */
    int ring_target, in_under, in_drop, in_started;
    double rfrac, fill_f, drift_i, ratio;     /* drift resampler: input frames read per output frame */
    volatile double cap_step;                 /* device rate / engine rate (1 when the device runs at the engine rate) */
    float *tap;
    volatile unsigned th_, tt_;
    volatile int tap_on;
    volatile int cap_period, cap_open, cap_mode;
    /* built-in mic: the Brick can't play and record at once, so while a take records from it the
     * speaker is closed and the mic's own clock drives the engine (time keeps running, in silence) */
    volatile int mic_src, mic_want, mic_on, mic_direct;
#if !defined(_WIN32)
    pthread_t th;
    volatile int run, paused, closed, want_frames, opened_frames, buffer_frames, period_frames, rt, open_fails;
    void *pcmh;
    char devs[256], dev[64], err[160];
    pthread_t cth;
    volatile int cap_run, cap_gen;
    char cap_dev[512], cap_err[160], cap_used[96];
    volatile double cap_reads;
    volatile float cap_pk;
    volatile int cap_ch;
    volatile int cap_req_fmt, cap_req_rate;   /* wanted: the device's best format and a rate it has (0 = engine's) */
    volatile int cap_fmt, cap_rate;           /* what is open */
    volatile float cap_lvl[2];                /* biggest raw sample per channel since the last eng_cap_level */
    volatile int cap_nz;                      /* any non-zero sample since the last eng_cap_level */
#endif
} Eng;

API Eng *eng_create(int sr, int block) {
    Eng *g = (Eng *)calloc(1, sizeof(Eng));
    g->sr = sr; g->block = block; g->max_voices = 8; g->kit_base = 36; g->bend_range = 2.0f;
    g->fx = fx_create(sr);
    g->L = (float *)calloc(block, sizeof(float));
    g->R = (float *)calloc(block, sizeof(float));
    g->pcm = (short *)calloc(block * 2, sizeof(short));
    LOCK_INIT(g->lock); LOCK_INIT(g->qlock);
    g->fxp[P_BITS] = 16; g->fxp[P_DS] = 1; g->fxp[P_MASTER] = 1; g->fxp[P_CUTOFF] = 18000;
    g->lfop[LP_RATE] = 5; g->rng = 22222u; g->mw = 0;
    /* looper: 60 s per track. calloc'd memory is only really used once it is written. */
    g->maxf = (long long)sr * 60;
    for (int i = 0; i < NTRK; i++) {
        g->t[i].d = (float *)calloc((size_t)g->maxf * 2, sizeof(float));
        g->t[i].ly = (float *)calloc((size_t)g->maxf * 2, sizeof(float));
        g->lp[LP2_V1 + i] = 1.0f;
    }
    g->inL = (float *)calloc(block, sizeof(float)); g->inR = (float *)calloc(block, sizeof(float));
    g->pL = (float *)calloc(block, sizeof(float)); g->pR = (float *)calloc(block, sizeof(float));
    g->ring = (float *)calloc(RING * 2, sizeof(float));
    g->tap = (float *)calloc(TAPN, sizeof(float));
    g->lp[LP2_INGAIN] = 1; g->lp[LP2_OUTVOL] = 1; g->lp[LP2_MON] = 1; g->lp[LP2_SRC] = 2;
    g->ring_target = 512;
    return g;
}

/* ---- calls from Python (any thread) */
static void push(Eng *g, int kind, int a, int b) {
    LOCK(g->qlock);
    int nt = (g->qt + 1) % E_QLEN;
    if (nt != g->qh) { g->q[g->qt].kind = kind; g->q[g->qt].a = a; g->q[g->qt].b = b; g->qt = nt; }
    UNLOCK(g->qlock);
}
API void eng_note_on(Eng *g, int note, int vel) { push(g, EV_ON, note, vel); }
API void eng_note_off(Eng *g, int note) { push(g, EV_OFF, note, 0); }
API void eng_cc(Eng *g, int num, int val) { push(g, EV_CC, num, val); }
API void eng_bend(Eng *g, int val14) { push(g, EV_BEND, val14, 0); }
API void eng_all_off(Eng *g) { push(g, EV_ALLOFF, 0, 0); }

API void eng_config(Eng *g, int kit, int max_voices, int kit_base, float bend_range) {
    LOCK(g->lock);
    g->kit = kit; g->max_voices = max_voices < E_MAXV - 4 ? max_voices : E_MAXV - 4;
    g->kit_base = kit_base; g->bend_range = bend_range;
    UNLOCK(g->lock);
}

/* New sample data for a slot: voices still reading the old data are dropped first. */
API void eng_set_slot_data(Eng *g, int idx, const float *d, int n) {
    if (idx < 0 || idx >= E_SLOTS) return;
    LOCK(g->lock);
    int k = 0;
    for (int i = 0; i < g->nv; i++) if (g->v[i].slot != idx) g->v[k++] = g->v[i];
    g->nv = k;
    g->slot[idx].d = d; g->slot[idx].n = d ? n : 0;
    UNLOCK(g->lock);
}
API void eng_set_slot_params(Eng *g, int idx, const float *p) {
    if (idx < 0 || idx >= E_SLOTS) return;
    LOCK(g->lock);
    memcpy(g->slot[idx].p, p, sizeof(float) * SP_COUNT);
    UNLOCK(g->lock);
}
API void eng_set_lfo(Eng *g, const float *p) {
    LOCK(g->lock);
    memcpy(g->lfop, p, sizeof(float) * LP_COUNT);
    UNLOCK(g->lock);
}
API void eng_set_fx(Eng *g, const float *p) {
    LOCK(g->lock);
    memcpy(g->fxp, p, sizeof(float) * P_COUNT);
    UNLOCK(g->lock);
}

/* ---- voices (audio thread, under g->lock) */
static void v_release(Eng *g, EVoice *v, int fast) {
    if (v->stage >= 3 && !fast) return;
    float r = fast ? 0.004f : (g->slot[v->slot].p[SP_RELEASE] > 0.002f ? g->slot[v->slot].p[SP_RELEASE] : 0.002f);
    v->stage = 3;
    v->rel_rate = (v->level > 1e-4f ? v->level : 1e-4f) / r;
}

static void v_start(Eng *g, int idx, int note, int vel, int keytrack) {
    ESlot *sl = &g->slot[idx];
    const float *p = sl->p;
    if (g->nv >= E_MAXV) { memmove(g->v, g->v + 1, sizeof(EVoice) * (g->nv - 1)); g->nv--; }
    EVoice *v = &g->v[g->nv++];
    memset(v, 0, sizeof(*v));
    int n = sl->n;
    /* +1e-3: a fraction sent as float32 (0.9 -> 0.8999999762) must not lose a whole sample */
    int s = (int)((double)p[SP_START] * n + 1e-3);
    int e = (int)((double)p[SP_END] * n + 1e-3);
    if (e < s + 2) e = s + 2;
    if (e > n) e = n;
    if (s > e - 2) s = e - 2;
    if (s < 0) s = 0;
    v->slot = idx; v->note = note; v->keytrack = keytrack; v->s = s; v->e = e;
    v->reverse = p[SP_REVERSE] != 0; v->oneshot = p[SP_TRIGGER] != 0;
    v->loop = p[SP_LOOP] != 0 && !v->oneshot;
    int ls = (int)((double)p[SP_LSTART] * n + 1e-3), le = (int)((double)p[SP_LEND] * n + 1e-3);
    if (ls < s) ls = s;
    if (ls > e - 2) ls = e - 2;
    if (le > e) le = e;
    if (le < ls + 2) le = ls + 2;
    if (v->reverse) { v->ls = e - le; v->le = e - ls; } else { v->ls = ls - s; v->le = le - s; }
    double L = v->le - v->ls, xf = p[SP_XFADE] * L;
    double avail = v->reverse ? (double)(n - e) + v->ls : (double)s + v->ls;   /* material before the loop start */
    if (xf > avail) xf = avail;
    if (xf > L) xf = L;
    v->xf = xf < 1 ? 0 : xf;
    float vs = p[SP_VELSENS];
    v->amp = p[SP_GAIN] * ((1 - vs) + vs * powf(vel / 127.0f, 1.5f));
    float ang = (p[SP_PAN] + 1) * 3.14159265f / 4;
    v->gl = cosf(ang) * 1.41421356f; v->gr = sinf(ang) * 1.41421356f;
    v->rel_rate = 1;
}

static void loop_cmd(Eng *g, int cmd, int ti);

static void handle(Eng *g, EEv *ev) {
    int a = ev->a, b = ev->b;
    if (ev->kind == EV_LOOP) { loop_cmd(g, a, b); return; }
    if (ev->kind == EV_ON) {
        int idx, keytrack;
        if (!g->kit) { idx = 0; keytrack = 1; }
        else {
            int pad = a - g->kit_base;
            if (pad < 0 || pad >= E_SLOTS - 1) return;
            idx = 1 + pad; keytrack = 0;
        }
        if (!g->slot[idx].d || g->slot[idx].n < 2) return;
        if (!keytrack)
            for (int i = 0; i < g->nv; i++) if (g->v[i].slot == idx) v_release(g, &g->v[i], 1);
        int live = 0, first = -1;
        for (int i = 0; i < g->nv; i++) if (g->v[i].stage < 3) { if (first < 0) first = i; live++; }
        if (live >= g->max_voices && first >= 0) v_release(g, &g->v[first], 1);
        if (live == 0) {            /* first note after silence restarts the LFO and its delay */
            g->lfo_phase = 0; g->lfo_t = 0;
            g->rng = g->rng * 1664525u + 1013904223u;
            g->lfo_sh = (float)(g->rng >> 8) / 8388608.0f - 1.0f;
        }
        v_start(g, idx, a, b, keytrack);
    } else if (ev->kind == EV_OFF) {
        for (int i = 0; i < g->nv; i++) {
            EVoice *v = &g->v[i];
            if (v->note == a && !v->oneshot && v->stage < 3) {
                if (g->sustain) v->sustained = 1; else v_release(g, v, 0);
            }
        }
    } else if (ev->kind == EV_CC) {
        if (a == 64) {
            g->sustain = b >= 64;
            if (!g->sustain)
                for (int i = 0; i < g->nv; i++)
                    if (g->v[i].sustained) { g->v[i].sustained = 0; v_release(g, &g->v[i], 0); }
        } else if (a == 1) {
            g->mw = b / 127.0f;
        } else if (a == 120 || a == 123) {
            for (int i = 0; i < g->nv; i++) v_release(g, &g->v[i], a == 120);
        }
    } else if (ev->kind == EV_BEND) {
        g->bend = (a - 8192) / 8192.0f * g->bend_range;
    } else if (ev->kind == EV_ALLOFF) {
        for (int i = 0; i < g->nv; i++) v_release(g, &g->v[i], 1);
        g->sustain = 0;
    }
}

/* =========================================================================================== looper */

static long long lmod(long long a, long long m) { long long r = a % m; return r < 0 ? r + m : r; }
static long long t_int(Eng *g) { return g->G - (long long)g->lp[LP2_COMPI]; }
static long long t_ext(Eng *g) { return g->mic_on ? g->G - (long long)g->lp[LP2_COMPI] : g->G - (long long)g->lp[LP2_COMPE]; }

static void check_reset_len(Eng *g) {
    /* With every track empty the next take sets a new loop length. */
    for (int i = 0; i < NTRK; i++) if (g->t[i].st != T_EMPTY || g->t[i].fresh) return;
    g->L0 = 0;
}

static void zero_to(Eng *g, Track *tr, long long hi) {
    if (hi > g->maxf) hi = g->maxf;
    if (hi > tr->zhi) { memset(tr->d + 2 * tr->zhi, 0, sizeof(float) * 2 * (size_t)(hi - tr->zhi)); tr->zhi = hi; }
}

/* A new take on an empty track, starting at index time S. The first take of all sets the loop length.
 * Later takes are lined up with the loop: they start from the loop boundary nearest to S (a little
 * before S is kept in a pre-roll at the top of the buffer and folded in at the end). */
/* One bar of the metronome in frames, when it's on and there is no loop yet (0 otherwise). */
static long long bar_frames(Eng *g) {
    if (g->lp[LP2_MET] < 0.5f || g->lp[LP2_BPM] < 20.0f) return 0;
    int beats = (int)g->lp[LP2_BEATS];
    if (beats < 1) beats = 4;
    return (long long)llround((double)g->sr * 60.0 / g->lp[LP2_BPM] * beats);
}

static void start_fresh(Eng *g, Track *tr, long long S) {
    tr->fresh = 1; tr->rs = S; tr->re = TFOREVER; tr->dub = 0; tr->undo = 0; tr->len = 0; tr->g = 0;
    long long bar = bar_frames(g);
    if (g->L0 <= 0 && bar <= 0) { tr->master = 1; tr->unit = 0; tr->base = S; }
    else {
        long long o = g->L0 > 0 ? g->om : g->mo;
        tr->master = g->L0 <= 0;
        tr->unit = g->L0 > 0 ? g->L0 : bar;         /* first take with the click on: whole bars */
        long long k = (long long)floor((double)(S - o) / tr->unit + 0.5);
        tr->base = o + k * tr->unit;
    }
    long long pre = tr->base > S ? tr->base - S : 0;
    if (pre) memset(tr->d + 2 * (g->maxf - pre), 0, sizeof(float) * 2 * (size_t)pre);
    tr->cap = g->maxf - pre;
    tr->zhi = 0;
    if (S > tr->base) zero_to(g, tr, S - tr->base);
    tr->st = T_REC;
    tr->ver++;
}

static void cancel_take(Eng *g, Track *tr) {
    /* len is kept until the next take so a playing track fades out instead of clicking off */
    if (tr->fresh) tr->len = 0;
    tr->fresh = 0; tr->dub = 0; tr->st = T_EMPTY; tr->undo = 0; tr->ver++;
    check_reset_len(g);
}

/* End of a new take at index time P: fixes its length (whole loops), then it keeps recording up to
 * the end of the last loop if P came early. */
static void close_fresh(Eng *g, Track *tr, long long P, int after) {
    if (!tr->fresh || tr->re != TFOREVER) return;
    if (tr->master && P - tr->rs < g->sr / 5) { cancel_take(g, tr); return; }   /* a slip, or A again in the count-in */
    if (P < tr->rs) P = tr->rs;
    if (tr->master && tr->unit <= 0) {
        long long len = P - tr->base;
        if (len > tr->cap) len = tr->cap;
        if (len < g->sr / 5) { cancel_take(g, tr); return; }      /* under 0.2 s: a slip, not a loop */
        tr->master = 1;
        tr->len = len; tr->ph = tr->base; tr->re = tr->base + len;
        g->L0 = len; g->om = tr->base;
    } else {
        long long L0 = tr->unit > 0 ? tr->unit : g->L0;
        if (L0 <= 0) { cancel_take(g, tr); return; }
        long long k = (long long)floor((double)(P - tr->base) / L0 + 0.5);
        if (k < 1) k = 1;
        while (k > 1 && k * L0 > tr->cap) k--;
        if (k * L0 > tr->cap) { cancel_take(g, tr); return; }
        tr->len = k * L0; tr->ph = tr->base;
        long long re = P > tr->base + tr->len ? P : tr->base + tr->len;
        if (re - tr->base > tr->cap) re = tr->base + tr->cap;
        tr->re = re;
        if (tr->master) { g->L0 = tr->len; g->om = tr->base; }     /* first loop, in whole bars */
    }
    tr->after = after;
}

/* All of the take has arrived: fold the bit recorded past the end onto the start, and the pre-roll
 * onto the end, so the loop runs on without a seam. */
static void finalize_fresh(Eng *g, Track *tr) {
    long long len = tr->len, hi = tr->re - tr->base;
    float *d = tr->d;
    for (long long i = len; i < hi; i++) { d[2 * (i - len)] += d[2 * i]; d[2 * (i - len) + 1] += d[2 * i + 1]; }
    long long pre = tr->base > tr->rs ? tr->base - tr->rs : 0;
    if (pre > len) pre = len;
    for (long long j = 0; j < pre; j++) {
        long long s = g->maxf - pre + j, o = len - pre + j;
        d[2 * o] += d[2 * s]; d[2 * o + 1] += d[2 * s + 1];
    }
    if (tr->master && tr->unit <= 0) {   /* 2 ms fades at the seam of a free first loop: no click */
        long long f = g->sr / 500;
        if (f > len / 4) f = len / 4;
        for (long long i = 0; i < f; i++) {
            float w = (float)(i + 1) / (float)(f + 1);
            d[2 * i] *= w; d[2 * i + 1] *= w;
            d[2 * (len - 1 - i)] *= w; d[2 * (len - 1 - i) + 1] *= w;
        }
    }
    tr->fresh = 0;
    if (!tr->dub) { tr->undo = 1; tr->ukind = 1; }
    tr->ver++;
}

static void start_dub(Eng *g, Track *tr, long long S) {
    (void)g;
    if (tr->len <= 0) return;
    if (tr->dub) { tr->de = TFOREVER; tr->st = T_DUB; return; }   /* the last one's tail is still arriving */
    memset(tr->ly, 0, sizeof(float) * 2 * (size_t)tr->len);
    tr->dub = 1; tr->ds = S; tr->de = TFOREVER; tr->undo = 0;
    tr->replace = g->lp[LP2_REPLACE] > 0.5f;
    tr->st = T_DUB;
}

static void close_dub(Track *tr, long long P, int st) {
    if (tr->dub && tr->de == TFOREVER) tr->de = P > tr->ds ? P : tr->ds;
    tr->st = st;
}

static void close_others(Eng *g, int except, long long now) {
    for (int i = 0; i < NTRK; i++) {
        if (i == except) continue;
        Track *tr = &g->t[i];
        if (tr->fresh && tr->re == TFOREVER) close_fresh(g, tr, now, T_PLAY);
        else if (tr->st == T_DUB) close_dub(tr, now, T_PLAY);
    }
}

static int any_sounding(Eng *g) {
    for (int i = 0; i < NTRK; i++) {
        Track *tr = &g->t[i];
        if (tr->st == T_PLAY || tr->st == T_DUB || tr->st == T_REC || tr->fresh || tr->dub || tr->g > 0) return 1;
    }
    return 0;
}

/* Starting again after everything was stopped: the loops start from the top, now. */
static void restart(Eng *g) {
    if (g->L0 <= 0 || any_sounding(g)) return;
    long long delta = g->G - g->om;
    g->om += delta;
    for (int i = 0; i < NTRK; i++) g->t[i].ph += delta;
}

static void loop_cmd(Eng *g, int cmd, int ti) {
    long long now = t_int(g);
    if (cmd == LC_ALL) {
        int any = 0;
        for (int i = 0; i < NTRK; i++) {
            int s = g->t[i].st;
            if (s == T_PLAY || s == T_DUB || s == T_REC) any = 1;
        }
        if (!any) restart(g);
        for (int i = 0; i < NTRK; i++) {
            Track *tr = &g->t[i];
            if (any) {
                if (tr->fresh && tr->re == TFOREVER) close_fresh(g, tr, now, T_STOP);
                else if (tr->fresh) tr->after = T_STOP;
                else if (tr->st == T_DUB) close_dub(tr, now, T_STOP);
                else if (tr->st == T_PLAY) tr->st = T_STOP;
            } else if (tr->st == T_STOP) tr->st = T_PLAY;
        }
        return;
    }
    if (cmd == LC_CLEARALL) {
        for (int i = 0; i < NTRK; i++) cancel_take(g, &g->t[i]);
        g->L0 = 0;
        return;
    }
    if (ti < 0 || ti >= NTRK) return;
    Track *tr = &g->t[ti];
    int recdub = g->lp[LP2_RECDUB] > 0.5f;
    switch (cmd) {
    case LC_RECDUB:
        if (tr->fresh && tr->re == TFOREVER) close_fresh(g, tr, now, recdub ? T_DUB : T_PLAY);
        else if (tr->fresh) break;                                   /* still finishing the take */
        else if (tr->st == T_EMPTY) {
            close_others(g, ti, now);
            long long bar = g->L0 <= 0 ? bar_frames(g) : 0;
            if (bar > 0 && g->lp[LP2_COUNTIN] > 0.5f && !any_sounding(g)) {
                /* count-in: the click restarts on beat 1 now, the take starts one bar later */
                g->mo = g->G;
                start_fresh(g, tr, g->mo + bar);
            } else start_fresh(g, tr, now);
        }
        else if (tr->st == T_PLAY) { close_others(g, ti, now); start_dub(g, tr, now); }
        else if (tr->st == T_DUB) close_dub(tr, now, T_PLAY);
        else if (tr->st == T_STOP) { restart(g); tr->st = T_PLAY; }
        break;
    case LC_STOP:
        if (tr->fresh && tr->re == TFOREVER) close_fresh(g, tr, now, T_STOP);
        else if (tr->fresh) tr->after = T_STOP;
        else if (tr->st == T_DUB) close_dub(tr, now, T_STOP);
        else if (tr->st == T_PLAY) tr->st = T_STOP;
        else if (tr->st == T_STOP) { restart(g); tr->st = T_PLAY; }
        break;
    case LC_UNDO:
        if (tr->fresh || tr->dub || tr->len <= 0) break;
        if (tr->undo == 1) {
            if (tr->ukind == 0) {
                for (long long i = 0; i < 2 * tr->len; i++) tr->d[i] -= tr->ly[i];
            } else {
                tr->st = T_EMPTY;
                check_reset_len(g);
            }
            tr->undo = 2;
        } else if (tr->undo == 2) {
            if (tr->ukind == 0) {
                for (long long i = 0; i < 2 * tr->len; i++) tr->d[i] += tr->ly[i];
            } else {
                if (g->L0 <= 0) { g->L0 = tr->len; g->om = tr->ph; tr->master = 1; }
                else if (tr->len % g->L0 != 0 || lmod(tr->ph - g->om, g->L0) != 0) break;   /* no longer fits */
                tr->st = T_PLAY;
            }
            tr->undo = 1;
        }
        tr->ver++;
        break;
    case LC_CLEAR:
        cancel_take(g, tr);
        break;
    }
}

/* Add one stream (pads or input) of n frames at index time t0 into a track that is recording. */
static void write_stream(Eng *g, Track *tr, long long t0, int n, const float *sL, const float *sR, float gain) {
    if (tr->fresh) {
        long long a = t0 > tr->rs ? t0 : tr->rs, b = t0 + n < tr->re ? t0 + n : tr->re;
        if (b - tr->base > tr->cap) b = tr->base + tr->cap;
        if (b > a) {
            if (b - tr->base > 0) zero_to(g, tr, b - tr->base);
            if (gain != 0.0f) {
                for (long long t = a; t < b; t++) {
                    long long li = t - tr->base, idx = li >= 0 ? li : g->maxf + li;
                    int k = (int)(t - t0);
                    tr->d[2 * idx] += sL[k] * gain;
                    tr->d[2 * idx + 1] += sR[k] * gain;
                }
            }
        }
    }
    if (tr->dub && tr->len > 0 && gain != 0.0f) {
        long long a = t0 > tr->ds ? t0 : tr->ds, b = t0 + n < tr->de ? t0 + n : tr->de;
        if (b > a) {
            long long pos = lmod(a - tr->ph, tr->len);
            for (long long t = a; t < b; t++) {
                int k = (int)(t - t0);
                float l = sL[k] * gain, r = sR[k] * gain;
                if (tr->replace) {           /* new sound in place of the old; ly keeps the difference */
                    tr->ly[2 * pos] += l - tr->d[2 * pos]; tr->ly[2 * pos + 1] += r - tr->d[2 * pos + 1];
                    tr->d[2 * pos] = l; tr->d[2 * pos + 1] = r;
                } else {
                    tr->d[2 * pos] += l; tr->d[2 * pos + 1] += r;
                    tr->ly[2 * pos] += l; tr->ly[2 * pos + 1] += r;
                }
                if (++pos >= tr->len) pos = 0;
            }
        }
    }
}

/* Input ring: filled by the capture thread (or eng_in_push), emptied once per block here. It keeps
 * about ring_target frames queued so a late capture period doesn't leave a gap; if the two sound
 * cards drift apart, the extra frames are skipped. */
static void ring_push(Eng *g, const float *st, int n) {
    unsigned h = g->rh, t = __atomic_load_n(&g->rt_, __ATOMIC_ACQUIRE);
    unsigned space = RING - (h - t);
    if ((unsigned)n > space) n = (int)space;
    for (int k = 0; k < n; k++) {
        unsigned i = (h + (unsigned)k) & (RING - 1);
        g->ring[2 * i] = st[2 * k]; g->ring[2 * i + 1] = st[2 * k + 1];
    }
    __atomic_store_n(&g->rh, h + (unsigned)n, __ATOMIC_RELEASE);
}

static void tap_push(Eng *g, const float *x, int n) {
    unsigned h = g->th_, t = __atomic_load_n(&g->tt_, __ATOMIC_ACQUIRE);
    unsigned space = TAPN - (h - t);
    if ((unsigned)n > space) n = (int)space;
    for (int k = 0; k < n; k++) g->tap[(h + (unsigned)k) & (TAPN - 1)] = x[k];
    __atomic_store_n(&g->th_, h + (unsigned)n, __ATOMIC_RELEASE);
}

/* The USB device and the Brick run on separate clocks (a USB synth or interface is often 0.1-0.3% off). Instead of skipping
 * frames when the ring fills up (a click every few seconds), the input is read at a slightly different
 * speed: a PI controller on the ring's fill level sets the ratio, and the frames are interpolated. */
static void pull_input(Eng *g, int n) {
    unsigned h = __atomic_load_n(&g->rh, __ATOMIC_ACQUIRE), t = g->rt_;
    unsigned avail = h - t;
    int per = g->cap_period > 0 ? g->cap_period : 256;
    int target = 2 * per;
    if (target < 256) target = 256;
    if (target < n) target = n;
    g->ring_target = target;
    int got = 0;
    double step = g->cap_step > 0 ? g->cap_step : 1.0;
    if (!g->in_started && avail >= (unsigned)(target + n * step)) {
        g->in_started = 1; g->rfrac = 0; g->fill_f = avail;
        if (g->ratio <= 0) g->ratio = 1.0;
    }
    if (g->in_started) {
        if (avail > (unsigned)(8 * target + 4096)) {        /* way behind (after a pause): jump */
            unsigned skip = avail - (unsigned)(target + n);
            t += skip; avail -= skip; g->in_drop += (int)skip; g->fill_f = avail;
        }
        g->fill_f += 0.02 * ((double)avail - g->fill_f);
        double err = (g->fill_f - (target + n)) / (double)target;   /* the fill before reading this block */
        g->drift_i += err * 2e-6;
        if (g->drift_i > 0.012) g->drift_i = 0.012;
        if (g->drift_i < -0.012) g->drift_i = -0.012;
        double r = 1.0 + 0.003 * err + g->drift_i;
        if (r > 1.012) r = 1.012;
        if (r < 0.988) r = 0.988;
        g->ratio = r;
        r *= step;                                  /* a device at another rate: linear conversion here */
        double end = g->rfrac + n * r;
        if ((double)avail >= end + 2.0) {
            for (int k = 0; k < n; k++) {
                double x = g->rfrac + k * r;
                unsigned i = (unsigned)x;
                float fr = (float)(x - i);
                unsigned i0 = (t + i) & (RING - 1), i1 = (t + i + 1) & (RING - 1);
                g->inL[k] = g->ring[2 * i0] + (g->ring[2 * i1] - g->ring[2 * i0]) * fr;
                g->inR[k] = g->ring[2 * i0 + 1] + (g->ring[2 * i1 + 1] - g->ring[2 * i0 + 1]) * fr;
            }
            unsigned adv = (unsigned)end;
            g->rfrac = end - adv;
            t += adv;
            got = 1;
        } else { g->in_under++; g->in_started = 0; }
    }
    if (!got) for (int k = 0; k < n; k++) g->inL[k] = g->inR[k] = 0;
    __atomic_store_n(&g->rt_, t, __ATOMIC_RELEASE);
}

/* Record into the tracks, then mix their playback and the input monitor into g->L / g->R (which hold
 * the pads/notes with their FX). Returns 1 if the looper made or recorded anything. */
static int loop_process(Eng *g, int n) {
    long long ti = t_int(g), te = t_ext(g);
    int src = (int)g->lp[LP2_SRC];
    float gin = g->lp[LP2_INGAIN];
    float gi = src != 0 ? 1.0f : 0.0f, ge = src != 1 ? gin : 0.0f;
    int active = 0;
    for (int i = 0; i < NTRK; i++) {
        Track *tr = &g->t[i];
        if (tr->fresh && tr->re == TFOREVER && ti + n - tr->base > tr->cap) close_fresh(g, tr, tr->base + tr->cap, T_PLAY);
        if (tr->fresh && tr->re != TFOREVER) {
            if (tr->st == T_REC && ti >= tr->re) {
                tr->st = tr->after;
                if (tr->after == T_DUB) start_dub(g, tr, tr->re);
            }
            if (ti >= tr->re && te >= tr->re) finalize_fresh(g, tr);
        }
        if (tr->dub && tr->de != TFOREVER && ti >= tr->de && te >= tr->de) {
            tr->dub = 0; tr->undo = 1; tr->ukind = 0; tr->ver++;
        }
        if (tr->fresh || tr->dub) {
            write_stream(g, tr, ti, n, g->L, g->R, gi);
            write_stream(g, tr, te, n, g->inL, g->inR, ge);
            active = 1;
        }
    }
    for (int i = 0; i < NTRK; i++) {
        Track *tr = &g->t[i];
        float tgt = ((tr->st == T_PLAY || tr->st == T_DUB) && tr->len > 0) ? 1.0f : 0.0f;
        if (tr->st == T_DUB && tr->replace) tgt = 0.0f;          /* replacing: the old part is not heard */
        if (tr->len <= 0 || (tgt == 0.0f && tr->g <= 0.0f)) { tr->g = 0; tr->pk *= 0.85f; continue; }
        float g0 = tr->g, g1 = tgt > g0 ? fminf(1.0f, g0 + 0.34f) : fmaxf(0.0f, g0 - 0.34f);
        tr->g = g1;
        float v = g->lp[LP2_V1 + i], pk = 0;
        long long len = tr->len, pos = lmod(g->G - tr->ph, len);
        const float *d = tr->d;
        for (int k = 0; k < n; k++) {
            float gk = (g0 + (g1 - g0) * (float)(k + 1) / (float)n) * v;
            float a = d[2 * pos] * gk, b = d[2 * pos + 1] * gk;
            g->L[k] += a; g->R[k] += b;
            float m = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
            if (m > pk) pk = m;
            if (++pos >= len) pos = 0;
        }
        tr->pk = tr->pk * 0.85f > pk ? tr->pk * 0.85f : pk;
        active = 1;
    }
    float pk[2] = {0, 0}, mon = g->lp[LP2_MON] > 0.5f ? gin : 0.0f;
    for (int k = 0; k < n; k++) {
        float a = g->inL[k] * gin, b = g->inR[k] * gin;
        if (fabsf(a) > pk[0]) pk[0] = fabsf(a);
        if (fabsf(b) > pk[1]) pk[1] = fabsf(b);
        if (mon != 0.0f) { g->L[k] += g->inL[k] * mon; g->R[k] += g->inR[k] * mon; }
    }
    for (int c = 0; c < 2; c++) g->in_pk[c] = g->in_pk[c] * 0.85f > pk[c] ? g->in_pk[c] * 0.85f : pk[c];
    if (g->cap_open || g->in_started) active = 1;
    if (g->mic_src) {               /* the mic is open only while something records (not in the count-in) */
        int rec = 0;
        for (int i = 0; i < NTRK; i++) {
            Track *tr = &g->t[i];
            if ((tr->fresh && ti + n >= tr->rs) || tr->dub) rec = 1;
        }
        g->mic_want = rec;
    }
    return active;
}

/* Metronome timing: beat length (frames) and the frame where beat 0 falls. With a loop the click
 * follows the loop: the loop is a whole number of bars, as near as possible to the BPM setting. */
static void metro_grid(Eng *g, double *beat, long long *origin, int *beats) {
    int b = (int)g->lp[LP2_BEATS];
    if (b < 1) b = 4;
    double bpm = g->lp[LP2_BPM] < 20.0f ? 120.0 : g->lp[LP2_BPM];
    double bl = (double)g->sr * 60.0 / bpm;
    *beats = b;
    if (g->L0 > 0) {
        long long bars = llround((double)g->L0 / (bl * b));
        if (bars < 1) bars = 1;
        *beat = (double)g->L0 / (double)(bars * b);
        *origin = g->om;
    } else {
        *beat = bl;
        *origin = g->mo;
    }
}

/* Click on every beat (higher on beat 1), added after the FX so the delay doesn't repeat it.
 * Block k of the output is heard at frame G + k, so the click lands where the loop expects it. */
static void click(Eng *g, int n) {
    if (g->lp[LP2_MET] < 0.5f) return;
    double beat; long long origin; int beats;
    metro_grid(g, &beat, &origin, &beats);
    float vol = g->lp[LP2_CLICK];
    long long G0 = g->G - n;                         /* G was already advanced */
    double clen = 0.03 * g->sr;
    for (int k = 0; k < n; k++) {
        double x = (double)(G0 + k - origin);
        double bi = floor(x / beat), ph = x - bi * beat;
        if (ph >= clen) continue;
        long long b = (long long)bi % beats;
        if (b < 0) b += beats;
        double f = b == 0 ? 1760.0 : 1100.0;
        float a = (float)(exp(-ph / (0.006 * g->sr)) * sin(6.283185307179586 * f * ph / g->sr)) * vol * (b == 0 ? 0.9f : 0.6f);
        g->L[k] += a; g->R[k] += a;
    }
}

/* Render one block into g->L / g->R: input -> looper -> FX -> volume. Returns 0 when idle. */
static int render(Eng *g) {
    int n = g->block;
    for (;;) {                      /* drain the event queue */
        EEv ev; int got = 0;
        LOCK(g->qlock);
        if (g->qh != g->qt) { ev = g->q[g->qh]; g->qh = (g->qh + 1) % E_QLEN; got = 1; }
        UNLOCK(g->qlock);
        if (!got) break;
        handle(g, &ev);
    }
    if (g->mic_direct) g->mic_direct = 0;        /* the writer already put the mic block in inL / inR */
    else pull_input(g, n);
    memset(g->L, 0, sizeof(float) * n);
    memset(g->R, 0, sizeof(float) * n);
    int loop_on = loop_process(g, n);
    if (g->lp[LP2_MET] > 0.5f) loop_on = 1;
    if ((g->fx->d_run && g->fx->gin_d < 1.0f) || (g->fx->v_run && g->fx->gin_v < 1.0f)) loop_on = 1;   /* FX tail */
    g->G += n;
    float tail = g->fxp[P_DMIX] > 0.0005f ? g->fxp[P_DTIME] * 1.5f : 0.0f;
    if (g->fxp[P_VMIX] > 0.0005f) tail += 3.0f;
    if (tail < 0.5f) tail = 0.5f;
    if (loop_on) g->quiet_blocks = 0;
    else g->quiet_blocks++;
    if (!loop_on && (float)g->quiet_blocks * n / g->sr > tail) {
        if (!g->idle) { fx_clear(g->fx); g->idle = 1; }
        g->peak[0] = g->peak[1] = 0;
        return 0;
    }
    g->idle = 0;
    fx_process(g->fx, g->L, g->R, n, g->fxp);
    click(g, n);
    float ov = g->lp[LP2_OUTVOL], pk[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        float x[2] = {g->L[i] * ov, g->R[i] * ov};
        for (int c = 0; c < 2; c++) {
            float ax = fabsf(x[c]);
            if (ax > 0.8f) x[c] = copysignf(0.8f + 0.2f * tanhf((ax - 0.8f) / 0.2f), x[c]);
            if (fabsf(x[c]) > pk[c]) pk[c] = fabsf(x[c]);
        }
        g->L[i] = x[0]; g->R[i] = x[1];
    }
    for (int c = 0; c < 2; c++) g->peak[c] = g->peak[c] * 0.9f > pk[c] ? g->peak[c] * 0.9f : pk[c];
    return 1;
}

/* Pull one block (used when Python drives the output: SDL on the desktop, aplay, tests).
 * out: n*2 interleaved floats. Idle blocks are +-1 LSB dither (the Brick's codec pops on true silence). */
API void eng_render(Eng *g, float *out) {
    LOCK(g->lock);
    int on = render(g);
    UNLOCK(g->lock);
    for (int i = 0; i < g->block; i++) {
        if (on) { out[2 * i] = g->L[i]; out[2 * i + 1] = g->R[i]; }
        else { float d = (i & 1 ? -1.01f : 1.01f) / 32767.0f; out[2 * i] = d; out[2 * i + 1] = d; }
    }
}

/* Status for the screen: [voices, peakL, peakR, load, idle, underruns, buffer_frames, rt, ok, blocks,
 * lfo value, lfo depth scale, then (slot, playhead 0..1) pairs]. Returns the number of floats written. */
API int eng_status(Eng *g, float *o, int max) {
    LOCK(g->lock);
    int n = 12;
    o[0] = (float)g->nv; o[1] = g->peak[0]; o[2] = g->peak[1]; o[3] = g->load; o[4] = (float)g->idle;
    o[5] = (float)g->underruns;
#if !defined(_WIN32)
    o[6] = (float)g->buffer_frames; o[7] = (float)g->rt; o[8] = (float)(g->pcmh != NULL);
#else
    o[6] = o[7] = o[8] = 0;
#endif
    o[9] = (float)g->blocks;
    o[10] = g->lfo_val; o[11] = g->lfo_scale;
    for (int i = 0; i < g->nv && n + 2 <= max; i++) {
        EVoice *v = &g->v[i];
        int L = v->e - v->s, total = g->slot[v->slot].n;
        double p = v->loop ? v->pos : (v->pos < L - 1 ? v->pos : L - 1);
        (void)L;
        double idx = v->reverse ? v->e - 1 - p : v->s + p;
        o[n++] = (float)v->slot;
        o[n++] = total ? (float)(idx / total) : 0.0f;
    }
    UNLOCK(g->lock);
    return n;
}


/* ---- looper calls from Python */
API void eng_loop(Eng *g, int cmd, int track) { push(g, EV_LOOP, cmd, track); }
API void eng_loop_params(Eng *g, const float *p) {
    LOCK(g->lock);
    if (g->L0 <= 0 && (p[LP2_MET] != g->lp[LP2_MET] || p[LP2_BPM] != g->lp[LP2_BPM] || p[LP2_BEATS] != g->lp[LP2_BEATS]))
        g->mo = g->G;                       /* the click starts on beat 1 now */
    memcpy(g->lp, p, sizeof(float) * LP2_COUNT);
    UNLOCK(g->lock);
}
API long long eng_loop_max(Eng *g) { return g->maxf; }
API float *eng_loop_data(Eng *g, int track) { return (track >= 0 && track < NTRK) ? g->t[track].d : NULL; }

/* Status for the screen, as doubles: [L0, om, G, in peak L, in peak R, capture open, capture period,
 * ring fill, input underruns, input frames skipped, ring target, max frames], then 12 per track:
 * [state, length, play position, phase, undo, new take, overdub, peak, version, fade, take closing, master] */
API int eng_loop_status(Eng *g, double *o, int max) {
    if (max < 20 + 12 * NTRK) return 0;
    LOCK(g->lock);
    o[0] = (double)g->L0; o[1] = (double)g->om; o[2] = (double)g->G;
    o[3] = g->in_pk[0]; o[4] = g->in_pk[1]; o[5] = g->cap_open; o[6] = g->cap_period;
    o[7] = (double)(unsigned)(g->rh - g->rt_); o[8] = g->in_under; o[9] = g->in_drop; o[10] = g->ring_target;
    o[11] = g->mic_on ? 1 : (g->mic_src ? 0.5 : 0);
#if !defined(_WIN32)
    o[12] = g->cap_reads; o[13] = g->cap_pk; o[14] = g->cap_ch;
#else
    o[12] = o[13] = o[14] = 0;
#endif
    {
        double beat; long long origin; int beats;
        metro_grid(g, &beat, &origin, &beats);
        o[15] = beat; o[16] = (double)origin; o[17] = beats; o[18] = (double)bar_frames(g); o[19] = (g->ratio - 1.0) * 1e6;
    }
    long long ti = t_int(g);
    for (int i = 0; i < NTRK; i++) {
        Track *tr = &g->t[i];
        double *q = o + 20 + 12 * i;
        q[0] = tr->st; q[1] = (double)tr->len;
        if (tr->fresh && tr->st == T_REC) q[2] = (double)(ti - tr->rs);
        else q[2] = tr->len > 0 ? (double)lmod(g->G - tr->ph, tr->len) : 0;
        q[3] = (double)tr->ph; q[4] = tr->undo; q[5] = tr->fresh; q[6] = tr->dub; q[7] = tr->pk;
        q[8] = tr->ver; q[9] = tr->g; q[10] = tr->fresh && tr->re != TFOREVER; q[11] = tr->master;
    }
    UNLOCK(g->lock);
    return 20 + 12 * NTRK;
}

/* Empty every track at once (loading another project). */
API void eng_loop_reset(Eng *g) {
    LOCK(g->lock);
    for (int i = 0; i < NTRK; i++) { cancel_take(g, &g->t[i]); g->t[i].len = 0; g->t[i].g = 0; }
    g->L0 = 0;
    UNLOCK(g->lock);
}

/* A saved loop back into a track (stopped). Its sample 0 lines up with the loop start. */
API void eng_loop_load(Eng *g, int track, const float *d, long long n, long long L0) {
    if (track < 0 || track >= NTRK || n < 2) return;
    LOCK(g->lock);
    Track *tr = &g->t[track];
    if (n > g->maxf) n = g->maxf;
    memcpy(tr->d, d, sizeof(float) * 2 * (size_t)n);
    if (g->L0 <= 0) { g->L0 = L0 > 0 ? L0 : n; g->om = g->G; }
    tr->len = n; tr->ph = g->om; tr->st = T_STOP; tr->fresh = tr->dub = 0; tr->undo = 0; tr->g = 0;
    tr->master = 0; tr->ver++;
    UNLOCK(g->lock);
}

/* Input frames from Python (tests, or a capture Python runs itself). Stereo interleaved. */
API void eng_in_push(Eng *g, const float *st, int n) { ring_push(g, st, n); }

/* 1: the input is the built-in mic (see mic_src). */
API void eng_mic(Eng *g, int on) { g->mic_src = on; if (!on) g->mic_want = 0; }

/* Mono copy of the input for the sampler's REC page. */
API void eng_tap(Eng *g, int on) {
    if (on && !g->tap_on) g->tt_ = g->th_;
    g->tap_on = on;
}
API int eng_tap_read(Eng *g, float *out, int max) {
    unsigned h = __atomic_load_n(&g->th_, __ATOMIC_ACQUIRE), t = g->tt_;
    int n = (int)(h - t);
    if (n > max) n = max;
    for (int k = 0; k < n; k++) out[k] = g->tap[(t + (unsigned)k) & (TAPN - 1)];
    __atomic_store_n(&g->tt_, t + (unsigned)n, __ATOMIC_RELEASE);
    return n;
}

#if !defined(_WIN32)
/* ---- ALSA writer thread (libasound loaded at run time, so the library builds without its headers) */
typedef int (*f_open)(void **, const char *, int, int);
typedef int (*f_setp)(void *, int, int, unsigned, unsigned, int, unsigned);
typedef int (*f_getp)(void *, unsigned long *, unsigned long *);
typedef long (*f_writei)(void *, const void *, unsigned long);
typedef long (*f_readi)(void *, void *, unsigned long);
typedef int (*f_recover)(void *, int, int);
typedef int (*f_pcm)(void *);
typedef const char *(*f_strerror)(int);
static struct { f_open open; f_setp setp; f_getp getp; f_writei writei; f_readi readi; f_recover recover; f_pcm drop, close, start; f_strerror strerror; } A;

static int alsa_load(void) {
    static int done = 0;
    if (done) return done > 0;
    void *h = dlopen("libasound.so.2", RTLD_NOW);
    if (!h) { done = -1; return 0; }
    A.open = (f_open)dlsym(h, "snd_pcm_open");
    A.setp = (f_setp)dlsym(h, "snd_pcm_set_params");
    A.getp = (f_getp)dlsym(h, "snd_pcm_get_params");
    A.writei = (f_writei)dlsym(h, "snd_pcm_writei");
    A.readi = (f_readi)dlsym(h, "snd_pcm_readi");
    A.start = (f_pcm)dlsym(h, "snd_pcm_start");
    A.recover = (f_recover)dlsym(h, "snd_pcm_recover");
    A.drop = (f_pcm)dlsym(h, "snd_pcm_drop");
    A.close = (f_pcm)dlsym(h, "snd_pcm_close");
    A.strerror = (f_strerror)dlsym(h, "snd_strerror");
    done = (A.open && A.setp && A.getp && A.writei && A.readi && A.start && A.recover && A.drop && A.close && A.strerror) ? 1 : -1;
    return done > 0;
}

static void a_close(Eng *g) {
    if (g->pcmh) { A.drop(g->pcmh); A.close(g->pcmh); g->pcmh = NULL; }
}

static int a_open(Eng *g) {
    char list[256];
    strncpy(list, g->devs, sizeof(list) - 1); list[sizeof(list) - 1] = 0;
    unsigned us = (unsigned)((double)g->want_frames * 1e6 / g->sr);
    for (char *dev = strtok(list, "\n"); dev; dev = strtok(NULL, "\n")) {
        void *pcm = NULL;
        int err = A.open(&pcm, dev, 0, 0);                      /* playback, blocking */
        if (err < 0) { snprintf(g->err, sizeof(g->err), "%s: %s", dev, A.strerror(err)); continue; }
        err = A.setp(pcm, 2, 3, 2, (unsigned)g->sr, 1, us);    /* S16_LE, RW interleaved, stereo, resample */
        if (err < 0) { snprintf(g->err, sizeof(g->err), "%s set_params: %s", dev, A.strerror(err)); A.close(pcm); continue; }
        unsigned long buf = 0, per = 0;
        A.getp(pcm, &buf, &per);
        g->pcmh = pcm; g->buffer_frames = (int)buf; g->period_frames = (int)per;
        g->opened_frames = g->want_frames;
        strncpy(g->dev, dev, sizeof(g->dev) - 1);
        g->err[0] = 0;
        return 1;
    }
    return 0;
}

static double cpu_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *writer(void *arg) {
    Eng *g = (Eng *)arg;
    struct sched_param sp;
    sp.sched_priority = 50;
    g->rt = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
    int n = g->block;
    double period = (double)n / g->sr;
    void *mic = NULL;
    static short mbuf[4096];
    while (g->run) {
        if (g->paused) {
            a_close(g);
            g->closed = 1;
            usleep(10000);
            continue;
        }
        if (g->mic_want) {
            if (!mic) {
                a_close(g);                                    /* the speaker must be closed for the mic */
                const char *devs[2] = {"hw:0,0", "plughw:0,0"};
                for (int i = 0; i < 2 && !mic; i++) {
                    if (A.open(&mic, devs[i], 1, 0) < 0) { mic = NULL; continue; }
                    int err = A.setp(mic, 2, 3, 1, (unsigned)g->sr, 1, 20000);
                    if (err < 0) {
                        snprintf(g->cap_err, sizeof(g->cap_err), "mic %s: %s", devs[i], A.strerror(err));
                        A.close(mic); mic = NULL;
                    }
                }
                if (!mic) { fprintf(stderr, "mic: can't open (%s)\n", g->cap_err); g->mic_want = 0; continue; }
                A.start(mic);
                fprintf(stderr, "mic: recording, speaker closed\n");
                g->mic_on = 1; g->cap_err[0] = 0;
            }
            long r = A.readi(mic, mbuf, (unsigned long)n);
            if (r < 0) {
                if (A.recover(mic, (int)r, 1) < 0) {
                    fprintf(stderr, "mic: read failed: %s\n", A.strerror((int)r));
                    A.close(mic); mic = NULL; g->mic_on = 0; g->mic_want = 0;
                } else A.start(mic);
                continue;
            }
            LOCK(g->lock);
            for (int i = 0; i < n; i++) g->inL[i] = g->inR[i] = i < r ? mbuf[i] / 32768.0f : 0.0f;
            g->mic_direct = 1;
            render(g);                                         /* recorded, not heard */
            UNLOCK(g->lock);
            g->blocks++;
            continue;
        }
        if (mic) { A.drop(mic); A.close(mic); mic = NULL; g->mic_on = 0; fprintf(stderr, "mic: done, speaker back on\n"); }
        g->closed = 0;
        if (g->pcmh && g->want_frames != g->opened_frames) a_close(g);
        if (!g->pcmh && !a_open(g)) { g->open_fails++; usleep(500000); continue; }
        g->open_fails = 0;
        double c0 = cpu_now();
        LOCK(g->lock);
        int on = render(g);
        UNLOCK(g->lock);
        for (int i = 0; i < n; i++) {
            if (on) {
                float l = clampf(g->L[i], -1, 1), r = clampf(g->R[i], -1, 1);
                g->pcm[2 * i] = (short)(l * 32767.0f); g->pcm[2 * i + 1] = (short)(r * 32767.0f);
            } else {
                short d = (i & 1) ? -1 : 1;
                g->pcm[2 * i] = d; g->pcm[2 * i + 1] = d;
            }
        }
        g->load = g->load * 0.9f + 0.1f * (float)((cpu_now() - c0) / period);
        g->blocks++;
        int done = 0;
        while (done < n && g->pcmh) {
            long w = A.writei(g->pcmh, g->pcm + done * 2, (unsigned long)(n - done));
            if (w < 0) {
                g->underruns++;
                if (A.recover(g->pcmh, (int)w, 1) < 0) {
                    snprintf(g->err, sizeof(g->err), "write failed: %s", A.strerror((int)w));
                    a_close(g);
                }
                continue;
            }
            done += (int)w;
        }
    }
    a_close(g);
    g->closed = 1;
    return NULL;
}


/* ---- capture thread: a USB interface / mic / synth into the input ring (and the REC page's tap) */
/* ALSA format codes: S16_LE 2, S24_LE 6, S32_LE 10, S24_3LE 32 */
static const char *fmt_name(int f) { return f == 32 ? "S24_3LE" : f == 10 ? "S32_LE" : f == 6 ? "S24_LE" : "S16_LE"; }
static int fmt_bytes(int f) { return f == 32 ? 3 : (f == 10 || f == 6) ? 4 : 2; }
/* one little-endian sample -> -1..1; *nz is set when it isn't exactly 0 */
static inline float pcm_sample(const unsigned char *b, long i, int fmt, int bps, int *nz) {
    const unsigned char *q = b + i * bps;
    int v;
    float s;
    if (fmt == 32) { v = q[0] | (q[1] << 8) | (q[2] << 16); if (v & 0x800000) v -= 0x1000000; s = v / 8388608.0f; }
    else if (fmt == 6) { v = (q[0] | (q[1] << 8) | (q[2] << 16)); if (v & 0x800000) v -= 0x1000000; s = v / 8388608.0f; }
    else if (fmt == 10) { v = (int)((unsigned)q[0] | ((unsigned)q[1] << 8) | ((unsigned)q[2] << 16) | ((unsigned)q[3] << 24)); s = v / 2147483648.0f; }
    else { v = (short)(q[0] | (q[1] << 8)); s = v / 32768.0f; }
    if (v) *nz = 1;
    return s;
}

static void *capture(void *arg) {
    Eng *g = (Eng *)arg;
    struct sched_param sp;
    sp.sched_priority = 52;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    void *pcm = NULL;
    int gen = -1, per = 128, chn = 2;
    char devs[512] = "", last_err[160] = "";
    static unsigned char buf[4096 * 2 * 4];
    int fmt = 2, bps = 2;                                        /* open format (ALSA code) and bytes per sample */
    static float st[4096 * 2], mono[4096];
    /* Switching input (another device, channel, or off): 20 ms S-curve fade out on the old input, then
       20 ms fade in on the new one, so the monitor and a running take don't click. */
    const int FN = g->sr / 50;
    int wfmt = 2, wrate = 0;
    int fade = 0, fpos = 0;                                      /* fade: 0 none, -1 out, +1 in */
    int curmode = 2;                                             /* channel mode of the open input */
    while (g->cap_run) {
        if (gen != g->cap_gen && pcm && fade != -1) { fade = -1; fpos = 0; }
        if (gen != g->cap_gen && (!pcm || (fade == -1 && fpos >= FN))) {
            if (pcm) { A.drop(pcm); A.close(pcm); pcm = NULL; fprintf(stderr, "capture: closed\n"); }
            fade = 0; fpos = 0;
            LOCK(g->qlock);
            strncpy(devs, g->cap_dev, sizeof(devs) - 1); devs[sizeof(devs) - 1] = 0;
            gen = g->cap_gen;
            curmode = g->cap_mode;
            wfmt = g->cap_req_fmt > 0 ? g->cap_req_fmt : 2;
            wrate = g->cap_req_rate > 0 ? g->cap_req_rate : g->sr;
            UNLOCK(g->qlock);
            g->cap_step = 1.0;
            g->cap_open = 0; g->cap_err[0] = 0; last_err[0] = 0;
        }
        if (!devs[0]) { usleep(50000); continue; }
        if (!pcm) {
            /* the same way BrickFX opens a USB device: each device name in turn, stereo then mono, 6 ms */
            char list[512], *save = NULL;
            strncpy(list, devs, sizeof(list) - 1); list[sizeof(list) - 1] = 0;
            for (char *dev = strtok_r(list, "\n", &save); dev && !pcm; dev = strtok_r(NULL, "\n", &save)) {
                for (int c = 2; c >= 1 && !pcm; c--) {
                    /* the device's own format and rate first; then S16_LE at that rate; then S16_LE at the
                       engine rate (ALSA converts) */
                    int tf[3] = {wfmt, 2, 2}, tr[3] = {wrate, wrate, g->sr};
                    for (int a = 0; a < 3 && !pcm; a++) {
                        if (a > 0 && tf[a] == tf[a - 1] && tr[a] == tr[a - 1]) continue;
                        void *p = NULL;
                        int err = A.open(&p, dev, 1, 0);                  /* capture, blocking */
                        if (err < 0) { snprintf(g->cap_err, sizeof(g->cap_err), "%s: %s", dev, A.strerror(err)); c = 0; break; }
                        err = A.setp(p, tf[a], 3, (unsigned)c, (unsigned)tr[a], 1, 6000);
                        if (err < 0) {
                            snprintf(g->cap_err, sizeof(g->cap_err), "%s %dch %s %d Hz: %s", dev, c, fmt_name(tf[a]), tr[a], A.strerror(err));
                            A.close(p);
                            continue;
                        }
                        unsigned long bsz = 0, p_ = 0;
                        A.getp(p, &bsz, &p_);
                        per = (p_ > 0 && p_ <= 4096) ? (int)p_ : 128;
                        chn = c; pcm = p; fmt = tf[a]; bps = fmt_bytes(fmt);
                        g->cap_fmt = fmt; g->cap_rate = tr[a];
                        g->cap_step = (double)tr[a] / g->sr;
                        strncpy(g->cap_used, dev, sizeof(g->cap_used) - 1);
                        A.start(pcm);
                        fprintf(stderr, "capture: opened %s, %s, %d ch, %d Hz%s, period %d, buffer %lu frames\n", dev, fmt_name(fmt), c,
                                tr[a], tr[a] != g->sr ? " (converted to the engine rate in the code)" : "", per, bsz);
                    }
                }
            }
            if (!pcm) {
                if (strcmp(last_err, g->cap_err) != 0) {
                    fprintf(stderr, "capture: can't open (%s), retrying every second\n", g->cap_err);
                    strncpy(last_err, g->cap_err, sizeof(last_err) - 1);
                }
                usleep(1000000);
                continue;
            }
            g->cap_period = per; g->cap_ch = chn; g->cap_open = 1; g->cap_err[0] = 0; last_err[0] = 0;
            g->cap_lvl[0] = g->cap_lvl[1] = 0; g->cap_nz = 0;
            fade = 1; fpos = 0;
        }
        long r = A.readi(pcm, buf, (unsigned long)per);
        if (r < 0) {
            if (A.recover(pcm, (int)r, 1) < 0) {
                snprintf(g->cap_err, sizeof(g->cap_err), "read failed: %s", A.strerror((int)r));
                fprintf(stderr, "capture: %s (unplugged?)\n", g->cap_err);
                A.close(pcm); pcm = NULL; g->cap_open = 0; g->cap_step = 1.0; fade = 0; usleep(500000);
            } else A.start(pcm);
            continue;
        }
        if (r == 0) continue;
        g->cap_reads += 1;
        int mode = curmode;
        float pk = 0, lv0 = g->cap_lvl[0], lv1 = g->cap_lvl[1];
        int nz = 0;
        for (long k = 0; k < r; k++) {
            float l, rr;
            if (chn == 2) { l = pcm_sample(buf, 2 * k, fmt, bps, &nz); rr = pcm_sample(buf, 2 * k + 1, fmt, bps, &nz); }
            else l = rr = pcm_sample(buf, k, fmt, bps, &nz);
            if (fabsf(l) > lv0) lv0 = fabsf(l);
            if (fabsf(rr) > lv1) lv1 = fabsf(rr);
            if (fabsf(l) > pk) pk = fabsf(l);
            if (fabsf(rr) > pk) pk = fabsf(rr);
            if (fade) {
                float x = fpos < FN ? (float)fpos / FN : 1.0f, w = 0.5f - 0.5f * cosf(3.14159265f * x);
                if (fade < 0) w = 1.0f - w;
                l *= w; rr *= w;
                if (fpos < FN) fpos++;
                else if (fade > 0) fade = 0;                     /* faded in; a fade out waits at 0 */
            }
            if (mode == 0) { st[2 * k] = st[2 * k + 1] = l; mono[k] = l; }
            else if (mode == 1) { st[2 * k] = st[2 * k + 1] = rr; mono[k] = rr; }
            else { st[2 * k] = l; st[2 * k + 1] = rr; mono[k] = 0.5f * (l + rr); }
        }
        g->cap_pk = fmaxf(g->cap_pk * 0.97f, pk);
        g->cap_lvl[0] = lv0; g->cap_lvl[1] = lv1;
        if (nz) g->cap_nz = 1;
        ring_push(g, st, (int)r);
        if (g->tap_on) tap_push(g, mono, (int)r);
    }
    if (pcm) { A.drop(pcm); A.close(pcm); }
    g->cap_open = 0;
    return NULL;
}

/* dev: ALSA capture devices to try, one per line ("" closes it). mode: 0 = input 1, 1 = input 2, 2 = both (stereo). */
API int eng_cap_open(Eng *g, const char *dev, int mode) {
    if (!alsa_load()) return 0;
    LOCK(g->qlock);
    strncpy(g->cap_dev, dev ? dev : "", sizeof(g->cap_dev) - 1);
    g->cap_mode = mode;
    g->cap_gen++;
    UNLOCK(g->qlock);
    if (!g->cap_run) {
        g->cap_run = 1;
        if (pthread_create(&g->cth, NULL, capture, g) != 0) { g->cap_run = 0; return 0; }
    }
    return 1;
}
API const char *eng_cap_error(Eng *g) { return g->cap_err; }
/* format: ALSA code (2 S16_LE, 6 S24_LE, 10 S32_LE, 32 S24_3LE); rate: one the device has (0 = engine rate).
   Used by the next eng_cap_open. */
API void eng_cap_format(Eng *g, int format, int rate) { g->cap_req_fmt = format; g->cap_req_rate = rate; }
/* o: [biggest sample L, R since the last call, any non-zero sample, open format, open rate]; resets the levels */
API int eng_cap_level(Eng *g, double *o) {
    o[0] = g->cap_lvl[0]; o[1] = g->cap_lvl[1]; o[2] = g->cap_nz; o[3] = g->cap_fmt; o[4] = g->cap_rate;
    g->cap_lvl[0] = g->cap_lvl[1] = 0; g->cap_nz = 0;
    return g->cap_open;
}

/* devs: newline-separated ALSA device names to try in order. frames: wanted buffer size. */
API int eng_start(Eng *g, const char *devs, int frames) {
    if (!alsa_load()) return 0;
    strncpy(g->devs, devs, sizeof(g->devs) - 1);
    g->want_frames = frames;
    g->run = 1;
    if (pthread_create(&g->th, NULL, writer, g) != 0) { g->run = 0; return 0; }
    return 1;
}
API void eng_set_buffer(Eng *g, int frames) { g->want_frames = frames; }
API int eng_pause(Eng *g, int on) {
    /* on=1: stop and free the sound card (waits up to 1 s); on=0: start playing again */
    if (!on) { g->paused = 0; return 1; }
    g->closed = 0;
    g->paused = 1;
    for (int i = 0; i < 100 && !g->closed; i++) usleep(10000);
    return g->closed;
}
API const char *eng_device(Eng *g) { return g->dev; }
API const char *eng_error(Eng *g) { return g->err; }
API int eng_period(Eng *g) { return g->period_frames; }
#endif
