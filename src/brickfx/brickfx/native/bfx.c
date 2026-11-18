/*
 * bfx - native audio engine for BrickFX (stereo multi-effect for the TrimUI Brick).
 *
 *   USB audio in (capture thread) -> ring buffer -> adaptive resampler -> FX chain -> ALSA out (writer thread)
 *   FX chain: DRIVE -> MOD -> DELAY -> REVERB
 *
 * The USB audio device and the Brick's own codec run on two different clocks. They are never
 * exactly 48000 Hz, so the ring buffer between them slowly fills up or runs dry. The writer reads the
 * ring with a fractional step (ratio ~ 1 +- a few hundred ppm) steered by a PI controller that keeps the
 * fill level on target: no clicks or dropouts however long it runs.
 *
 * Python only sends parameters and reads a status array; the audio threads never touch Python.
 * Build: see build.sh (zig cc cross-compiles for the Brick and Linux x86_64).
 *
 * Copyright (C) 2026 Alessandro Pietrosanti - BrickWorld (https://brickworldhub.com)
 * Free software under the GNU General Public License version 3 (see LICENSE). NO WARRANTY.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#define API __attribute__((visibility("default")))

#define RB (1 << 14)               /* input ring: 16384 frames (0.34 s) */
#define RBM (RB - 1)
#define DSIZE (1 << 17)            /* delay line: 131072 frames (2.7 s at 48 kHz) */
#define DMASK (DSIZE - 1)
#define CSIZE 4096                 /* chorus / vibrato line: 85 ms */
#define CMASK (CSIZE - 1)
#define PSIZE 16384                /* reverb pre-delay: 0.34 s */
#define PMASK (PSIZE - 1)
#define MAXBLOCK 1024
#define TWO_PI 6.283185307179586

/* Parameter order: ENGINE_KEYS in brickfx.py */
enum { P_ON, P_TIME, P_FB, P_MIX, P_MODE, P_WIDTH, P_HICUT, P_LOCUT, P_MODDEPTH, P_MODRATE, P_TRAILS,
       P_INGAIN, P_OUTGAIN, P_ROUTE,
       P_DRV_TYPE, P_DRV_AMT, P_DRV_TONE, P_DRV_LEVEL,
       P_MOD_TYPE, P_MOD_RATE, P_MOD_DEPTH, P_MOD_MIX,
       P_DLY_TYPE,
       P_REV_TYPE, P_REV_DECAY, P_REV_PRE, P_REV_DAMP, P_REV_MOD, P_REV_MIX,
       P_EQ_TYPE, P_EQ_LOW, P_EQ_LOWF, P_EQ_MID, P_EQ_MIDF, P_EQ_Q, P_EQ_HIGH, P_EQ_HIGHF,
       P_COUNT };
#define SCOPE (1 << 13)            /* oscilloscope history: 8192 frames (170 ms) of input and output */
#define SMASK (SCOPE - 1)
enum { MODE_STEREO, MODE_PINGPONG, MODE_MONO };
enum { ROUTE_STEREO, ROUTE_L, ROUTE_R, ROUTE_SUM };
enum { DRV_OFF, DRV_DRIVE, DRV_FUZZ, DRV_CRUSH };
enum { MOD_OFF, MOD_CHORUS, MOD_PHASER, MOD_TREMOLO, MOD_PAN, MOD_VIBRATO };
enum { DLY_OFF, DLY_DIGITAL, DLY_TAPE, DLY_ANALOG };
enum { REV_OFF, REV_ROOM, REV_PLATE, REV_HALL };

typedef struct { float b0, b1, b2, a1, a2, z1[2], z2[2], fc; int on; } Biquad;
/* per-section switching: cur = type running, f = effect fade, gin = input into the effect (spillover),
 * spill = switched off but the tail still rings, quiet/spill_t = seconds of silence / of tail so far */
typedef struct { int cur; float f, gin, quiet, spill_t; int spill, force; } Xfade;

/* Reverb line: power-of-two buffer, all lines share one write counter */
typedef struct { float *buf; unsigned mask; int len; } Line;

#define NTANK 8
typedef struct {
    Line pre, in[4], tank[NTANK];
    unsigned w;
    float bw, lp[2], lastL, lastR, precur;
    double lph;
    float size;
    int type;
    int taps[14];
} Reverb;

typedef struct {
    int sr, block;
    float p[P_COUNT];
    float on_s;                     /* smoothed chain on/off (0 bypass .. 1 on) */
    float in_s, out_s;              /* smoothed input / output gains */
    int cleared;
    /* drive */
    float drv_lp[2], drv_hp[2], drv_dc[2][2], drv_hold[2];
    int drv_cnt;
    /* mod */
    float *cl, *cr;
    unsigned cw;
    double mod_ph;
    float ph_x[2][6], ph_y[2][6], ph_fb[2], ph_a[2];
    float mod_mix_s, mod_depth_s;
    int mod_prev;
    /* delay */
    float *dl, *dr;
    unsigned dw;
    float dcur;                     /* current delay (samples), glides toward the target: tape-like, no clicks */
    double mph, flut;               /* delay modulation LFO, tape flutter */
    float mdepth;
    float dry_s, wet_s;
    float tone_z[2];
    int dly_prev;
    Biquad lp, hp;
    /* eq */
    Biquad eq[3];
    float eq_set_v[7];
    /* oscilloscope */
    float *sc_in, *sc_out;
    unsigned sc_w;
    /* reverb */
    Reverb rv;
    float rdry_s, rwet_s;
    int rev_prev;
    /* input ring (written by the capture thread) */
    float *rb;
    volatile unsigned long long wpos;
    double rpos, ratio, fill_avg, integ;
    int priming, target;
    /* buffers */
    float L[MAXBLOCK], R[MAXBLOCK], DL[MAXBLOCK], DR[MAXBLOCK], ON[MAXBLOCK], XL[MAXBLOCK], XR[MAXBLOCK], XG[MAXBLOCK];
    Xfade xf[5];
    short pcm[MAXBLOCK * 2];
    /* status */
    float in_pk[2], out_pk[2], load;
    int underruns, cap_xruns, rs_under, rs_over, blocks, nan_resets;
    pthread_mutex_t lock;
    /* threads */
    pthread_t pth, cth;
    volatile int run, cap_run, play_open, cap_open, rt_play, rt_cap, want_frames, opened_frames;
    volatile int cap_change, cap_period, cap_channels, play_buffer, play_period, cap_rate, cap_buffer, cap_opens;
    volatile double cap_frames, cap_reads;
    volatile float cap_raw_pk;
    void *ph, *ch;
    char play_devs[256], cap_devs[512], play_dev[96], cap_dev[96], err[160], cap_err[160];
    /* input switch: fade the input out, then the capture thread swaps to cap_pending; it fades back in */
    char cap_pending[512];
    volatile int sw_req, cap_switch;
    /* capture format and rate asked for (the device's own best ones), active and pending (applied at the swap) */
    int cap_fmt_want, cap_rate_want, pend_fmt, pend_rate, cap_fmt;
    double rs_pos; float rs_prev[2]; int rs_have;              /* linear rate conversion cap_rate -> sr */
    float lvl_pk[2]; volatile long lvl_nz;                     /* per-channel peak + non-zero samples, for log.txt */
    float in_g;
    int in_hold;
    unsigned char cbuf[MAXBLOCK * 16];
} Eng;

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

static inline float softclip(float x, float knee) {
    float ax = fabsf(x);
    if (ax <= knee) return x;
    float r = 1.0f - knee;
    return copysignf(knee + r * tanhf((ax - knee) / r), x);
}

/* 4-point Hermite (Catmull-Rom) interpolation */
static inline float herm(float y0, float y1, float y2, float y3, float t) {
    float c1 = 0.5f * (y2 - y0);
    float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
    return ((c3 * t + c2) * t + c1) * t + y1;
}

/* one-pole low-pass coefficient for a cutoff (y += c * (x - y)) */
static inline float op_coef(float fc, float sr) { return 1.0f - expf(-6.2831853f * fc / sr); }

/* Butterworth (Q 0.707) low/high-pass, RBJ formulas */
static void bq_set(Biquad *b, int hp, float fc, float sr) {
    fc = clampf(fc, 10.0f, sr * 0.45f);
    float w0 = 2.0f * 3.14159265f * fc / sr, c = cosf(w0), s = sinf(w0), al = s / (2 * 0.70710678f);
    float a0 = 1 + al, b0, b1;
    if (hp) { b0 = (1 + c) / 2; b1 = -(1 + c); } else { b0 = (1 - c) / 2; b1 = 1 - c; }
    b->b0 = b0 / a0; b->b1 = b1 / a0; b->b2 = b0 / a0; b->a1 = -2 * c / a0; b->a2 = (1 - al) / a0;
    b->fc = fc;
}
static inline float bq_run(Biquad *b, int c, float x) {
    float y = b->b0 * x + b->z1[c];
    b->z1[c] = b->b1 * x - b->a1 * y + b->z2[c];
    b->z2[c] = b->b2 * x - b->a2 * y;
    return y;
}

/* ---------------------------------------------------------------- reverb (Dattorro plate topology) */
static const int DAT_IN[4] = {142, 107, 379, 277};
static const int DAT_TANK[NTANK] = {672, 4453, 1800, 3720, 908, 4217, 2656, 3163};
/* output taps (Dattorro 1997): line index, offset */
static const int TAP_L[7][2] = {{5, 266}, {5, 2974}, {6, 1913}, {7, 1996}, {1, 1990}, {2, 187}, {3, 1066}};
static const int TAP_R[7][2] = {{1, 353}, {1, 3627}, {2, 1228}, {3, 2673}, {5, 2111}, {6, 335}, {7, 121}};
static const float TAP_SIGN[7] = {1, 1, -1, 1, -1, -1, -1};
#define MAX_SIZE 1.7f

static void line_alloc(Line *l, int maxlen) {
    unsigned cap = 64;
    while (cap < (unsigned)maxlen + 64) cap <<= 1;
    l->buf = (float *)calloc(cap, sizeof(float));
    l->mask = cap - 1;
    l->len = maxlen;
}
static inline float line_rd(const Line *l, unsigned w, int d) { return l->buf[(w - (unsigned)d) & l->mask]; }
static inline void line_wr(Line *l, unsigned w, float x) { l->buf[w & l->mask] = x; }

static void rv_alloc(Reverb *r, int sr) {
    float k = sr / 29761.0f;
    line_alloc(&r->pre, PSIZE - 64);
    for (int i = 0; i < 4; i++) line_alloc(&r->in[i], (int)(DAT_IN[i] * k) + 4);
    for (int i = 0; i < NTANK; i++) line_alloc(&r->tank[i], (int)(DAT_TANK[i] * k * MAX_SIZE) + 80);
    r->type = -1;
}
static void rv_clear(Reverb *r) {
    memset(r->pre.buf, 0, (r->pre.mask + 1) * sizeof(float));
    for (int i = 0; i < 4; i++) memset(r->in[i].buf, 0, (r->in[i].mask + 1) * sizeof(float));
    for (int i = 0; i < NTANK; i++) memset(r->tank[i].buf, 0, (r->tank[i].mask + 1) * sizeof(float));
    r->bw = r->lp[0] = r->lp[1] = r->lastL = r->lastR = 0;
}
static void rv_size(Reverb *r, int type, int sr) {
    float k = sr / 29761.0f;
    float size = type == REV_ROOM ? 0.55f : type == REV_HALL ? MAX_SIZE : 1.0f;
    r->size = size;
    for (int i = 0; i < 4; i++) r->in[i].len = (int)(DAT_IN[i] * k);
    for (int i = 0; i < NTANK; i++) r->tank[i].len = (int)(DAT_TANK[i] * k * size);
    for (int i = 0; i < 7; i++) { r->taps[i] = (int)(TAP_L[i][1] * k * size); r->taps[7 + i] = (int)(TAP_R[i][1] * k * size); }
    r->type = type;
}

/* allpass on a line: v = delayed, u = x + g v written, y = v - g u */
static inline float ap(Line *l, unsigned w, float x, float g) {
    float v = line_rd(l, w, l->len);
    float u = x + g * v;
    line_wr(l, w, u);
    return v - g * u;
}
/* modulated allpass: fractional (linear) read */
static inline float ap_mod(Line *l, unsigned w, float x, float g, float d) {
    int i = (int)d;
    float f = d - i;
    float a = line_rd(l, w, i), b = line_rd(l, w, i + 1);
    float v = a + (b - a) * f;
    float u = x + g * v;
    line_wr(l, w, u);
    return v - g * u;
}

/* Long tails used to come out up to 14 dB louder than the dry signal and hit the output limiter.
 * The level into the tank now follows type and decay (measured with a sustained pad), so the wet
 * stays about 2 dB under the dry at mix 100% and reaches the dry level only at full decay. */
static const float REV_DEC_PTS[5] = {0.3f, 0.6f, 0.85f, 0.95f, 1.0f};
static const float REV_COMP_DB[4][5] = {
    {0, 0, 0, 0, 0},
    {-2.9f, -5.8f, -10.6f, -13.6f, -14.4f},     /* room */
    {-0.1f, -0.4f, -2.1f, -3.8f, -5.4f},        /* plate */
    {-3.9f, -6.2f, -8.5f, -10.0f, -10.9f},      /* hall */
};
static float rev_gain(int type, float d) {
    const float *c = REV_COMP_DB[type < 0 || type > 3 ? 0 : type];
    float db;
    if (d <= REV_DEC_PTS[0]) db = c[0];
    else {
        int k = 0;
        while (k < 3 && d > REV_DEC_PTS[k + 1]) k++;
        float t = (d - REV_DEC_PTS[k]) / (REV_DEC_PTS[k + 1] - REV_DEC_PTS[k]);
        db = c[k] + (c[k + 1] - c[k]) * clampf(t, 0, 1);
    }
    return powf(10.0f, db / 20.0f);
}

static void reverb_block(Eng *g, float *L, float *R, int n, int key) {
    const float *p = g->p;
    Reverb *r = &g->rv;
    int type = key;
    float sr = (float)g->sr;
    if (type == REV_OFF) { g->rev_prev = 0; return; }
    if (g->rev_prev == 0 || type != r->type) { rv_size(r, type, g->sr); rv_clear(r); r->precur = -1; }
    g->rev_prev = type;
    float decay = 0.2f + 0.78f * clampf(p[P_REV_DECAY], 0, 1);
    float gin = rev_gain(type, clampf(p[P_REV_DECAY], 0, 1));
    float damp = 0.05f + 0.8f * clampf(p[P_REV_DAMP], 0, 1);
    float pre_t = clampf(p[P_REV_PRE], 0, 0.25f) * sr + 1;
    if (r->precur < 0) r->precur = pre_t;
    float exc = clampf(p[P_REV_MOD], 0, 1) * 24.0f * (sr / 29761.0f) * r->size;
    double lstep = 0.9 / sr;
    float mix = clampf(p[P_REV_MIX], 0, 1);
    float dry_t = mix < 0.5f ? 1.0f : 2.0f * (1.0f - mix);
    float wet_t = mix < 0.5f ? 2.0f * mix : 1.0f;
    float ramp = 1.0f - expf(-1.0f / (0.012f * sr));
    float bwc = 0.6f;
    Line *t = r->tank;
    int l0 = t[0].len, l4 = t[4].len;
    unsigned w = r->w;
    float dry_s = g->rdry_s, wet_s = g->rwet_s, lastL = r->lastL, lastR = r->lastR, bw = r->bw;
    float lp0 = r->lp[0], lp1 = r->lp[1], precur = r->precur;
    double lph = r->lph;
    for (int i = 0; i < n; i++) {
        dry_s += (dry_t - dry_s) * ramp;
        wet_s += (wet_t - wet_s) * ramp;
        precur += (pre_t - precur) * 0.0005f;
        lph += lstep; if (lph >= 1.0) lph -= 1.0;
        float s = sinf((float)(TWO_PI * lph)), c = cosf((float)(TWO_PI * lph));
        /* input: mono sum -> pre-delay -> bandwidth -> 4 diffusers */
        float x = 0.5f * (L[i] + R[i]) * gin;
        line_wr(&r->pre, w, x);
        int pi0 = (int)precur; float pf = precur - pi0;
        float pa = line_rd(&r->pre, w, pi0), pb = line_rd(&r->pre, w, pi0 + 1);
        x = pa + (pb - pa) * pf;
        bw += bwc * (x - bw);
        x = bw;
        x = ap(&r->in[0], w, x, 0.75f);
        x = ap(&r->in[1], w, x, 0.75f);
        x = ap(&r->in[2], w, x, 0.625f);
        x = ap(&r->in[3], w, x, 0.625f);
        /* tank, left half */
        float a = x + decay * lastR;
        a = ap_mod(&t[0], w, a, -0.7f, l0 + exc * (1 + s));
        float d1 = line_rd(&t[1], w, t[1].len); line_wr(&t[1], w, a);
        lp0 += (1 - damp) * (d1 - lp0);
        float b = ap(&t[2], w, lp0 * decay, 0.5f);
        float leftOut = line_rd(&t[3], w, t[3].len); line_wr(&t[3], w, b);
        /* tank, right half */
        float a2 = x + decay * leftOut;
        a2 = ap_mod(&t[4], w, a2, -0.7f, l4 + exc * (1 + c));
        float d2 = line_rd(&t[5], w, t[5].len); line_wr(&t[5], w, a2);
        lp1 += (1 - damp) * (d2 - lp1);
        float b2 = ap(&t[6], w, lp1 * decay, 0.5f);
        float rightOut = line_rd(&t[7], w, t[7].len); line_wr(&t[7], w, b2);
        lastL = softclip(leftOut, 0.9f); lastR = softclip(rightOut, 0.9f);
        /* output taps */
        float yl = 0, yr = 0;
        for (int k = 0; k < 7; k++) {
            yl += TAP_SIGN[k] * line_rd(&t[TAP_L[k][0]], w + 1, r->taps[k]);
            yr += TAP_SIGN[k] * line_rd(&t[TAP_R[k][0]], w + 1, r->taps[7 + k]);
        }
        w++;
        L[i] = dry_s * L[i] + wet_s * 0.6f * yl;
        R[i] = dry_s * R[i] + wet_s * 0.6f * yr;
    }
    r->w = w; r->lastL = lastL; r->lastR = lastR; r->bw = bw; r->lp[0] = lp0; r->lp[1] = lp1; r->precur = precur;
    r->lph = lph; g->rdry_s = dry_s; g->rwet_s = wet_s;
}

/* ---------------------------------------------------------------- drive */
static void drive_block(Eng *g, float *L, float *R, int n, int key) {
    const float *p = g->p;
    int type = key;
    if (type == DRV_OFF) return;
    float sr = (float)g->sr;
    float amt = clampf(p[P_DRV_AMT], 0, 1), tone = clampf(p[P_DRV_TONE], 0, 1), lvl = p[P_DRV_LEVEL];
    float *ch[2] = {L, R};
    if (type == DRV_CRUSH) {
        float q = powf(2.0f, 15.0f - amt * 12.0f);             /* 16 bit down to 4 bit */
        int hold = 1 + (int)((1.0f - tone) * (1.0f - tone) * 31.0f);  /* tone 100% = full rate */
        int cnt = g->drv_cnt;
        for (int i = 0; i < n; i++) {
            if (cnt == 0) { g->drv_hold[0] = roundf(L[i] * q) / q; g->drv_hold[1] = roundf(R[i] * q) / q; }
            L[i] = g->drv_hold[0] * lvl; R[i] = g->drv_hold[1] * lvl;
            if (++cnt >= hold) cnt = 0;
        }
        g->drv_cnt = cnt;
        return;
    }
    float gain = type == DRV_FUZZ ? 1.0f + 80.0f * amt * amt : 1.0f + 24.0f * amt * amt;
    /* loudness stays about level as the gain goes up (for a -10 dB signal) */
    float comp = type == DRV_FUZZ ? 0.3f / fminf(1.0f, gain * 0.3f) : 0.3f / tanhf(gain * 0.3f);
    float lpc = op_coef(700.0f * powf(24.0f, tone), sr);       /* 700 Hz .. 16 kHz */
    float hpc = op_coef(type == DRV_FUZZ ? 150.0f : 30.0f, sr);
    float dcc = op_coef(8.0f, sr);
    const float bias = 0.12f, tb = tanhf(0.12f);
    for (int c = 0; c < 2; c++) {
        float *x = ch[c], lp = g->drv_lp[c], hp = g->drv_hp[c], dc = g->drv_dc[c][0];
        for (int i = 0; i < n; i++) {
            float v = x[i];
            hp += hpc * (v - hp);
            v -= hp;                                            /* tighten the lows before the clipper */
            float y;
            if (type == DRV_FUZZ) {
                float u = v * gain;
                y = clampf(u, -1.0f, 1.0f);
                y = 0.85f * y + 0.15f * tanhf(u * 0.5f);
            } else {
                y = tanhf(v * gain + bias) - tb;                /* a little asymmetric: warmer, even harmonics */
            }
            dc += dcc * (y - dc);
            y -= dc;
            lp += lpc * (y - lp);
            x[i] = lp * comp * lvl;
        }
        g->drv_lp[c] = lp; g->drv_hp[c] = hp; g->drv_dc[c][0] = dc;
    }
}

/* ---------------------------------------------------------------- modulation */
static void mod_block(Eng *g, float *L, float *R, int n, int key) {
    const float *p = g->p;
    int type = key;
    float sr = (float)g->sr;
    if (type == MOD_OFF) { g->mod_prev = 0; return; }
    if (g->mod_prev != type) {
        memset(g->ph_x, 0, sizeof(g->ph_x)); memset(g->ph_y, 0, sizeof(g->ph_y));
        g->ph_fb[0] = g->ph_fb[1] = 0;
        g->mod_prev = type;
    }
    double step = clampf(p[P_MOD_RATE], 0.02f, 20.0f) / sr;
    float depth_t = clampf(p[P_MOD_DEPTH], 0, 1), mix_t = clampf(p[P_MOD_MIX], 0, 1);
    float ramp = 1.0f - expf(-1.0f / (0.02f * sr));
    float depth = g->mod_depth_s, mix = g->mod_mix_s;
    double ph = g->mod_ph;
    unsigned w = g->cw;
    for (int i = 0; i < n; i++) {
        depth += (depth_t - depth) * ramp;
        mix += (mix_t - mix) * ramp;
        ph += step; if (ph >= 1.0) ph -= 1.0;
        float s = sinf((float)(TWO_PI * ph)), c = cosf((float)(TWO_PI * ph));
        float xl = L[i], xr = R[i], yl, yr;
        if (type == MOD_CHORUS || type == MOD_VIBRATO) {
            float base = type == MOD_CHORUS ? 0.007f : 0.004f, sweep = type == MOD_CHORUS ? 0.006f : 0.0035f;
            float dlt = (base + sweep * depth * 0.5f * (1 + s)) * sr;
            float drt = (base + sweep * depth * 0.5f * (1 + c)) * sr;
            float rp = (float)w - dlt; int i0 = (int)floorf(rp); float f = rp - i0;
            yl = herm(g->cl[(i0 - 1) & CMASK], g->cl[i0 & CMASK], g->cl[(i0 + 1) & CMASK], g->cl[(i0 + 2) & CMASK], f);
            rp = (float)w - drt; i0 = (int)floorf(rp); f = rp - i0;
            yr = herm(g->cr[(i0 - 1) & CMASK], g->cr[i0 & CMASK], g->cr[(i0 + 1) & CMASK], g->cr[(i0 + 2) & CMASK], f);
        } else if (type == MOD_PHASER) {
            if ((i & 15) == 0) {                                /* coefficients at control rate */
                for (int k = 0; k < 2; k++) {
                    float l = k ? c : s;
                    float fc = 180.0f * powf(2.0f, depth * 4.6f * 0.5f * (1 + l));
                    float t = tanf(3.14159265f * fminf(fc, sr * 0.45f) / sr);
                    g->ph_a[k] = (t - 1) / (t + 1);
                }
            }
            float in[2] = {xl, xr}, out[2];
            for (int k = 0; k < 2; k++) {
                float a = g->ph_a[k], v = in[k] + 0.55f * g->ph_fb[k];
                for (int st = 0; st < 6; st++) {
                    float y = a * v + g->ph_x[k][st] - a * g->ph_y[k][st];
                    g->ph_x[k][st] = v; g->ph_y[k][st] = y;
                    v = y;
                }
                g->ph_fb[k] = v;
                out[k] = v;
            }
            yl = out[0]; yr = out[1];
        } else if (type == MOD_TREMOLO) {
            float gn = 1.0f - depth * 0.5f * (1 - s);
            yl = xl * gn; yr = xr * gn;
        } else {                                                 /* auto-pan */
            float gl = 1.0f - depth * 0.5f * (1 + s), gr = 1.0f - depth * 0.5f * (1 - s);
            yl = xl * gl; yr = xr * gr;
        }
        L[i] = xl + (yl - xl) * mix;
        R[i] = xr + (yr - xr) * mix;
        w = (w + 1) & CMASK;
    }
    g->mod_ph = ph; g->mod_depth_s = depth; g->mod_mix_s = mix;
}

/* ---------------------------------------------------------------- delay */
static void delay_block(Eng *g, float *L, float *R, int n, int key) {
    const float *p = g->p;
    int type = key / 4;                                         /* key = type * 4 + mode */
    float sr = (float)g->sr;
    if (type == DLY_OFF) { g->dly_prev = 0; return; }
    if (g->dly_prev == 0) {                                      /* switched on: start from a clean line */
        memset(g->dl, 0, DSIZE * sizeof(float)); memset(g->dr, 0, DSIZE * sizeof(float));
        g->dcur = -1;
    }
    g->dly_prev = type;
    float hc = p[P_HICUT], lc = p[P_LOCUT];
    g->lp.on = hc < 19000.0f;
    g->hp.on = lc > 21.0f;
    if (g->lp.on && fabsf(hc - g->lp.fc) > 0.5f) bq_set(&g->lp, 0, hc, sr);
    if (g->hp.on && fabsf(lc - g->hp.fc) > 0.5f) bq_set(&g->hp, 1, lc, sr);

    float target = clampf(p[P_TIME] * sr, 8.0f, (float)(DSIZE - 600));
    if (g->dcur < 0) g->dcur = target;
    float glide = 1.0f - expf(-1.0f / (0.07f * sr));        /* ~70 ms time constant */
    float fb = clampf(p[P_FB], 0.0f, 1.0f);
    int mode = key % 4;
    float mix = clampf(p[P_MIX], 0, 1);
    float dry_t = mix < 0.5f ? 1.0f : 2.0f * (1.0f - mix);   /* both at unity at 50% */
    float wet_t = mix < 0.5f ? 2.0f * mix : 1.0f;
    float ramp = 1.0f - expf(-1.0f / (0.012f * sr));
    float md_t = clampf(p[P_MODDEPTH], 0, 1) * 0.003f * sr;  /* up to 3 ms of wow */
    double mstep = clampf(p[P_MODRATE], 0.05f, 10.0f) / sr;
    double fstep = 5.7 / sr;
    float width = clampf(p[P_WIDTH], 0, 1);
    /* character: tape = flutter, more saturation, softer top; analog (BBD) = dark repeats, warm clip */
    float knee = type == DLY_DIGITAL ? 0.8f : 0.45f;
    float flut = type == DLY_TAPE ? 0.00012f * sr : 0.0f;
    float tc = type == DLY_TAPE ? op_coef(6500.0f, sr) : type == DLY_ANALOG ? op_coef(3200.0f, sr) : 1.0f;

    float *bl = g->dl, *br = g->dr;
    unsigned w = g->dw;
    float dry_s = g->dry_s, wet_s = g->wet_s, mdepth = g->mdepth, dcur = g->dcur, tz0 = g->tone_z[0], tz1 = g->tone_z[1];
    double mph = g->mph, fph = g->flut;
    Biquad *lp = &g->lp, *hp = &g->hp;
    for (int i = 0; i < n; i++) {
        dry_s += (dry_t - dry_s) * ramp;
        wet_s += (wet_t - wet_s) * ramp;
        mdepth += (md_t - mdepth) * ramp;
        dcur += (target - dcur) * glide;
        mph += mstep; if (mph >= 1.0) mph -= 1.0;
        fph += fstep; if (fph >= 1.0) fph -= 1.0;
        float ml = 0, mr = 0;
        if (mdepth > 0.01f) {
            float s = sinf((float)(TWO_PI * mph));
            float c = cosf((float)(TWO_PI * mph));
            ml = mdepth * 0.5f * (1 + s);
            mr = mode == MODE_MONO ? ml : mdepth * 0.5f * (1 + c);      /* 90 deg apart: wide repeats */
        }
        if (flut > 0) {
            float fl = flut * sinf((float)(TWO_PI * fph));
            ml += flut + fl; mr += flut + fl;
        }
        float tl = dcur + ml, tr = dcur + mr;
        float rp = (float)w - tl; int i0 = (int)floorf(rp); float fr = rp - i0;
        float yl = herm(bl[(i0 - 1) & DMASK], bl[i0 & DMASK], bl[(i0 + 1) & DMASK], bl[(i0 + 2) & DMASK], fr);
        rp = (float)w - tr; i0 = (int)floorf(rp); fr = rp - i0;
        float yr = herm(br[(i0 - 1) & DMASK], br[i0 & DMASK], br[(i0 + 1) & DMASK], br[(i0 + 2) & DMASK], fr);
        if (tc < 1.0f) { tz0 += tc * (yl - tz0); tz1 += tc * (yr - tz1); yl = tz0; yr = tz1; }
        if (lp->on) { yl = bq_run(lp, 0, yl); yr = bq_run(lp, 1, yr); }
        if (hp->on) { yl = bq_run(hp, 0, yl); yr = bq_run(hp, 1, yr); }
        float xl = L[i], xr = R[i];
        float wl, wr;
        if (mode == MODE_PINGPONG) {
            wl = 0.5f * (xl + xr) + fb * yr;
            wr = fb * yl;
        } else if (mode == MODE_MONO) {
            float m = 0.5f * (yl + yr);
            wl = wr = 0.5f * (xl + xr) + fb * m;
            yl = yr = m;
        } else {
            wl = xl + fb * yl;
            wr = xr + fb * yr;
        }
        bl[w] = softclip(wl, knee);
        br[w] = softclip(wr, knee);
        w = (w + 1) & DMASK;
        float mid = 0.5f * (yl + yr), side = 0.5f * (yl - yr) * width;
        L[i] = dry_s * xl + wet_s * (mid + side);
        R[i] = dry_s * xr + wet_s * (mid - side);
    }
    g->dw = w; g->dry_s = dry_s; g->wet_s = wet_s; g->mdepth = mdepth; g->dcur = dcur; g->mph = mph; g->flut = fph;
    g->tone_z[0] = tz0; g->tone_z[1] = tz1;
}

/* ---------------------------------------------------------------- EQ: low shelf, mid peak, high shelf (RBJ) */
static void eq_coef(Biquad *b, int kind, float f, float gdb, float q, float sr) {
    f = clampf(f, 20.0f, sr * 0.45f);
    float A = powf(10.0f, gdb / 40.0f), w0 = 2.0f * 3.14159265f * f / sr, c = cosf(w0), sn = sinf(w0);
    float b0, b1, b2, a0, a1, a2;
    if (kind == 1) {
        float al = sn / (2.0f * clampf(q, 0.2f, 10.0f));
        b0 = 1 + al * A; b1 = -2 * c; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * c; a2 = 1 - al / A;
    } else {
        float al = sn / 2.0f * 1.41421356f, sa = 2.0f * sqrtf(A) * al;
        if (kind == 0) {
            b0 = A * ((A + 1) - (A - 1) * c + sa); b1 = 2 * A * ((A - 1) - (A + 1) * c); b2 = A * ((A + 1) - (A - 1) * c - sa);
            a0 = (A + 1) + (A - 1) * c + sa; a1 = -2 * ((A - 1) + (A + 1) * c); a2 = (A + 1) + (A - 1) * c - sa;
        } else {
            b0 = A * ((A + 1) + (A - 1) * c + sa); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - sa);
            a0 = (A + 1) - (A - 1) * c + sa; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - sa;
        }
    }
    b->b0 = b0 / a0; b->b1 = b1 / a0; b->b2 = b2 / a0; b->a1 = a1 / a0; b->a2 = a2 / a0;
}

static void eq_block(Eng *g, float *L, float *R, int n, int key) {
    const float *p = g->p;
    if (key == 0) return;
    float sr = (float)g->sr;
    float v[7] = {p[P_EQ_LOW], p[P_EQ_LOWF], p[P_EQ_MID], p[P_EQ_MIDF], p[P_EQ_Q], p[P_EQ_HIGH], p[P_EQ_HIGHF]};
    if (memcmp(v, g->eq_set_v, sizeof(v)) != 0) {
        eq_coef(&g->eq[0], 0, v[1], v[0], 0.707f, sr);
        eq_coef(&g->eq[1], 1, v[3], v[2], v[4], sr);
        eq_coef(&g->eq[2], 2, v[6], v[5], 0.707f, sr);
        memcpy(g->eq_set_v, v, sizeof(v));
    }
    for (int k = 0; k < 3; k++) {
        if (fabsf(v[k == 0 ? 0 : k == 1 ? 2 : 5]) < 0.05f) continue;     /* flat band: skip it */
        Biquad *b = &g->eq[k];
        for (int i = 0; i < n; i++) { L[i] = bq_run(b, 0, L[i]); R[i] = bq_run(b, 1, R[i]); }
    }
}

/* Oscilloscope: copy the last n frames (oldest first) of the input (which 0) or output (1). */
API int eng_scope(Eng *g, int which, float *out, int n) {
    if (n > SCOPE - MAXBLOCK) n = SCOPE - MAXBLOCK;
    pthread_mutex_lock(&g->lock);
    const float *b = which ? g->sc_out : g->sc_in;
    unsigned w = g->sc_w;
    for (int i = 0; i < n; i++) {
        unsigned k = ((w - (unsigned)n + (unsigned)i) & SMASK) * 2;
        out[2 * i] = b[k]; out[2 * i + 1] = b[k + 1];
    }
    pthread_mutex_unlock(&g->lock);
    return n;
}

/* ---------------------------------------------------------------- smooth switching
 * Switching a section on/off, or changing its type (or the delay mode), used to jump from one signal path
 * to another in one sample: a click. Now each section fades its effect out to the dry signal, changes over
 * while silent (clearing its state as before), and fades the new one in. */
typedef void (*SectionFn)(Eng *, float *, float *, int, int);

/* spill_s > 0: delay/reverb spillover. Switching the section OFF only closes its input (fade_s); the
 * echoes / tail keep playing and die away on their own, while the dry signal comes back at full level.
 * The section really stops once its output has been silent for spill_s seconds (longer than the delay
 * time, so no echo is still on its way), or after 20 s (then a 1 s fade, for 100% feedback). */
static void section(Eng *g, Xfade *x, int key, float fade_s, SectionFn fn, int *prev, float spill_s,
                    float *L, float *R, int n) {
    if (x->cur == 0 && key == 0) return;                        /* off and staying off */
    if (x->gin <= 0.0f && !x->spill) x->gin = 1.0f;              /* first use */
    if (x->cur == key && x->f >= 1.0f && x->gin >= 1.0f) { x->spill = 0; fn(g, L, R, n, key); return; }
    if (x->cur == 0) { x->cur = key; x->f = 0.0f; x->gin = 1.0f; x->spill = 0; x->force = 0; }
    float sr = (float)g->sr, step = 1.0f / (fade_s * sr);
    /* what to do: spill (key off, tail rings), resume (back to the same type), or crossfade to another type */
    int spilling = key == 0 && spill_s > 0.0f && !x->force;
    if (spilling && !x->spill) { x->spill = 1; x->quiet = 0; x->spill_t = 0; }
    if (key == x->cur) x->spill = 0;
    float fdir = (spilling || key == x->cur) ? 1.0f : -1.0f;
    float gdir = spilling ? -1.0f : 1.0f;
    float fstep = x->force ? 1.0f / sr : step;                  /* forced end of a never-ending tail: 1 s */
    float *dl = g->XL, *dr = g->XR;
    memcpy(dl, L, n * sizeof(float)); memcpy(dr, R, n * sizeof(float));
    /* the effect gets its input through gin; what it no longer gets passes straight through, so the dry
     * signal stays at full level: out = in * (1 - gin) + effect(in * gin) */
    float gin = x->gin;
    float *gv = g->XG;
    for (int i = 0; i < n; i++) {
        gin += gdir * step;
        gin = gin < 0 ? 0 : gin > 1 ? 1 : gin;
        gv[i] = gin * gin * (3.0f - 2.0f * gin);
        L[i] *= gv[i]; R[i] *= gv[i];
    }
    x->gin = gin;
    fn(g, L, R, n, x->cur);
    float f = x->f, pk = 0;
    for (int i = 0; i < n; i++) {
        float yl = dl[i] * (1.0f - gv[i]) + L[i], yr = dr[i] * (1.0f - gv[i]) + R[i];
        if (gv[i] <= 0.0f) pk = fmaxf(pk, fmaxf(fabsf(L[i]), fabsf(R[i])));
        f += fdir * fstep;
        f = f < 0 ? 0 : f > 1 ? 1 : f;
        float k = f * f * (3.0f - 2.0f * f);                     /* smooth S-curve */
        L[i] = dl[i] + (yl - dl[i]) * k;
        R[i] = dr[i] + (yr - dr[i]) * k;
    }
    x->f = f;
    if (x->spill && gin <= 0.0f) {
        float dt = (float)n / sr;
        x->spill_t += dt;
        x->quiet = pk < 3.0e-5f ? x->quiet + dt : 0.0f;          /* -90 dB */
        if (x->quiet >= spill_s) {                               /* the tail has died away: stop */
            x->cur = 0; x->spill = 0; x->f = 0; x->gin = 1.0f;
            if (prev) *prev = 0;
            return;
        }
        if (x->spill_t > 20.0f) x->force = 1;                    /* 100% feedback never dies: fade it out */
    }
    if (fdir < 0 && f <= 0.0f) {                                 /* faded out: change over now */
        x->cur = key; x->spill = 0; x->force = 0; x->gin = 1.0f;
        if (prev) *prev = 0;                                     /* the next start clears old echoes/tails */
    }
}

/* ---------------------------------------------------------------- chain */
API Eng *eng_create(int sr, int block) {
    Eng *g = (Eng *)calloc(1, sizeof(Eng));
    if (!g) return NULL;
    if (block > MAXBLOCK) block = MAXBLOCK;
    g->sr = sr; g->block = block;
    g->dl = (float *)calloc(DSIZE, sizeof(float));
    g->dr = (float *)calloc(DSIZE, sizeof(float));
    g->cl = (float *)calloc(CSIZE, sizeof(float));
    g->cr = (float *)calloc(CSIZE, sizeof(float));
    g->rb = (float *)calloc(RB * 2, sizeof(float));
    rv_alloc(&g->rv, sr);
    g->sc_in = (float *)calloc(SCOPE * 2, sizeof(float));
    g->sc_out = (float *)calloc(SCOPE * 2, sizeof(float));
    g->eq_set_v[0] = -999;
    g->p[P_ON] = 1; g->p[P_TIME] = 0.375f; g->p[P_FB] = 0.4f; g->p[P_MIX] = 0.3f; g->p[P_WIDTH] = 1;
    g->p[P_HICUT] = 8000; g->p[P_LOCUT] = 80; g->p[P_MODRATE] = 0.5f; g->p[P_TRAILS] = 1;
    g->p[P_INGAIN] = 1; g->p[P_OUTGAIN] = 1; g->p[P_DLY_TYPE] = DLY_DIGITAL; g->p[P_DRV_LEVEL] = 1;
    g->dcur = -1; g->on_s = 1; g->in_s = 1; g->out_s = 1; g->dry_s = 1; g->wet_s = 0.6f; g->rdry_s = 1;
    g->ratio = 1.0; g->priming = 1; g->target = 512; g->in_g = 0;
    g->cap_fmt_want = g->pend_fmt = 2; g->cap_rate_want = g->pend_rate = sr;
    g->lp.fc = g->hp.fc = -1;
    pthread_mutex_init(&g->lock, NULL);
    return g;
}

API void eng_set_params(Eng *g, const float *p) {
    pthread_mutex_lock(&g->lock);
    memcpy(g->p, p, sizeof(float) * P_COUNT);
    pthread_mutex_unlock(&g->lock);
}

static void clear_tails(Eng *g) {
    memset(g->dl, 0, DSIZE * sizeof(float));
    memset(g->dr, 0, DSIZE * sizeof(float));
    memset(g->cl, 0, CSIZE * sizeof(float));
    memset(g->cr, 0, CSIZE * sizeof(float));
    rv_clear(&g->rv);
}

API void eng_clear(Eng *g) {
    pthread_mutex_lock(&g->lock);
    clear_tails(g);
    pthread_mutex_unlock(&g->lock);
}

static void fx_block(Eng *g, float *L, float *R, int n) {
    const float *p = g->p;
    float sr = (float)g->sr;
    /* input routing and gain */
    int route = (int)p[P_ROUTE];
    float gi0 = g->in_s, gi1 = p[P_INGAIN];
    g->in_s = gi1;
    float pk0 = 0, pk1 = 0;
    for (int i = 0; i < n; i++) {
        float gi = gi0 + (gi1 - gi0) * (i + 1) / n;
        float l = L[i], r = R[i];
        if (route == ROUTE_L) r = l; else if (route == ROUTE_R) l = r;
        else if (route == ROUTE_SUM) { l = r = 0.5f * (l + r); }
        l *= gi; r *= gi;
        L[i] = l; R[i] = r;
        float al = fabsf(l), ar = fabsf(r);
        if (al > pk0) pk0 = al;
        if (ar > pk1) pk1 = ar;
    }
    g->in_pk[0] = fmaxf(g->in_pk[0] * 0.9f, pk0);
    g->in_pk[1] = fmaxf(g->in_pk[1] * 0.9f, pk1);
    for (int i = 0; i < n; i++) {
        unsigned k = ((g->sc_w + (unsigned)i) & SMASK) * 2;
        g->sc_in[k] = L[i]; g->sc_in[k + 1] = R[i];
    }

    /* chain on/off: the input to the chain fades out, the dry signal fades in.
     * With trails the delay and reverb tails keep playing after bypass. */
    float on_t = p[P_ON] > 0.5f ? 1.0f : 0.0f;
    float ramp = 1.0f - expf(-1.0f / (0.012f * sr));
    float on_s = g->on_s;
    for (int i = 0; i < n; i++) {
        on_s += (on_t - on_s) * ramp;
        g->ON[i] = on_s;
        g->DL[i] = L[i]; g->DR[i] = R[i];
        L[i] *= on_s; R[i] *= on_s;
    }
    g->on_s = on_s;
    int trails = p[P_TRAILS] > 0.5f;
    if (!trails && on_s < 1e-4f && on_t < 0.5f) {
        if (!g->cleared) { clear_tails(g); g->cleared = 1; }
        for (int i = 0; i < n; i++) { L[i] = R[i] = 0; }
    } else {
        g->cleared = 0;
        int dt = (int)p[P_DLY_TYPE];
        section(g, &g->xf[0], (int)p[P_DRV_TYPE], 0.010f, drive_block, NULL, 0, L, R, n);
        /* the chorus/vibrato line always holds the recent input, so a chorus fading in has signal from the start */
        for (int i = 0; i < n; i++) { unsigned k = (g->cw + (unsigned)i) & CMASK; g->cl[k] = L[i]; g->cr[k] = R[i]; }
        section(g, &g->xf[1], (int)p[P_MOD_TYPE], 0.025f, mod_block, &g->mod_prev, 0, L, R, n);
        g->cw = (g->cw + (unsigned)n) & CMASK;
        section(g, &g->xf[2], dt ? dt * 4 + (int)p[P_MODE] : 0, 0.030f, delay_block, &g->dly_prev,
                g->dcur / g->sr + 0.3f, L, R, n);
        section(g, &g->xf[3], (int)p[P_REV_TYPE], 0.080f, reverb_block, &g->rev_prev,
                clampf(p[P_REV_PRE], 0, 0.25f) + 0.4f, L, R, n);
        section(g, &g->xf[4], (int)p[P_EQ_TYPE], 0.010f, eq_block, NULL, 0, L, R, n);
        float chk = 0;
        for (int i = 0; i < n; i++) chk += L[i] + R[i];
        if (!isfinite(chk)) {                       /* never let a bad number stick in the lines */
            clear_tails(g);
            memset(g->lp.z1, 0, sizeof(g->lp.z1)); memset(g->lp.z2, 0, sizeof(g->lp.z2));
            memset(g->hp.z1, 0, sizeof(g->hp.z1)); memset(g->hp.z2, 0, sizeof(g->hp.z2));
            memset(g->ph_x, 0, sizeof(g->ph_x)); memset(g->ph_y, 0, sizeof(g->ph_y));
            g->ph_fb[0] = g->ph_fb[1] = 0; g->tone_z[0] = g->tone_z[1] = 0;
            g->drv_lp[0] = g->drv_lp[1] = g->drv_hp[0] = g->drv_hp[1] = g->drv_dc[0][0] = g->drv_dc[1][0] = 0;
            memset(g->eq, 0, sizeof(g->eq)); g->eq_set_v[0] = -999;
            g->dcur = -1; g->nan_resets++;
            for (int i = 0; i < n; i++) { L[i] = R[i] = 0; }
        }
    }
    float og0 = g->out_s, og1 = p[P_OUTGAIN];
    g->out_s = og1;
    float opk0 = 0, opk1 = 0;
    for (int i = 0; i < n; i++) {
        float on = g->ON[i], wg = trails ? 1.0f : on;
        float og = og0 + (og1 - og0) * (i + 1) / n;
        float ol = softclip((L[i] * wg + g->DL[i] * (1.0f - on)) * og, 0.9f);
        float orr = softclip((R[i] * wg + g->DR[i] * (1.0f - on)) * og, 0.9f);
        L[i] = ol; R[i] = orr;
        float al = fabsf(ol), ar = fabsf(orr);
        if (al > opk0) opk0 = al;
        if (ar > opk1) opk1 = ar;
    }
    g->out_pk[0] = fmaxf(g->out_pk[0] * 0.9f, opk0);
    g->out_pk[1] = fmaxf(g->out_pk[1] * 0.9f, opk1);
    for (int i = 0; i < n; i++) {
        unsigned k = ((g->sc_w + (unsigned)i) & SMASK) * 2;
        g->sc_out[k] = L[i]; g->sc_out[k + 1] = R[i];
    }
    g->sc_w += (unsigned)n;
}

/* Process a block straight from given input (desktop and tests: no ring, no resampler).
 * in/out: n*2 interleaved floats, n <= MAXBLOCK. */
API void eng_process(Eng *g, const float *in, float *out, int n) {
    if (n > MAXBLOCK) n = MAXBLOCK;
    pthread_mutex_lock(&g->lock);
    for (int i = 0; i < n; i++) { g->L[i] = in[2 * i]; g->R[i] = in[2 * i + 1]; }
    fx_block(g, g->L, g->R, n);
    for (int i = 0; i < n; i++) { out[2 * i] = g->L[i]; out[2 * i + 1] = g->R[i]; }
    g->blocks++;
    pthread_mutex_unlock(&g->lock);
}

/* ---------------------------------------------------------------- input ring + drift resampler */
/* Capture side: append n interleaved stereo frames. */
static void ring_push(Eng *g, const float *x, int n) {
    unsigned long long w = g->wpos;
    for (int i = 0; i < n; i++) {
        unsigned k = (unsigned)((w + i) & RBM) * 2;
        g->rb[k] = x[2 * i]; g->rb[k + 1] = x[2 * i + 1];
    }
    __sync_synchronize();
    g->wpos = w + n;
}

API void eng_push_input(Eng *g, const float *x, int n) { ring_push(g, x, n); }   /* tests */

/* Writer side: n output frames into L/R from the ring. Returns 1 if real input was read. */
static int ring_pull(Eng *g, float *L, float *R, int n) {
    unsigned long long w = g->wpos;
    __sync_synchronize();
    double avail = (double)w - g->rpos;
    int tgt = g->target;
    if (g->priming) {
        if (avail >= tgt + n) {
            g->rpos = (double)w - tgt;
            g->priming = 0;
            g->fill_avg = tgt;
            avail = tgt;
        } else {
            memset(L, 0, n * sizeof(float)); memset(R, 0, n * sizeof(float));
            return 0;
        }
    }
    if (avail > RB - 2 * MAXBLOCK) {                 /* far too full (input came back after a gap) */
        g->rpos = (double)w - tgt; avail = tgt; g->fill_avg = tgt; g->rs_over++;
        g->integ += 0.0005 / (1.0e-5 / 3.0);
    }
    double last = g->rpos + (n - 1) * g->ratio;
    if ((double)((long long)floor(last) + 2) >= (double)w || g->rpos < 1) {   /* would run dry */
        g->priming = 1; g->rs_under++;
        g->integ -= 0.0005 / (1.0e-5 / 3.0);         /* ran dry: the input is slower, read slower (-500 ppm) */
        memset(L, 0, n * sizeof(float)); memset(R, 0, n * sizeof(float));
        return 0;
    }
    const float *rb = g->rb;
    double pos = g->rpos, r = g->ratio;
    for (int k = 0; k < n; k++) {
        double x = pos + k * r;
        long long i0 = (long long)floor(x);
        float t = (float)(x - (double)i0);
        unsigned a = (unsigned)((i0 - 1) & RBM) * 2, b = (unsigned)(i0 & RBM) * 2;
        unsigned c = (unsigned)((i0 + 1) & RBM) * 2, d = (unsigned)((i0 + 2) & RBM) * 2;
        L[k] = herm(rb[a], rb[b], rb[c], rb[d], t);
        R[k] = herm(rb[a + 1], rb[b + 1], rb[c + 1], rb[d + 1], t);
    }
    g->rpos = pos + n * r;
    /* PI controller on the (smoothed) fill level */
    g->fill_avg += (avail - g->fill_avg) * 0.02;
    double err = g->fill_avg - tgt;
    double dt = (double)n / g->sr;
    g->integ += err * dt;
    const double kp = 1.0e-5, ki = kp / 3.0;
    /* Wide limits: a USB synth against the Brick measured +2250 ppm (0.23%), far more than usual crystals */
    if (g->integ * ki > 0.01) g->integ = 0.01 / ki;
    if (g->integ * ki < -0.01) g->integ = -0.01 / ki;
    double ratio = 1.0 + kp * err + ki * g->integ;
    g->ratio = ratio < 0.988 ? 0.988 : ratio > 1.012 ? 1.012 : ratio;
    return 1;
}

/* Tests: pull + FX one block, as the writer thread does. out: n*2 interleaved. */
API int eng_pull(Eng *g, float *out, int n) {
    if (n > MAXBLOCK) n = MAXBLOCK;
    pthread_mutex_lock(&g->lock);
    int got = ring_pull(g, g->L, g->R, n);
    fx_block(g, g->L, g->R, n);
    for (int i = 0; i < n; i++) { out[2 * i] = g->L[i]; out[2 * i + 1] = g->R[i]; }
    pthread_mutex_unlock(&g->lock);
    return got;
}
API void eng_set_target(Eng *g, int frames) { g->target = frames; }

/* Status: see STATUS_KEYS in brickfx.py */
API int eng_status(Eng *g, float *o, int max) {
    float s[33] = {
        g->in_pk[0], g->in_pk[1], g->out_pk[0], g->out_pk[1], g->load,
        (float)g->underruns, (float)g->cap_xruns, (float)g->rs_under, (float)g->fill_avg,
        (float)((g->ratio - 1.0) * 1e6), (float)g->cap_open, (float)g->play_open, (float)g->rt_play,
        (float)g->cap_period, (float)g->play_buffer, g->dcur / g->sr, (float)g->blocks, (float)g->cap_rate,
        (float)g->target, (float)g->priming, (float)g->cap_channels, (float)g->rs_over, (float)g->play_period,
        (float)g->rt_cap, (float)g->cap_frames, (float)g->cap_reads, g->cap_raw_pk, (float)g->cap_buffer,
        (float)g->cap_opens, (float)g->wpos, (float)g->nan_resets, (float)g->xf[2].spill, (float)g->xf[3].spill };
    int n = max < 33 ? max : 33;
    memcpy(o, s, n * sizeof(float));
    return n;
}

/* ---------------------------------------------------------------- ALSA (libasound loaded at run time) */
typedef int (*f_open)(void **, const char *, int, int);
typedef int (*f_setp)(void *, int, int, unsigned, unsigned, int, unsigned);
typedef int (*f_getp)(void *, unsigned long *, unsigned long *);
typedef long (*f_writei)(void *, const void *, unsigned long);
typedef long (*f_readi)(void *, void *, unsigned long);
typedef int (*f_recover)(void *, int, int);
typedef int (*f_pcm)(void *);
typedef const char *(*f_strerror)(int);
static struct { f_open open; f_setp setp; f_getp getp; f_writei writei; f_readi readi; f_recover recover;
                f_pcm drop, close, start; f_strerror strerror; } A;

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
    A.recover = (f_recover)dlsym(h, "snd_pcm_recover");
    A.drop = (f_pcm)dlsym(h, "snd_pcm_drop");
    A.close = (f_pcm)dlsym(h, "snd_pcm_close");
    A.start = (f_pcm)dlsym(h, "snd_pcm_start");
    A.strerror = (f_strerror)dlsym(h, "snd_strerror");
    done = (A.open && A.setp && A.getp && A.writei && A.readi && A.recover && A.drop && A.close && A.start
            && A.strerror) ? 1 : -1;
    return done > 0;
}

static double cpu_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int set_rt(int prio) {
    struct sched_param sp;
    sp.sched_priority = prio;
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
}

/* Ring fill target: enough to ride out one capture period plus one output block, with margin. */
static void update_target(Eng *g) {
    int per = g->cap_period > 0 ? g->cap_period : 256;
    int margin = g->want_frames <= 512 ? 64 : g->want_frames <= 1024 ? 192 : 512;
    g->target = per + g->block + margin;
}

/* ---- capture thread: USB card -> ring */
static void c_close(Eng *g) {
    if (g->ch) { A.drop(g->ch); A.close(g->ch); g->ch = NULL; }
    g->cap_open = 0;
}

static int c_open(Eng *g) {
    char list[512];
    strncpy(list, g->cap_devs, sizeof(list) - 1); list[sizeof(list) - 1] = 0;
    char *save = NULL;
    for (char *dev = strtok_r(list, "\n", &save); dev; dev = strtok_r(NULL, "\n", &save)) {
        /* the device's own format and rate first, then S16_LE at that rate, then S16_LE at the app's rate */
        int tries[3][2] = {{g->cap_fmt_want, g->cap_rate_want}, {2, g->cap_rate_want}, {2, g->sr}};
        int opened = 0, gone = 0;
        for (int t = 0; t < 3 && !opened && !gone; t++) {
            if (t > 0 && tries[t][0] == tries[t - 1][0] && tries[t][1] == tries[t - 1][1]) continue;
            if (t == 2 && tries[0][0] == 2 && tries[0][1] == g->sr) continue;
            int fmt = tries[t][0], rate = tries[t][1];
            for (int chn = 2; chn >= 1; chn--) {
                void *pcm = NULL;
                int err = A.open(&pcm, dev, 1, 0);                     /* capture, blocking */
                if (err < 0) { snprintf(g->cap_err, sizeof(g->cap_err), "%s: %s", dev, A.strerror(err)); gone = 1; break; }
                /* period ~6 ms at the device rate */
                err = A.setp(pcm, fmt, 3, (unsigned)chn, (unsigned)rate, 1, 6000);
                if (err < 0) {
                    snprintf(g->cap_err, sizeof(g->cap_err), "%s fmt %d %dch %d Hz: %s", dev, fmt, chn, rate, A.strerror(err));
                    A.close(pcm);
                    continue;
                }
                unsigned long buf = 0, per = 0;
                A.getp(pcm, &buf, &per);
                if (per == 0 || per > MAXBLOCK / 2) per = 128;
                g->ch = pcm; g->cap_period = (int)per; g->cap_channels = chn; g->cap_rate = rate; g->cap_buffer = (int)buf;
                g->cap_fmt = fmt; g->rs_pos = 0; g->rs_have = 0;
                opened = 1;
                break;
            }
        }
        if (opened) {
            g->cap_opens++;
            strncpy(g->cap_dev, dev, sizeof(g->cap_dev) - 1);
            g->cap_err[0] = 0;
            A.start(g->ch);
            return 1;
        }
    }
    return 0;
}

/* raw sample i (interleaved index) of the read buffer -> float */
static inline float cap_sample(const Eng *g, long i) {
    const unsigned char *b = g->cbuf;
    switch (g->cap_fmt) {
    case 32: {                                                      /* S24_3LE */
        const unsigned char *q = b + 3 * i;
        int v = q[0] | (q[1] << 8) | (q[2] << 16);
        if (v & 0x800000) v -= 0x1000000;
        return (float)v / 8388608.0f;
    }
    case 6: {                                                       /* S24_LE (in 32 bits) */
        int v; memcpy(&v, b + 4 * i, 4); v &= 0xFFFFFF;
        if (v & 0x800000) v -= 0x1000000;
        return (float)v / 8388608.0f;
    }
    case 10: { int v; memcpy(&v, b + 4 * i, 4); return (float)v / 2147483648.0f; }   /* S32_LE */
    default: { short v; memcpy(&v, b + 2 * i, 2); return (float)v / 32768.0f; }      /* S16_LE */
    }
}

static void *capturer(void *arg) {
    Eng *g = (Eng *)arg;
    g->rt_cap = set_rt(52);
    float fbuf[MAXBLOCK * 2], obuf[MAXBLOCK * 4];
    while (g->run) {
        if (g->cap_change) { c_close(g); g->cap_change = 0; }
        if (g->cap_switch) {                                  /* the writer has faded the old input out */
            pthread_mutex_lock(&g->lock);
            memcpy(g->cap_devs, g->cap_pending, sizeof(g->cap_devs));
            g->cap_fmt_want = g->pend_fmt; g->cap_rate_want = g->pend_rate;
            g->cap_switch = 0;
            pthread_mutex_unlock(&g->lock);
            c_close(g);
        }
        if (!g->cap_devs[0]) { usleep(100000); continue; }
        if (!g->ch) {
            if (!c_open(g)) { usleep(1000000); continue; }
            g->cap_open = 1;
            pthread_mutex_lock(&g->lock);
            update_target(g);
            g->priming = 1;
            pthread_mutex_unlock(&g->lock);
        }
        int per = g->cap_period, chn = g->cap_channels;
        long r = A.readi(g->ch, g->cbuf, (unsigned long)per);
        if (r < 0) {
            g->cap_xruns++;
            if (A.recover(g->ch, (int)r, 1) < 0) {
                snprintf(g->cap_err, sizeof(g->cap_err), "read failed: %s", A.strerror((int)r));
                c_close(g);                                           /* unplugged: retry every second */
                usleep(500000);
            } else A.start(g->ch);
            continue;
        }
        g->cap_reads += 1;
        if (r == 0) continue;
        float pk = 0, pl = g->lvl_pk[0], pr = g->lvl_pk[1];
        long nz = 0;
        for (long i = 0; i < r; i++) {
            if (chn == 2) { fbuf[2 * i] = cap_sample(g, 2 * i); fbuf[2 * i + 1] = cap_sample(g, 2 * i + 1); }
            else { fbuf[2 * i] = fbuf[2 * i + 1] = cap_sample(g, i); }
            float a = fabsf(fbuf[2 * i]), b = fabsf(fbuf[2 * i + 1]);
            pk = fmaxf(pk, fmaxf(a, b)); pl = fmaxf(pl, a); pr = fmaxf(pr, b);
            nz += (a > 0) + (b > 0);
        }
        g->lvl_pk[0] = pl; g->lvl_pk[1] = pr; g->lvl_nz += nz;
        g->cap_raw_pk = fmaxf(g->cap_raw_pk * 0.97f, pk);     /* raw USB level, before Input gain */
        g->cap_frames += (double)r;
        if (g->cap_rate == g->sr) ring_push(g, fbuf, (int)r);
        else {                                                /* device can't do the app's rate: linear conversion */
            double step = (double)g->cap_rate / g->sr, pos = g->rs_pos;
            int m = 0;
            if (!g->rs_have) { g->rs_prev[0] = fbuf[0]; g->rs_prev[1] = fbuf[1]; g->rs_have = 1; }
            /* sample k of this block is index k+1, index 0 = last sample of the previous block */
            while (pos < (double)r && m < MAXBLOCK * 2) {
                long i0 = (long)pos; float t = (float)(pos - i0);
                float a0 = i0 == 0 ? g->rs_prev[0] : fbuf[2 * (i0 - 1)], b0 = i0 == 0 ? g->rs_prev[1] : fbuf[2 * (i0 - 1) + 1];
                float a1 = fbuf[2 * i0], b1 = fbuf[2 * i0 + 1];
                obuf[2 * m] = a0 + (a1 - a0) * t; obuf[2 * m + 1] = b0 + (b1 - b0) * t;
                m++; pos += step;
            }
            g->rs_pos = pos - (double)r;
            g->rs_prev[0] = fbuf[2 * (r - 1)]; g->rs_prev[1] = fbuf[2 * (r - 1) + 1];
            if (m) ring_push(g, obuf, m);
        }
    }
    c_close(g);
    return NULL;
}

/* ---- writer thread: ring -> FX -> Brick output */
static void p_close(Eng *g) {
    if (g->ph) { A.drop(g->ph); A.close(g->ph); g->ph = NULL; }
    g->play_open = 0;
}

static int p_open(Eng *g) {
    char list[256];
    strncpy(list, g->play_devs, sizeof(list) - 1); list[sizeof(list) - 1] = 0;
    unsigned us = (unsigned)((double)g->want_frames * 1e6 / g->sr);
    char *save = NULL;
    for (char *dev = strtok_r(list, "\n", &save); dev; dev = strtok_r(NULL, "\n", &save)) {
        void *pcm = NULL;
        int err = A.open(&pcm, dev, 0, 0);
        if (err < 0) { snprintf(g->err, sizeof(g->err), "%s: %s", dev, A.strerror(err)); continue; }
        err = A.setp(pcm, 2, 3, 2, (unsigned)g->sr, 1, us);
        if (err < 0) { snprintf(g->err, sizeof(g->err), "%s set_params: %s", dev, A.strerror(err)); A.close(pcm); continue; }
        unsigned long buf = 0, per = 0;
        A.getp(pcm, &buf, &per);
        g->ph = pcm; g->play_buffer = (int)buf; g->play_period = (int)per;
        g->opened_frames = g->want_frames;
        strncpy(g->play_dev, dev, sizeof(g->play_dev) - 1);
        g->err[0] = 0;
        g->play_open = 1;
        return 1;
    }
    return 0;
}

static void *writer(void *arg) {
    Eng *g = (Eng *)arg;
    g->rt_play = set_rt(50);
    int n = g->block;
    double period = (double)n / g->sr;
    while (g->run) {
        if (g->ph && g->want_frames != g->opened_frames) {
            p_close(g);
            pthread_mutex_lock(&g->lock); update_target(g); g->priming = 1; pthread_mutex_unlock(&g->lock);
        }
        if (!g->ph && !p_open(g)) {
            if (g->sw_req) { g->sw_req = 0; g->cap_switch = 1; }   /* no output: nothing to fade */
            usleep(500000); continue;
        }
        double c0 = cpu_now();
        pthread_mutex_lock(&g->lock);
        int got = ring_pull(g, g->L, g->R, n);
        /* Input fade (20 ms): in after every (re)start of the input, out before switching device.
         * Applied before the FX, so delay and reverb tails ring on through a switch. */
        if (!got) { g->in_g = 0; g->in_hold = (int)(0.015f * (float)g->sr); }   /* settle 15 ms, then fade in */
        else if (g->in_hold > 0) { g->in_hold -= n; g->in_g = 0; }
        float tg = (g->sw_req || g->in_hold > 0) ? 0.0f : 1.0f, st = 1.0f / (0.02f * (float)g->sr);
        if (got || g->in_g != tg)
            for (int i = 0; i < n; i++) {
                g->in_g += clampf(tg - g->in_g, -st, st);
                g->L[i] *= g->in_g; g->R[i] *= g->in_g;
            }
        if (g->sw_req && g->in_g <= 0.0f) { g->sw_req = 0; g->cap_switch = 1; }
        fx_block(g, g->L, g->R, n);
        g->blocks++;
        pthread_mutex_unlock(&g->lock);
        float pk = 0;
        for (int i = 0; i < n; i++) {
            float l = clampf(g->L[i], -1, 1), r = clampf(g->R[i], -1, 1);
            g->pcm[2 * i] = (short)lrintf(l * 32767.0f); g->pcm[2 * i + 1] = (short)lrintf(r * 32767.0f);
            pk = fmaxf(pk, fmaxf(fabsf(l), fabsf(r)));
        }
        if (pk < 1.0f / 32767.0f)                       /* the Brick's codec pops on true digital silence */
            for (int i = 0; i < n; i++) { short d = (i & 1) ? -1 : 1; g->pcm[2 * i] = d; g->pcm[2 * i + 1] = d; }
        g->load = g->load * 0.95f + 0.05f * (float)((cpu_now() - c0) / period);
        int done = 0;
        while (done < n && g->ph) {
            long wr = A.writei(g->ph, g->pcm + done * 2, (unsigned long)(n - done));
            if (wr < 0) {
                g->underruns++;
                if (A.recover(g->ph, (int)wr, 1) < 0) {
                    snprintf(g->err, sizeof(g->err), "write failed: %s", A.strerror((int)wr));
                    p_close(g);
                }
                continue;
            }
            done += (int)wr;
        }
    }
    p_close(g);
    return NULL;
}

/* play_devs / cap_devs: newline-separated ALSA names to try in order (cap_devs may be empty). */
API int eng_start(Eng *g, const char *play_devs, const char *cap_devs, int frames) {
    if (!alsa_load()) return 0;
    strncpy(g->play_devs, play_devs, sizeof(g->play_devs) - 1);
    strncpy(g->cap_devs, cap_devs ? cap_devs : "", sizeof(g->cap_devs) - 1);
    g->want_frames = frames;
    update_target(g);
    g->run = 1;
    if (pthread_create(&g->pth, NULL, writer, g) != 0) { g->run = 0; return 0; }
    pthread_create(&g->cth, NULL, capturer, g);
    return 1;
}
API void eng_set_capture(Eng *g, const char *cap_devs) {
    pthread_mutex_lock(&g->lock);
    memset(g->cap_pending, 0, sizeof(g->cap_pending));
    strncpy(g->cap_pending, cap_devs ? cap_devs : "", sizeof(g->cap_pending) - 1);
    if (g->run) g->sw_req = 1;                              /* writer fades out, then the capture thread swaps */
    else {
        memcpy(g->cap_devs, g->cap_pending, sizeof(g->cap_devs));
        g->cap_fmt_want = g->pend_fmt; g->cap_rate_want = g->pend_rate;
        g->cap_change = 1;
    }
    pthread_mutex_unlock(&g->lock);
}
/* Format (ALSA number: 2 S16_LE, 6 S24_LE, 10 S32_LE, 32 S24_3LE) and rate for the next eng_set_capture / start */
API void eng_set_cap_format(Eng *g, int fmt, int rate) {
    if (fmt != 2 && fmt != 6 && fmt != 10 && fmt != 32) fmt = 2;
    if (rate < 8000 || rate > 192000) rate = g->sr;
    g->pend_fmt = fmt; g->pend_rate = rate;
    if (!g->run) { g->cap_fmt_want = fmt; g->cap_rate_want = rate; }
}
API int eng_cap_format(Eng *g) { return g->cap_open ? g->cap_fmt : 0; }
/* per-channel peak and non-zero sample count since the last call (then reset) */
API void eng_cap_level(Eng *g, float *o) {
    o[0] = g->lvl_pk[0]; o[1] = g->lvl_pk[1]; o[2] = (float)g->lvl_nz;
    g->lvl_pk[0] = g->lvl_pk[1] = 0; g->lvl_nz = 0;
}
API void eng_set_buffer(Eng *g, int frames) { g->want_frames = frames; }
API const char *eng_device(Eng *g) { return g->play_dev; }
API const char *eng_cap_device(Eng *g) { return g->cap_dev; }
API const char *eng_error(Eng *g) { return g->err; }
API const char *eng_cap_error(Eng *g) { return g->cap_err; }
API int bfx_version(void) { return 8; }
