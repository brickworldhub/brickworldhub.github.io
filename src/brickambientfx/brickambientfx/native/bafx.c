/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Alessandro Pietrosanti (BrickWorld, brickworldhub.com)
 *
 * bafx - native audio engine for BrickAmbientFX (Sarta Edition), an ambient stereo multi-effect for the TrimUI Brick.
 * Built on the BrickFX 0.2.2 engine (bfx.c): same USB capture, drift resampler and ALSA writer.
 *
 *   USB audio in (capture thread) -> ring buffer -> adaptive resampler -> FX chain -> ALSA out (writer thread)
 *
 *   FX chain:  COMP -> AMP (+CAB) -> MOD -> DELAY -> REVERB            (routing SERIAL)
 *              COMP -> AMP (+CAB) -> MOD -+-> DELAY  -+-> out         (routing PARALLEL: split into two paths,
 *                                         +-> REVERB -+                 balance and stereo spread between them)
 *              then output width (mid/side) and FREEZE on delay + reverb.
 *
 * Python only sends parameters and reads a status array; the audio threads never touch Python.
 * Build: see build.sh (zig cc cross-compiles for the Brick and Linux x86_64).
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
#define DSIZE (1 << 18)            /* delay line: 262144 frames (5.4 s at 48 kHz) */
#define DMASK (DSIZE - 1)
#define CSIZE 4096                 /* chorus / flanger / vibrato line: 85 ms */
#define CMASK (CSIZE - 1)
#define GSIZE (1 << 17)            /* glitch buffer: 2.7 s */
#define GMASK (GSIZE - 1)
#define PSIZE 16384                /* reverb pre-delay: 0.34 s */
#define PSSIZE 8192                /* pitch shifter line */
#define PSMASK (PSSIZE - 1)
#define MAXBLOCK 1024
#define TWO_PI 6.283185307179586
#define PI_F 3.14159265f

/* Parameter order: ENGINE_KEYS in brickambientfx.py */
enum { P_ON, P_INGAIN, P_OUTGAIN, P_ROUTE, P_TRAILS, P_FREEZE, P_ROUTING, P_BALANCE, P_SPREAD, P_WIDTH_OUT,
       P_CMP_TYPE, P_CMP_THR, P_CMP_RATIO, P_CMP_ATK, P_CMP_REL, P_CMP_MAKEUP, P_CMP_MIX, P_CMP_SWELL,
       P_AMP_TYPE, P_AMP_GAIN, P_AMP_BASS, P_AMP_MID, P_AMP_TREBLE, P_AMP_CAB, P_AMP_LEVEL,
       P_MOD_TYPE, P_MOD_RATE, P_MOD_DEPTH, P_MOD_COLOR, P_MOD_MIX,
       P_DLY_TYPE, P_DLY_TIME, P_DLY_FB, P_DLY_MIX, P_DLY_MODE, P_DLY_WIDTH, P_DLY_HICUT, P_DLY_LOCUT,
       P_DLY_MODDEPTH, P_DLY_MODRATE, P_DLY_COLOR,
       P_REV_TYPE, P_REV_DECAY, P_REV_PRE, P_REV_DAMP, P_REV_LOCUT, P_REV_MOD, P_REV_COLOR, P_REV_MIX,
       P_COUNT };
enum { MODE_STEREO, MODE_PINGPONG, MODE_MONO };
enum { ROUTE_STEREO, ROUTE_L, ROUTE_R, ROUTE_SUM };
enum { ROUTING_SERIAL, ROUTING_PARALLEL };
enum { CMP_OFF, CMP_STUDIO, CMP_OPTO, CMP_LIMIT, CMP_SWELL };
enum { AMP_OFF, AMP_CLEAN, AMP_CRUNCH, AMP_LEAD, AMP_FUZZ, AMP_LOFI };
enum { CAB_OFF, CAB_1X12, CAB_2X12, CAB_4X12, CAB_RADIO };
enum { MOD_OFF, MOD_CHORUS, MOD_ENSEMBLE, MOD_FLANGER, MOD_PHASER, MOD_TREMOLO, MOD_PAN, MOD_VIBRATO, MOD_GLITCH };
enum { DLY_OFF, DLY_DIGITAL, DLY_TAPE, DLY_ANALOG, DLY_DUAL, DLY_REVERSE, DLY_SHIMMER };
enum { REV_OFF, REV_ROOM, REV_PLATE, REV_HALL, REV_CLOUD, REV_SHIMMER, REV_BLOOM };
enum { GL_NONE, GL_REPEAT, GL_REVERSE, GL_OCTAVE, GL_STOP, GL_DROP };

enum { BQ_LP, BQ_HP, BQ_PEAK, BQ_LSHELF, BQ_HSHELF };
typedef struct { float b0, b1, b2, a1, a2, z1[2], z2[2]; float key[4]; int on; } Biquad;

/* delay line with its own length; all reverb lines share one write counter */
typedef struct { float *buf; unsigned mask; int len; } Line;

/* section crossfade: `cur` is the key now running (type, or type*8 + mode / cab), pos 0..1 its fade */
typedef struct { int cur; float pos; } XF;
/* delay / reverb: crossfade for type changes, plus an input gate for spillover when switched off */
typedef struct { XF xf; float pin, tg; int quiet, tail_n; } Spill;

/* granular pitch shifter (two crossfaded heads), mono */
typedef struct { float buf[PSSIZE]; unsigned w; float d; } Pitch;

#define NTANK 8
typedef struct {
    Line pre, in[4], bloom[4], tank[NTANK];
    unsigned w;
    float bw, lp[2], hp[2], lastL, lastR, precur, shim_hp, shim_lp;
    float tank_pk; int tank_clip;
    float tcomp;                   /* smoothed input gain into the tank (longer decay -> less in) */
    double lph;
    float size;
    int type;
    int taps[14];
    Pitch ps;
} Reverb;

typedef struct {
    int sr, block;
    float p[P_COUNT];
    float on_s;                     /* smoothed chain on/off (0 bypass .. 1 on) */
    float in_s, out_s;              /* smoothed input / output gains */
    int cleared;
    float FRZ[MAXBLOCK], frz_s;     /* smoothed freeze, per sample */
    /* routing */
    float pdry_s, bal_a_s, bal_b_s, spread_s, width_s;
    float AL[MAXBLOCK], AR[MAXBLOCK], BL[MAXBLOCK], BR[MAXBLOCK];
    /* compressor */
    float cmp_env, cmp_gr, cmp_rms, cmp_swell_g, cmp_mix_s, cmp_gr_show;
    int cmp_armed, cmp_below, cmp_prev;
    /* amp */
    Biquad amp_pre, amp_bass, amp_mid, amp_treb, cab[4];
    float amp_hp[2], amp_lp1[2], amp_hp2[2], amp_dc[2], amp_hold[2];
    int amp_cnt;
    /* mod */
    float *cl, *cr;
    unsigned cw;
    double mod_ph, mod_ph2;
    float ph_x[2][6], ph_y[2][6], ph_fb[2], ph_a[2];
    float mod_mix_s, mod_depth_s, fl_fb[2], drift, drift_t;
    float ch_d[2][3], ch_inc[2][3];   /* chorus voice delays (samples) and their per-sample glide */
    int drift_cnt, mod_prev;
    /* glitch */
    float *gl, *gr;
    unsigned gw;
    int gcnt, gS, gev, gsub, gstart;
    float grp, genv;
    unsigned rnd;
    /* delay */
    float *dl, *dr;
    unsigned dw;
    float dcur[2];                  /* current delay (samples), glides toward the target: tape-like, no clicks */
    double mph, flut, wow;          /* delay modulation LFO, tape flutter and wow */
    float mdepth;
    float dry_s, wet_s;
    float tone_z[2];
    float rc;                       /* reverse: phase in the window */
    Line dif[2][3];
    unsigned difw;
    Pitch dps[2];
    int dly_prev;
    Biquad lp, hp;
    /* switching: each section fades out, changes over while silent, fades back in (S-curve).
     * Delay and reverb switched off only close their input: echoes and tail ring out (spillover). */
    XF xf_cmp, xf_amp, xf_mod, xf_route;
    Spill sp_dly, sp_rev;
    float XG[MAXBLOCK], XI[MAXBLOCK], TL[MAXBLOCK], TR[MAXBLOCK];
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
    float L[MAXBLOCK], R[MAXBLOCK], DL[MAXBLOCK], DR[MAXBLOCK], ON[MAXBLOCK];
    short pcm[MAXBLOCK * 2];
    /* status */
    float in_pk[2], out_pk[2], load;
    int underruns, cap_xruns, rs_under, rs_over, blocks, nan_resets, glitch_count;
    pthread_mutex_t lock;
    /* threads */
    pthread_t pth, cth;
    volatile int run, cap_run, play_open, cap_open, rt_play, rt_cap, want_frames, opened_frames;
    volatile int cap_change, cap_period, cap_channels, play_buffer, play_period, cap_rate, cap_buffer, cap_opens;
    volatile double cap_frames, cap_reads;
    volatile float cap_raw_pk;
    void *ph, *ch;
    char play_devs[256], cap_devs[512], play_dev[96], cap_dev[96], err[160], cap_err[160];
    short cbuf[MAXBLOCK * 8];
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
/* fractional read d samples behind write position w on a power-of-two buffer */
static inline float rd_herm(const float *b, unsigned mask, unsigned w, float d) {
    float rp = (float)w - d;
    int i0 = (int)floorf(rp);
    float f = rp - i0;
    return herm(b[(i0 - 1) & mask], b[i0 & mask], b[(i0 + 1) & mask], b[(i0 + 2) & mask], f);
}

/* one-pole low-pass coefficient for a cutoff (y += c * (x - y)) */
static inline float op_coef(float fc, float sr) { return 1.0f - expf(-6.2831853f * fc / sr); }

static inline unsigned xrand(unsigned *s) { unsigned x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return *s = x; }
static inline float rnd01(unsigned *s) { return (xrand(s) >> 8) * (1.0f / 16777216.0f); }

/* ---------------------------------------------------------------- biquads (RBJ cookbook) */
static void bq_design(Biquad *b, int type, float fc, float q, float db, float sr) {
    if (b->key[0] == type && b->key[1] == fc && b->key[2] == q && b->key[3] == db) return;
    b->key[0] = (float)type; b->key[1] = fc; b->key[2] = q; b->key[3] = db;
    fc = clampf(fc, 10.0f, sr * 0.45f);
    float w0 = 2.0f * PI_F * fc / sr, c = cosf(w0), s = sinf(w0), al = s / (2 * q);
    float A = powf(10.0f, db / 40.0f), b0, b1, b2, a0, a1, a2;
    switch (type) {
    case BQ_HP: b0 = (1 + c) / 2; b1 = -(1 + c); b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; break;
    case BQ_PEAK: b0 = 1 + al * A; b1 = -2 * c; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * c; a2 = 1 - al / A; break;
    case BQ_LSHELF: {
        float sq = 2 * sqrtf(A) * al;
        b0 = A * ((A + 1) - (A - 1) * c + sq); b1 = 2 * A * ((A - 1) - (A + 1) * c); b2 = A * ((A + 1) - (A - 1) * c - sq);
        a0 = (A + 1) + (A - 1) * c + sq; a1 = -2 * ((A - 1) + (A + 1) * c); a2 = (A + 1) + (A - 1) * c - sq; break; }
    case BQ_HSHELF: {
        float sq = 2 * sqrtf(A) * al;
        b0 = A * ((A + 1) + (A - 1) * c + sq); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - sq);
        a0 = (A + 1) - (A - 1) * c + sq; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - sq; break; }
    default: b0 = (1 - c) / 2; b1 = 1 - c; b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; break;
    }
    b->b0 = b0 / a0; b->b1 = b1 / a0; b->b2 = b2 / a0; b->a1 = a1 / a0; b->a2 = a2 / a0;
}
static inline float bq_run(Biquad *b, int c, float x) {
    float y = b->b0 * x + b->z1[c];
    b->z1[c] = b->b1 * x - b->a1 * y + b->z2[c];
    b->z2[c] = b->b2 * x - b->a2 * y;
    return y;
}
static void bq_reset(Biquad *b) { memset(b->z1, 0, sizeof(b->z1)); memset(b->z2, 0, sizeof(b->z2)); }

/* ---------------------------------------------------------------- lines and pitch shifter */
static void line_alloc(Line *l, int maxlen) {
    unsigned cap = 64;
    while (cap < (unsigned)maxlen + 64) cap <<= 1;
    l->buf = (float *)calloc(cap, sizeof(float));
    l->mask = cap - 1;
    l->len = maxlen;
}
static void line_clear(Line *l) { memset(l->buf, 0, (l->mask + 1) * sizeof(float)); }
static inline float line_rd(const Line *l, unsigned w, int d) { return l->buf[(w - (unsigned)d) & l->mask]; }
static inline void line_wr(Line *l, unsigned w, float x) { l->buf[w & l->mask] = x; }

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

/* Pitch shift by `ratio` (2 = octave up): two read heads sweep a window, crossfaded so the jumps are silent. */
static inline float ps_run(Pitch *s, float x, float ratio, float win) {
    s->buf[s->w & PSMASK] = x;
    s->d -= ratio - 1.0f;
    if (s->d < 0) s->d += win;
    if (s->d >= win) s->d -= win;
    float d2 = s->d + 0.5f * win;
    if (d2 >= win) d2 -= win;
    float y1, y2;
    { float rp = (float)s->w - s->d - 1.0f; int i = (int)floorf(rp); float f = rp - i;
      float a = s->buf[i & PSMASK], b = s->buf[(i + 1) & PSMASK]; y1 = a + (b - a) * f; }
    { float rp = (float)s->w - d2 - 1.0f; int i = (int)floorf(rp); float f = rp - i;
      float a = s->buf[i & PSMASK], b = s->buf[(i + 1) & PSMASK]; y2 = a + (b - a) * f; }
    float ph = s->d / win, g = 4.0f * ph * (1.0f - ph);   /* ~sin(pi ph), squared: smooth crossfade */
    g *= g;
    s->w++;
    return y1 * g + y2 * (1.0f - g);
}

/* ---------------------------------------------------------------- compressor / swell */
static void comp_block(Eng *g, float *L, float *R, int n, int type) {
    const float *p = g->p;
    float sr = (float)g->sr;
    if (type == CMP_OFF) { g->cmp_prev = 0; g->cmp_gr_show *= 0.8f; return; }
    if (g->cmp_prev != type) { g->cmp_gr = 0; g->cmp_env = 0; g->cmp_rms = 0; g->cmp_swell_g = 1; g->cmp_armed = 0; g->cmp_below = 0; }
    g->cmp_prev = type;
    float thr = p[P_CMP_THR], ratio = clampf(p[P_CMP_RATIO], 1.0f, 30.0f);
    float atk_t = clampf(p[P_CMP_ATK], 0.0001f, 0.5f), rel_t = clampf(p[P_CMP_REL], 0.01f, 4.0f);
    float makeup = powf(10.0f, clampf(p[P_CMP_MAKEUP], 0, 30) / 20.0f);
    float mix_t = clampf(p[P_CMP_MIX], 0, 1);
    float knee = 6.0f;
    if (type == CMP_OPTO) knee = 12.0f;
    if (type == CMP_LIMIT) { ratio = 100.0f; knee = 1.0f; atk_t = fminf(atk_t, 0.002f); }
    float atk = 1.0f - expf(-1.0f / (atk_t * sr)), rel = 1.0f - expf(-1.0f / (rel_t * sr));
    float rms_c = op_coef(1.0f / 0.012f, sr);
    float ramp = 1.0f - expf(-1.0f / (0.02f * sr));
    float slope = 1.0f / ratio - 1.0f;
    float env = g->cmp_env, gr = g->cmp_gr, rms = g->cmp_rms, mix = g->cmp_mix_s, sg = g->cmp_swell_g;
    float most = 0;
    if (type == CMP_SWELL) {
        /* volume swell (slow gear): a new note above the threshold starts a fade-in over the swell time;
         * when the signal stays below threshold - 6 dB for the reset time the volume goes down, ready for the next one */
        float swell = clampf(p[P_CMP_SWELL], 0.02f, 6.0f);
        float up = 1.0f / (swell * sr);
        float ea = op_coef(1.0f / 0.001f, sr), er = op_coef(1.0f / 0.08f, sr);
        float thr_lin = powf(10.0f, thr / 20.0f), low_lin = thr_lin * 0.5f;
        int reset = (int)(rel_t * sr);
        float down = 1.0f - expf(-1.0f / (0.03f * sr));
        int armed = g->cmp_armed, below = g->cmp_below;
        for (int i = 0; i < n; i++) {
            mix += (mix_t - mix) * ramp;
            float lv = fmaxf(fabsf(L[i]), fabsf(R[i]));
            env += (lv > env ? ea : er) * (lv - env);
            if (env < low_lin) { if (below < reset) below++; else armed = 1; } else below = 0;
            if (armed) {
                if (env > thr_lin) { armed = 0; gr = 0; }                /* gr = ramp position 0..1 */
                else sg += (0.0f - sg) * down;
            }
            if (!armed && gr < 1.0f) { gr += up; if (gr > 1.0f) gr = 1.0f; sg = gr * gr; }
            float gn = sg * makeup;
            L[i] = L[i] + (L[i] * gn - L[i]) * mix;
            R[i] = R[i] + (R[i] * gn - R[i]) * mix;
        }
        g->cmp_armed = armed; g->cmp_below = below;
        g->cmp_env = env; g->cmp_gr = gr; g->cmp_mix_s = mix; g->cmp_swell_g = sg;
        g->cmp_gr_show = sg > 1e-4f ? 20.0f * log10f(sg) : -60.0f;
        return;
    }
    for (int i = 0; i < n; i++) {
        mix += (mix_t - mix) * ramp;
        float lv = fmaxf(fabsf(L[i]), fabsf(R[i]));
        float det;
        if (type == CMP_OPTO) { rms += rms_c * (lv * lv - rms); det = sqrtf(rms) * 1.41f; }
        else det = lv;
        float db = 20.0f * log10f(det + 1e-9f);
        float over = db - thr, t;
        if (over <= -knee * 0.5f) t = 0;
        else if (over < knee * 0.5f) { float k = over + knee * 0.5f; t = slope * k * k / (2 * knee); }
        else t = slope * over;
        if (t < gr) gr += (t - gr) * atk;
        else {
            float r = rel;
            if (type == CMP_OPTO) r = rel / (1.0f + 0.25f * -gr);     /* opto: the deeper it went, the slower it lets go */
            gr += (t - gr) * r;
        }
        if (gr < most) most = gr;
        float gn = powf(10.0f, gr * 0.05f) * makeup;
        L[i] = L[i] + (L[i] * gn - L[i]) * mix;
        R[i] = R[i] + (R[i] * gn - R[i]) * mix;
    }
    g->cmp_env = env; g->cmp_gr = gr; g->cmp_rms = rms; g->cmp_mix_s = mix;
    g->cmp_gr_show = fminf(most, g->cmp_gr_show * 0.85f);
}

/* ---------------------------------------------------------------- amp + cabinet */
static const float CAB_TAB[5][4][4] = {     /* type, fc, q, dB for the 4 cabinet filters */
    {{0}},
    {{BQ_HP, 90, 0.7f, 0}, {BQ_PEAK, 2200, 1.0f, 3}, {BQ_LP, 5200, 0.9f, 0}, {BQ_LP, 7500, 0.6f, 0}},
    {{BQ_HP, 80, 0.7f, 0}, {BQ_PEAK, 120, 1.2f, 2.5f}, {BQ_PEAK, 2500, 1.0f, 2}, {BQ_LP, 5500, 0.8f, 0}},
    {{BQ_HP, 70, 0.7f, 0}, {BQ_PEAK, 110, 1.3f, 4}, {BQ_PEAK, 400, 1.0f, -3}, {BQ_LP, 4500, 0.9f, 0}},
    {{BQ_HP, 400, 0.7f, 0}, {BQ_PEAK, 1500, 0.8f, 4}, {BQ_LP, 3000, 0.9f, 0}, {BQ_LP, 3500, 0.7f, 0}},
};

static void amp_block(Eng *g, float *L, float *R, int n, int type, int cab) {
    const float *p = g->p;
    if (type == AMP_OFF) return;
    float sr = (float)g->sr;
    float amt = clampf(p[P_AMP_GAIN], 0, 1), lvl = p[P_AMP_LEVEL];
    bq_design(&g->amp_bass, BQ_LSHELF, 150, 0.7f, (clampf(p[P_AMP_BASS], 0, 1) - 0.5f) * 24, sr);
    bq_design(&g->amp_mid, BQ_PEAK, 750, 0.7f, (clampf(p[P_AMP_MID], 0, 1) - 0.5f) * 24, sr);
    bq_design(&g->amp_treb, BQ_HSHELF, 3200, 0.7f, (clampf(p[P_AMP_TREBLE], 0, 1) - 0.5f) * 24, sr);
    float pre_db = type == AMP_CRUNCH ? 3.0f : type == AMP_LEAD ? 6.0f : 0.0f;
    bq_design(&g->amp_pre, BQ_PEAK, type == AMP_LEAD ? 900.0f : 1200.0f, 0.8f, pre_db, sr);
    if (cab) for (int k = 0; k < 4; k++) bq_design(&g->cab[k], (int)CAB_TAB[cab][k][0], CAB_TAB[cab][k][1], CAB_TAB[cab][k][2], CAB_TAB[cab][k][3], sr);
    float *ch[2] = {L, R};
    if (type == AMP_LOFI) {
        float q = powf(2.0f, 15.0f - amt * 11.0f);              /* 16 bit down to 5 bit */
        int hold = 1 + (int)(amt * amt * 15.0f);
        int cnt = g->amp_cnt;
        for (int i = 0; i < n; i++) {
            if (cnt == 0) { g->amp_hold[0] = roundf(L[i] * q) / q; g->amp_hold[1] = roundf(R[i] * q) / q; }
            L[i] = g->amp_hold[0]; R[i] = g->amp_hold[1];
            if (++cnt >= hold) cnt = 0;
        }
        g->amp_cnt = cnt;
    }
    float gain, comp, bias;
    switch (type) {
    case AMP_CLEAN: gain = 1.0f + 7.0f * amt * amt; bias = 0.05f; break;
    case AMP_CRUNCH: gain = 1.0f + 40.0f * amt * amt; bias = 0.15f; break;
    case AMP_LEAD: gain = 2.0f + 150.0f * amt * amt; bias = 0.2f; break;
    case AMP_FUZZ: gain = 1.0f + 80.0f * amt * amt; bias = 0; break;
    default: gain = 1.0f; bias = 0; break;
    }
    /* loudness stays about level as the gain goes up (for a -10 dB signal) */
    comp = type == AMP_FUZZ ? 0.3f / fminf(1.0f, gain * 0.3f) : 0.3f / tanhf(gain * 0.3f);
    if (type == AMP_CRUNCH) comp *= 0.8f;
    if (type == AMP_LEAD) comp *= 0.6f;
    float hpc = op_coef(type == AMP_CLEAN ? 30.0f : type == AMP_CRUNCH ? 70.0f : type == AMP_LEAD ? 100.0f : 150.0f, sr);
    float lp1c = op_coef(type == AMP_LEAD ? 5500.0f : 6500.0f, sr), hp2c = op_coef(200.0f, sr), dcc = op_coef(8.0f, sr);
    float tb = tanhf(bias);
    for (int c = 0; c < 2; c++) {
        float *x = ch[c], hp = g->amp_hp[c], lp1 = g->amp_lp1[c], hp2 = g->amp_hp2[c], dc = g->amp_dc[c];
        for (int i = 0; i < n; i++) {
            float v = x[i], y;
            if (type != AMP_LOFI) {
                hp += hpc * (v - hp);
                v -= hp;
                if (pre_db > 0) v = bq_run(&g->amp_pre, c, v);
                if (type == AMP_FUZZ) {
                    float u = v * gain;
                    y = clampf(u, -1.0f, 1.0f);
                    y = 0.85f * y + 0.15f * tanhf(u * 0.5f);
                } else {
                    y = tanhf(v * gain + bias) - tb;                    /* asymmetric: warmer, even harmonics */
                    if (type == AMP_CRUNCH) { lp1 += lp1c * (y - lp1); y = tanhf(1.8f * lp1); }
                    else if (type == AMP_LEAD) {
                        hp2 += hp2c * (y - hp2); y -= hp2;
                        lp1 += lp1c * (y - lp1); y = tanhf(3.0f * lp1 + 0.1f) - 0.0997f;
                    }
                }
                dc += dcc * (y - dc);
                y = (y - dc) * comp;
            } else y = v;
            y = bq_run(&g->amp_bass, c, y);
            y = bq_run(&g->amp_mid, c, y);
            y = bq_run(&g->amp_treb, c, y);
            if (cab) for (int k = 0; k < 4; k++) y = bq_run(&g->cab[k], c, y);
            x[i] = y * lvl;
        }
        g->amp_hp[c] = hp; g->amp_lp1[c] = lp1; g->amp_hp2[c] = hp2; g->amp_dc[c] = dc;
    }
}

/* ---------------------------------------------------------------- modulation */
static inline float shaped(float s, float shape) {      /* sine -> square */
    if (shape < 0.01f) return s;
    float k = 1.0f + shape * 12.0f;
    return tanhf(k * s) / tanhf(k);
}

static void glitch_block(Eng *g, float *L, float *R, int n);

static void mod_block(Eng *g, float *L, float *R, int n, int type) {
    const float *p = g->p;
    float sr = (float)g->sr;
    if (type != MOD_CHORUS && type != MOD_ENSEMBLE && type != MOD_FLANGER && type != MOD_VIBRATO) {
        unsigned w0 = g->cw;
        for (int i = 0; i < n; i++) { g->cl[w0 & CMASK] = L[i]; g->cr[w0 & CMASK] = R[i]; w0 = (w0 + 1) & CMASK; }
        g->cw = w0;
    }
    if (type == MOD_OFF) { g->mod_prev = 0; g->gev = GL_NONE; return; }
    if (g->mod_prev != type) {
        memset(g->ph_x, 0, sizeof(g->ph_x)); memset(g->ph_y, 0, sizeof(g->ph_y));
        g->ph_fb[0] = g->ph_fb[1] = 0; g->fl_fb[0] = g->fl_fb[1] = 0;
        g->gcnt = 0; g->gev = GL_NONE; g->genv = 0;
        memset(g->ch_d, 0, sizeof(g->ch_d)); memset(g->ch_inc, 0, sizeof(g->ch_inc));
        g->mod_prev = type;
    }
    if (type == MOD_GLITCH) { glitch_block(g, L, R, n); return; }
    float rate = clampf(p[P_MOD_RATE], 0.02f, 20.0f);
    double step = rate / sr, step2 = rate * 6.3 / sr;
    float depth_t = clampf(p[P_MOD_DEPTH], 0, 1), mix_t = clampf(p[P_MOD_MIX], 0, 1), color = clampf(p[P_MOD_COLOR], 0, 1);
    float ramp = 1.0f - expf(-1.0f / (0.02f * sr));
    float depth = g->mod_depth_s, mix = g->mod_mix_s;
    double ph = g->mod_ph, ph2 = g->mod_ph2;
    unsigned w = g->cw;
    float spread = 0.5f * color;                                 /* chorus / ensemble: R voices offset (0..180 deg) */
    float drift_c = op_coef(rate * 2.0f + 0.1f, sr);
    int drift_per = (int)(sr / (rate * 4.0f + 0.2f));
    for (int i = 0; i < n; i++) {
        depth += (depth_t - depth) * ramp;
        mix += (mix_t - mix) * ramp;
        ph += step; if (ph >= 1.0) ph -= 1.0;
        float xl = L[i], xr = R[i], yl, yr;
        if (type == MOD_CHORUS || type == MOD_ENSEMBLE) {
            g->cl[w & CMASK] = xl; g->cr[w & CMASK] = xr;
            float base = type == MOD_CHORUS ? 0.008f : 0.006f, sweep = type == MOD_CHORUS ? 0.005f : 0.003f;
            if (type == MOD_ENSEMBLE) { ph2 += step2; if (ph2 >= 1.0) ph2 -= 1.0; }
            if ((i & 15) == 0) {                                /* voice delays at control rate, then glide */
                for (int v = 0; v < 3; v++) {
                    float o = v / 3.0f;
                    double pa = ph + 16 * step, pb = ph2 + 16 * step2;
                    float sl = sinf((float)(TWO_PI * (pa + o))), sr2 = sinf((float)(TWO_PI * (pa + o + spread)));
                    if (type == MOD_ENSEMBLE) {
                        float f1 = sinf((float)(TWO_PI * (pb + o))), f2 = sinf((float)(TWO_PI * (pb + o + spread)));
                        sl = 0.8f * sl + 0.2f * f1; sr2 = 0.8f * sr2 + 0.2f * f2;
                    }
                    float tl = (base + sweep * depth * 0.5f * (1 + sl)) * sr, tr = (base + sweep * depth * 0.5f * (1 + sr2)) * sr;
                    if (g->ch_d[0][v] <= 0) { g->ch_d[0][v] = tl; g->ch_d[1][v] = tr; }
                    g->ch_inc[0][v] = (tl - g->ch_d[0][v]) / 16.0f; g->ch_inc[1][v] = (tr - g->ch_d[1][v]) / 16.0f;
                }
            }
            yl = yr = 0;
            for (int v = 0; v < 3; v++) {
                g->ch_d[0][v] += g->ch_inc[0][v]; g->ch_d[1][v] += g->ch_inc[1][v];
                yl += rd_herm(g->cl, CMASK, w, g->ch_d[0][v]);
                yr += rd_herm(g->cr, CMASK, w, g->ch_d[1][v]);
            }
            yl *= 0.4f; yr *= 0.4f;
            w = (w + 1) & CMASK;
        } else if (type == MOD_FLANGER || type == MOD_VIBRATO) {
            float s = sinf((float)(TWO_PI * ph)), c = cosf((float)(TWO_PI * ph));
            float dlt, drt;
            if (type == MOD_FLANGER) {
                dlt = (0.0005f + 0.006f * depth * 0.5f * (1 + s)) * sr;
                drt = (0.0005f + 0.006f * depth * 0.5f * (1 + c)) * sr;
            } else {
                if (--g->drift_cnt <= 0) { g->drift_cnt = drift_per; g->drift_t = rnd01(&g->rnd) * 2.0f - 1.0f; }
                g->drift += drift_c * (g->drift_t - g->drift);
                float m = (1.0f - color) * s + color * g->drift;        /* drift: random warble, like worn tape */
                dlt = drt = (0.004f + 0.0035f * depth * 0.5f * (1 + m)) * sr;
            }
            float ol = rd_herm(g->cl, CMASK, w, dlt), orr = rd_herm(g->cr, CMASK, w, drt);
            if (type == MOD_FLANGER) {
                float fb = color * 0.92f;
                g->cl[w & CMASK] = softclip(xl + fb * ol, 0.95f); g->cr[w & CMASK] = softclip(xr + fb * orr, 0.95f);
            } else { g->cl[w & CMASK] = xl; g->cr[w & CMASK] = xr; }
            yl = ol; yr = orr;
            w = (w + 1) & CMASK;
        } else if (type == MOD_PHASER) {
            float s = sinf((float)(TWO_PI * ph)), c = cosf((float)(TWO_PI * ph));
            if ((i & 15) == 0) {                                /* coefficients at control rate */
                for (int k = 0; k < 2; k++) {
                    float l = k ? c : s;
                    float fc = 180.0f * powf(2.0f, depth * 4.6f * 0.5f * (1 + l));
                    float t = tanf(PI_F * fminf(fc, sr * 0.45f) / sr);
                    g->ph_a[k] = (t - 1) / (t + 1);
                }
            }
            float in[2] = {xl, xr}, out[2], fb = 0.2f + color * 0.65f;
            for (int k = 0; k < 2; k++) {
                float a = g->ph_a[k], v = in[k] + fb * g->ph_fb[k];
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
            float s = shaped(sinf((float)(TWO_PI * ph)), color);
            float gn = 1.0f - depth * 0.5f * (1 - s);
            yl = xl * gn; yr = xr * gn;
        } else {                                                 /* auto-pan */
            float s = shaped(sinf((float)(TWO_PI * ph)), color);
            float gl = 1.0f - depth * 0.5f * (1 + s), gr = 1.0f - depth * 0.5f * (1 - s);
            yl = xl * gl; yr = xr * gr;
        }
        L[i] = xl + (yl - xl) * mix;
        R[i] = xr + (yr - xr) * mix;
    }
    g->mod_ph = ph; g->mod_ph2 = ph2; g->cw = w; g->mod_depth_s = depth; g->mod_mix_s = mix;
}

/* raised-cosine edge: 0 at the ends, 1 once `n` samples away from both (a = samples from start, b = to end) */
static inline float edge(int a, int b, int n) {
    int d = a < b ? a : b;
    if (d >= n) return 1.0f;
    if (d <= 0) return 0.0f;
    return 0.5f - 0.5f * cosf(PI_F * (float)d / (float)n);
}

/* Glitch: the input is cut into slices (rate = slices per second, or a note value from the tempo).
 * At each slice a dice roll (depth) may replace it with the slice before: repeated (stutter), reversed,
 * an octave up, slowed to a stop, or dropped out. Color opens up more kinds of event. */
static void glitch_block(Eng *g, float *L, float *R, int n) {
    const float *p = g->p;
    float sr = (float)g->sr;
    float rate = clampf(p[P_MOD_RATE], 0.5f, 40.0f);
    int S_t = (int)clampf(sr / rate, 512.0f, (float)(GSIZE / 2 - 64));
    float chance = clampf(p[P_MOD_DEPTH], 0, 1), color = clampf(p[P_MOD_COLOR], 0, 1), mix_t = clampf(p[P_MOD_MIX], 0, 1);
    float ramp = 1.0f - expf(-1.0f / (0.02f * sr));
    const int FADE = (int)(0.004f * sr), SEAM = (int)(0.0025f * sr);   /* 4 ms in/out, 2.5 ms at repeat seams */
    float mix = g->mod_mix_s;
    unsigned w = g->gw;
    for (int i = 0; i < n; i++) {
        mix += (mix_t - mix) * ramp;
        float xl = L[i], xr = R[i];
        g->gl[w & GMASK] = xl; g->gr[w & GMASK] = xr;
        if (g->gcnt <= 0) {                                      /* slice boundary: roll the dice */
            g->gS = S_t;
            g->gcnt = S_t;
            g->gev = GL_NONE;
            if (rnd01(&g->rnd) < chance) {
                int opts[5], k = 0;
                opts[k++] = GL_REPEAT;
                if (color > 0.2f) opts[k++] = GL_REVERSE;
                if (color > 0.45f) opts[k++] = GL_OCTAVE;
                if (color > 0.65f) opts[k++] = GL_STOP;
                if (color > 0.85f) opts[k++] = GL_DROP;
                g->gev = opts[xrand(&g->rnd) % k];
                g->glitch_count++;
                int div = 1 << (xrand(&g->rnd) % 3);             /* repeats: whole, half or quarter slice */
                g->gsub = g->gev == GL_REPEAT ? S_t / div : S_t;
                g->gstart = (int)w;
                g->grp = 0;
            }
        }
        int pos = g->gS - g->gcnt;
        float yl = xl, yr = xr, win = 1.0f;
        unsigned idx;
        switch (g->gev) {
        case GL_REPEAT: {
            int sub = g->gsub, pp = pos % sub;
            idx = (unsigned)(g->gstart - sub + pp);
            yl = g->gl[idx & GMASK]; yr = g->gr[idx & GMASK];
            win = edge(pp, sub - pp, SEAM);
            break; }
        case GL_REVERSE:
            idx = (unsigned)(g->gstart - 1 - pos);
            yl = g->gl[idx & GMASK]; yr = g->gr[idx & GMASK];
            break;
        case GL_OCTAVE: {
            int S = g->gS, pp = (2 * pos) % S;
            idx = (unsigned)(g->gstart - S + pp);
            yl = g->gl[idx & GMASK]; yr = g->gr[idx & GMASK];
            win = edge(pp, S - pp, 2 * SEAM);       /* read at double speed: the same fade in time */
            break; }
        case GL_STOP: {
            float v = 1.0f - (float)pos / g->gS;
            g->grp += v;
            float rp = (float)g->gstart - 64.0f + g->grp;
            int i0 = (int)floorf(rp); float f = rp - i0;
            yl = g->gl[i0 & GMASK] + (g->gl[(i0 + 1) & GMASK] - g->gl[i0 & GMASK]) * f;
            yr = g->gr[i0 & GMASK] + (g->gr[(i0 + 1) & GMASK] - g->gr[i0 & GMASK]) * f;
            win = edge((int)(v * 4.0f * SEAM), SEAM, SEAM);
            break; }
        case GL_DROP: yl = yr = 0; break;
        default: break;
        }
        /* the event fades in from the live sound and back out to it, and is always fully out
         * before the slice ends: the next slice (live or another event) starts from the live sound */
        float e = g->gev != GL_NONE ? edge(pos, g->gcnt - 1, FADE) : 0.0f;
        float ol = xl + (yl * win - xl) * e, orr = xr + (yr * win - xr) * e;
        L[i] = xl + (ol - xl) * mix;
        R[i] = xr + (orr - xr) * mix;
        g->gcnt--;
        w++;
    }
    g->gw = w; g->mod_mix_s = mix;
}

/* ---------------------------------------------------------------- delay */
static const float DIF_MS[2][3] = {{3.1f, 7.7f, 13.3f}, {3.7f, 8.9f, 12.1f}};
static const float DUAL_RATIO[8] = {1.0f, 0.875f, 0.75f, 0.6667f, 0.5f, 0.375f, 0.3333f, 0.25f};

static void delay_block(Eng *g, float *L, float *R, int n, int wet_only, int type, int mode) {
    const float *p = g->p;
    float sr = (float)g->sr;
    if (type == DLY_OFF) { g->dly_prev = 0; return; }
    if (g->dly_prev != type) {                                   /* switched on or another type (while silent): clean line */
        memset(g->dl, 0, DSIZE * sizeof(float)); memset(g->dr, 0, DSIZE * sizeof(float));
        for (int c = 0; c < 2; c++) for (int k = 0; k < 3; k++) line_clear(&g->dif[c][k]);
        bq_reset(&g->lp); bq_reset(&g->hp);                      /* filters must not remember the old echoes */
        g->tone_z[0] = g->tone_z[1] = 0;
        memset(g->dps, 0, sizeof(g->dps));
        g->dcur[0] = g->dcur[1] = -1; g->rc = 0;
    }
    g->dly_prev = type;
    float hc = p[P_DLY_HICUT], lc = p[P_DLY_LOCUT];
    g->lp.on = hc < 19000.0f;
    g->hp.on = lc > 21.0f;
    if (g->lp.on) bq_design(&g->lp, BQ_LP, hc, 0.7071f, 0, sr);
    if (g->hp.on) bq_design(&g->hp, BQ_HP, lc, 0.7071f, 0, sr);
    float color = clampf(p[P_DLY_COLOR], 0, 1);
    float t = clampf(p[P_DLY_TIME] * sr, 8.0f, (float)(DSIZE - 600));
    float tgt[2] = {t, t};
    if (type == DLY_DUAL) tgt[1] = fmaxf(8.0f, t * DUAL_RATIO[(int)clampf(color * 7.99f, 0, 7)]);
    if (type == DLY_REVERSE) tgt[0] = tgt[1] = fminf(t, (float)(DSIZE / 2 - 1200));
    if (g->dcur[0] < 0) { g->dcur[0] = tgt[0]; g->dcur[1] = tgt[1]; }
    float glide = 1.0f - expf(-1.0f / (0.07f * sr));        /* ~70 ms time constant */
    float fb = clampf(p[P_DLY_FB], 0.0f, 1.0f);
    if (type == DLY_DUAL) mode = MODE_STEREO;
    float mix = clampf(p[P_DLY_MIX], 0, 1);
    float dry_t = wet_only ? 0.0f : mix < 0.5f ? 1.0f : 2.0f * (1.0f - mix);   /* both at unity at 50% */
    float wet_t = mix < 0.5f ? 2.0f * mix : 1.0f;
    float ramp = 1.0f - expf(-1.0f / (0.012f * sr));
    float md_t = clampf(p[P_DLY_MODDEPTH], 0, 1) * 0.003f * sr;  /* up to 3 ms of wow */
    double mstep = clampf(p[P_DLY_MODRATE], 0.05f, 10.0f) / sr;
    double fstep = 5.7 / sr, wstep = 0.63 / sr;
    float width = clampf(p[P_DLY_WIDTH], 0, 1);
    /* character: tape = flutter + wow, saturation, softer top (Age: more of all of it);
     * analog (BBD) = dark repeats, warm clip; digital and reverse can be diffused into a wash */
    float age = (type == DLY_TAPE || type == DLY_ANALOG) ? color : 0.0f;
    float knee = type == DLY_TAPE ? 0.6f - 0.3f * age : type == DLY_ANALOG ? 0.5f - 0.25f * age : 0.85f;
    float flut = type == DLY_TAPE ? (0.00008f + 0.00025f * age) * sr : 0.0f;
    float wowd = type == DLY_TAPE ? 0.0012f * age * sr : 0.0f;
    float tc = type == DLY_TAPE ? op_coef(7500.0f - 4500.0f * age, sr) : type == DLY_ANALOG ? op_coef(3500.0f - 2300.0f * age, sr) : 1.0f;
    float diff = (type == DLY_DIGITAL || type == DLY_REVERSE) ? color : 0.0f;
    float dg = 0.72f * diff;
    float shim = type == DLY_SHIMMER ? color : 0.0f;
    float win = 2048.0f;

    float *bl = g->dl, *br = g->dr;
    unsigned w = g->dw, dfw = g->difw;
    float dry_s = g->dry_s, wet_s = g->wet_s, mdepth = g->mdepth, d0 = g->dcur[0], d1 = g->dcur[1];
    float tz0 = g->tone_z[0], tz1 = g->tone_z[1], rc = g->rc;
    double mph = g->mph, fph = g->flut, wph = g->wow;
    Biquad *lp = &g->lp, *hp = &g->hp;
    float T = tgt[0];
    for (int i = 0; i < n; i++) {
        float frz = g->FRZ[i];
        dry_s += (dry_t - dry_s) * ramp;
        wet_s += (wet_t - wet_s) * ramp;
        mdepth += (md_t - mdepth) * ramp;
        d0 += (tgt[0] - d0) * glide;
        d1 += (tgt[1] - d1) * glide;
        mph += mstep; if (mph >= 1.0) mph -= 1.0;
        fph += fstep; if (fph >= 1.0) fph -= 1.0;
        wph += wstep; if (wph >= 1.0) wph -= 1.0;
        float yl, yr;
        if (type == DLY_REVERSE) {
            /* two heads read the last T samples backwards, each in a sin^2 window, half a window apart */
            rc += 1.0f; if (rc >= T) rc -= T;
            float rc2 = rc + 0.5f * T; if (rc2 >= T) rc2 -= T;
            float g1 = sinf(PI_F * rc / T); g1 *= g1;
            unsigned a = (w - (unsigned)(2.0f * rc) - 1) & DMASK, b = (w - (unsigned)(2.0f * rc2) - 1) & DMASK;
            yl = g1 * bl[a] + (1 - g1) * bl[b];
            yr = g1 * br[a] + (1 - g1) * br[b];
        } else {
            float ml = 0, mr = 0;
            if (mdepth > 0.01f) {
                float s = sinf((float)(TWO_PI * mph));
                float c = cosf((float)(TWO_PI * mph));
                ml = mdepth * 0.5f * (1 + s);
                mr = mode == MODE_MONO ? ml : mdepth * 0.5f * (1 + c);      /* 90 deg apart: wide repeats */
            }
            if (flut > 0) {
                float fl = flut * sinf((float)(TWO_PI * fph)) + wowd * (1 + sinf((float)(TWO_PI * wph)));
                ml += flut + fl; mr += flut + fl;
            }
            yl = rd_herm(bl, DMASK, w, d0 + ml);
            yr = rd_herm(br, DMASK, w, d1 + mr);
        }
        float keep = 1.0f - frz;                              /* freeze: the loop stops losing anything */
        if (tc < 1.0f) {
            tz0 += tc * (yl - tz0); tz1 += tc * (yr - tz1);
            yl = tz0 + (yl - tz0) * frz; yr = tz1 + (yr - tz1) * frz;
        }
        if (lp->on) { float a = bq_run(lp, 0, yl), b = bq_run(lp, 1, yr); yl = a + (yl - a) * frz; yr = b + (yr - b) * frz; }
        if (hp->on) { float a = bq_run(hp, 0, yl), b = bq_run(hp, 1, yr); yl = a + (yl - a) * frz; yr = b + (yr - b) * frz; }
        if (dg > 0.01f) {                                     /* diffusion inside the loop: each repeat gets softer */
            for (int k = 0; k < 3; k++) { yl = ap(&g->dif[0][k], dfw, yl, dg); yr = ap(&g->dif[1][k], dfw, yr, dg); }
            dfw++;
        }
        if (shim > 0.01f) {                                   /* shimmer: the repeats climb an octave each time */
            float sh = shim * keep;
            float pl = ps_run(&g->dps[0], yl, 2.0f, win), pr = ps_run(&g->dps[1], yr, 2.0f, win);
            yl += (pl - yl) * sh; yr += (pr - yr) * sh;
        }
        float xl = L[i], xr = R[i];
        float il = xl * keep, ir = xr * keep;
        float f = fb + (1.0f - fb) * frz;
        float wl, wr;
        if (mode == MODE_PINGPONG) {
            wl = 0.5f * (il + ir) + f * yr;
            wr = f * yl;
        } else if (mode == MODE_MONO) {
            float m = 0.5f * (yl + yr);
            wl = wr = 0.5f * (il + ir) + f * m;
            yl = yr = m;
        } else {
            wl = il + f * yl;
            wr = ir + f * yr;
        }
        bl[w] = softclip(wl, knee);
        br[w] = softclip(wr, knee);
        w = (w + 1) & DMASK;
        float mid = 0.5f * (yl + yr), side = 0.5f * (yl - yr) * width;
        L[i] = dry_s * xl + wet_s * (mid + side);
        R[i] = dry_s * xr + wet_s * (mid - side);
    }
    g->dw = w; g->difw = dfw; g->dry_s = dry_s; g->wet_s = wet_s; g->mdepth = mdepth; g->dcur[0] = d0; g->dcur[1] = d1;
    g->mph = mph; g->flut = fph; g->wow = wph; g->rc = rc;
    g->tone_z[0] = tz0; g->tone_z[1] = tz1;
}

/* ---------------------------------------------------------------- reverb (Dattorro plate topology, extended) */
static const int DAT_IN[4] = {142, 107, 379, 277};
static const int DAT_TANK[NTANK] = {672, 4453, 1800, 3720, 908, 4217, 2656, 3163};
static const float BLOOM_MS[4] = {35.3f, 53.1f, 79.7f, 113.9f};
/* output taps (Dattorro 1997): line index, offset */
static const int TAP_L[7][2] = {{5, 266}, {5, 2974}, {6, 1913}, {7, 1996}, {1, 1990}, {2, 187}, {3, 1066}};
static const int TAP_R[7][2] = {{1, 353}, {1, 3627}, {2, 1228}, {3, 2673}, {5, 2111}, {6, 335}, {7, 121}};
static const float TAP_SIGN[7] = {1, 1, -1, 1, -1, -1, -1};
static const float REV_SIZES[7] = {1.0f, 0.55f, 1.0f, 1.7f, 2.6f, 2.0f, 2.2f};
#define MAX_SIZE 2.7f
/* output level per type (measured on a long pad: wet about -3 dB from the dry at decay 60%) */
static const float REV_OUT[7] = {0.6f, 0.21f, 0.28f, 0.37f, 0.25f, 0.34f, 0.28f};

static void rv_alloc(Reverb *r, int sr) {
    float k = sr / 29761.0f;
    line_alloc(&r->pre, PSIZE - 64);
    for (int i = 0; i < 4; i++) line_alloc(&r->in[i], (int)(DAT_IN[i] * k) + 4);
    for (int i = 0; i < 4; i++) line_alloc(&r->bloom[i], (int)(BLOOM_MS[i] * 0.001f * sr) + 4);
    for (int i = 0; i < NTANK; i++) line_alloc(&r->tank[i], (int)(DAT_TANK[i] * k * MAX_SIZE) + 80);
    r->type = -1;
}
static void rv_clear(Reverb *r) {
    line_clear(&r->pre);
    for (int i = 0; i < 4; i++) { line_clear(&r->in[i]); line_clear(&r->bloom[i]); }
    for (int i = 0; i < NTANK; i++) line_clear(&r->tank[i]);
    r->bw = r->lp[0] = r->lp[1] = r->hp[0] = r->hp[1] = r->lastL = r->lastR = r->shim_hp = r->shim_lp = 0;
    memset(r->ps.buf, 0, sizeof(r->ps.buf));
}
static void rv_size(Reverb *r, int type, int sr) {
    float k = sr / 29761.0f;
    float size = REV_SIZES[type];
    r->size = size;
    for (int i = 0; i < 4; i++) r->in[i].len = (int)(DAT_IN[i] * k);
    for (int i = 0; i < NTANK; i++) r->tank[i].len = (int)(DAT_TANK[i] * k * size);
    for (int i = 0; i < 7; i++) { r->taps[i] = (int)(TAP_L[i][1] * k * size); r->taps[7 + i] = (int)(TAP_R[i][1] * k * size); }
    r->type = type;
}

static void reverb_block(Eng *g, float *L, float *R, int n, int wet_only, int type) {
    const float *p = g->p;
    Reverb *r = &g->rv;
    float sr = (float)g->sr;
    if (type == REV_OFF) { g->rev_prev = 0; return; }
    if (g->rev_prev == 0 || type != r->type) { rv_size(r, type, g->sr); rv_clear(r); r->precur = -1; r->tcomp = 0; }
    g->rev_prev = type;
    float decay0 = 0.2f + 0.795f * clampf(p[P_REV_DECAY], 0, 1);
    float damp0 = 0.05f + 0.8f * clampf(p[P_REV_DAMP], 0, 1);
    float lc = p[P_REV_LOCUT];
    float hpc0 = lc > 21.0f ? op_coef(lc, sr) : 0.0f;
    float pre_t = clampf(p[P_REV_PRE], 0, 0.3f) * sr + 1;
    if (r->precur < 0) r->precur = pre_t;
    float exc0 = clampf(p[P_REV_MOD], 0, 1) * (type == REV_CLOUD ? 40.0f : 24.0f) * (sr / 29761.0f) * r->size;
    double lstep = (type == REV_CLOUD ? 0.6 : 0.9) / sr;
    float color = clampf(p[P_REV_COLOR], 0, 1);
    /* color: diffusion for room/plate/hall/cloud, octave amount for shimmer, slow build for bloom */
    float diff = (type == REV_SHIMMER || type == REV_BLOOM) ? 0.8f : color;
    float g12 = 0.45f + 0.35f * diff, g34 = 0.35f + 0.33f * diff;
    float shim = type == REV_SHIMMER ? color : 0.0f;
    float bloom_g = type == REV_BLOOM ? 0.45f + 0.4f * color : 0.0f;
    float shp = op_coef(300.0f, sr), slp = op_coef(9000.0f, sr);
    float mix = clampf(p[P_REV_MIX], 0, 1);
    float dry_t = wet_only ? 0.0f : mix < 0.5f ? 1.0f : 2.0f * (1.0f - mix);
    float wet_t = mix < 0.5f ? 2.0f * mix : 1.0f;
    float ramp = 1.0f - expf(-1.0f / (0.012f * sr));
    /* Loudness: the tank keeps adding up energy, about 1 / (1 - decay^8) in power (decay four times per loop).
     * Long tails would get far louder than short ones and run into the limiters (digital distortion), so less
     * goes into the tank as the decay gets longer: about the same loudness as decay 60%, a little louder above.
     * The octave feedback of Shimmer adds energy too. */
    float d2 = decay0 * decay0, d8 = d2 * d2 * d2 * d2;
    float tcomp_t = powf((1.0f - d8) / 0.956f, 0.4f);
    tcomp_t = clampf(tcomp_t, 0.06f, 1.0f) / (1.0f + 0.8f * shim);
    if (r->tcomp <= 0) r->tcomp = tcomp_t;
    float bwc = 0.6f;
    float outg = REV_OUT[type];                 /* wet a few dB under the dry at 100% mix, for every type */
    Line *t = r->tank;
    int l0 = t[0].len, l4 = t[4].len;
    unsigned w = r->w;
    float dry_s = g->rdry_s, wet_s = g->rwet_s, lastL = r->lastL, lastR = r->lastR, bw = r->bw;
    float lp0 = r->lp[0], lp1 = r->lp[1], hp0 = r->hp[0], hp1 = r->hp[1], precur = r->precur;
    float shz = r->shim_hp, slz = r->shim_lp;
    double lph = r->lph;
    for (int i = 0; i < n; i++) {
        float frz = g->FRZ[i], keep = 1.0f - frz;
        float decay = decay0 + (1.0f - decay0) * frz, damp = damp0 * keep, hpc = hpc0 * keep, exc = exc0 * keep;
        dry_s += (dry_t - dry_s) * ramp;
        wet_s += (wet_t - wet_s) * ramp;
        precur += (pre_t - precur) * 0.0005f;
        lph += lstep; if (lph >= 1.0) lph -= 1.0;
        float s = sinf((float)(TWO_PI * lph)), c = cosf((float)(TWO_PI * lph));
        /* input: mono sum -> pre-delay -> bandwidth -> diffusers (-> bloom) */
        float x = 0.5f * (L[i] + R[i]) * keep;
        line_wr(&r->pre, w, x);
        int pi0 = (int)precur; float pf = precur - pi0;
        float pa = line_rd(&r->pre, w, pi0), pb = line_rd(&r->pre, w, pi0 + 1);
        x = pa + (pb - pa) * pf;
        bw += bwc * (x - bw);
        x = bw;
        x = ap(&r->in[0], w, x, g12);
        x = ap(&r->in[1], w, x, g12);
        x = ap(&r->in[2], w, x, g34);
        x = ap(&r->in[3], w, x, g34);
        if (bloom_g > 0) for (int k = 0; k < 4; k++) x = ap(&r->bloom[k], w, x, bloom_g);
        r->tcomp += (tcomp_t - r->tcomp) * 0.0005f;
        x *= r->tcomp;
        if (shim > 0.01f) {                                     /* the tail, an octave up, back into the tank */
            float y = ps_run(&r->ps, 0.5f * (lastL + lastR), 2.0f, 2400.0f);
            shz += shp * (y - shz); y -= shz;
            slz += slp * (y - slz);
            x += slz * shim * 0.55f * keep;
        }
        /* tank, left half */
        float a = x + decay * lastR;
        a = ap_mod(&t[0], w, a, -0.7f, l0 + exc * (1 + s));
        float d1 = line_rd(&t[1], w, t[1].len); line_wr(&t[1], w, a);
        lp0 += (1 - damp) * (d1 - lp0);
        hp0 += hpc * (lp0 - hp0);
        float b = ap(&t[2], w, (lp0 - hp0) * decay, 0.5f);
        float leftOut = line_rd(&t[3], w, t[3].len); line_wr(&t[3], w, b);
        /* tank, right half */
        float a2 = x + decay * leftOut;
        a2 = ap_mod(&t[4], w, a2, -0.7f, l4 + exc * (1 + c));
        float d2 = line_rd(&t[5], w, t[5].len); line_wr(&t[5], w, a2);
        lp1 += (1 - damp) * (d2 - lp1);
        hp1 += hpc * (lp1 - hp1);
        float b2 = ap(&t[6], w, (lp1 - hp1) * decay, 0.5f);
        float rightOut = line_rd(&t[7], w, t[7].len); line_wr(&t[7], w, b2);
        { float m = fmaxf(fabsf(leftOut), fabsf(rightOut)); if (m > r->tank_pk) r->tank_pk = m; if (m > 0.9f) r->tank_clip++; }
        lastL = softclip(leftOut, 0.9f); lastR = softclip(rightOut, 0.9f);
        /* output taps */
        float yl = 0, yr = 0;
        for (int k = 0; k < 7; k++) {
            yl += TAP_SIGN[k] * line_rd(&t[TAP_L[k][0]], w + 1, r->taps[k]);
            yr += TAP_SIGN[k] * line_rd(&t[TAP_R[k][0]], w + 1, r->taps[7 + k]);
        }
        w++;
        L[i] = dry_s * L[i] + wet_s * outg * yl;
        R[i] = dry_s * R[i] + wet_s * outg * yr;
    }
    r->w = w; r->lastL = lastL; r->lastR = lastR; r->bw = bw; r->lp[0] = lp0; r->lp[1] = lp1; r->hp[0] = hp0; r->hp[1] = hp1;
    r->precur = precur; r->shim_hp = shz; r->shim_lp = slz;
    r->lph = lph; g->rdry_s = dry_s; g->rwet_s = wet_s;
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
    g->gl = (float *)calloc(GSIZE, sizeof(float));
    g->gr = (float *)calloc(GSIZE, sizeof(float));
    g->rb = (float *)calloc(RB * 2, sizeof(float));
    for (int c = 0; c < 2; c++) for (int k = 0; k < 3; k++) line_alloc(&g->dif[c][k], (int)(DIF_MS[c][k] * 0.001f * sr));
    rv_alloc(&g->rv, sr);
    g->p[P_ON] = 1; g->p[P_INGAIN] = 1; g->p[P_OUTGAIN] = 1; g->p[P_TRAILS] = 1; g->p[P_BALANCE] = 0.5f; g->p[P_WIDTH_OUT] = 1;
    g->p[P_CMP_THR] = -20; g->p[P_CMP_RATIO] = 4; g->p[P_CMP_ATK] = 0.01f; g->p[P_CMP_REL] = 0.2f; g->p[P_CMP_MIX] = 1;
    g->p[P_CMP_SWELL] = 0.5f; g->p[P_AMP_BASS] = g->p[P_AMP_MID] = g->p[P_AMP_TREBLE] = 0.5f; g->p[P_AMP_LEVEL] = 1;
    g->p[P_DLY_TIME] = 0.375f; g->p[P_DLY_FB] = 0.4f; g->p[P_DLY_MIX] = 0.3f; g->p[P_DLY_WIDTH] = 1;
    g->p[P_DLY_HICUT] = 8000; g->p[P_DLY_LOCUT] = 80; g->p[P_DLY_MODRATE] = 0.5f; g->p[P_DLY_TYPE] = DLY_DIGITAL;
    g->p[P_REV_LOCUT] = 20; g->p[P_MOD_RATE] = 1;
    g->dcur[0] = g->dcur[1] = -1; g->on_s = 1; g->in_s = 1; g->out_s = 1; g->dry_s = 1; g->wet_s = 0.6f; g->rdry_s = 1;
    g->pdry_s = 1; g->bal_a_s = g->bal_b_s = 1; g->width_s = 1; g->cmp_swell_g = 1; g->cmp_mix_s = 1;
    g->rnd = 0x9E3779B9u;
    g->ratio = 1.0; g->priming = 1; g->target = 512;
    for (int k = 0; k < 4; k++) g->cab[k].key[0] = -1;
    g->lp.key[0] = g->hp.key[0] = g->amp_pre.key[0] = g->amp_bass.key[0] = g->amp_mid.key[0] = g->amp_treb.key[0] = -1;
    pthread_mutex_init(&g->lock, NULL);
    return g;
}

API void eng_set_params(Eng *g, const float *p) {
    pthread_mutex_lock(&g->lock);
    memcpy(g->p, p, sizeof(float) * P_COUNT);
    pthread_mutex_unlock(&g->lock);
}

static void clear_tails(Eng *g) {
    memset(&g->sp_dly, 0, sizeof(g->sp_dly)); memset(&g->sp_rev, 0, sizeof(g->sp_rev));
    g->dly_prev = 0; g->rev_prev = 0;
    memset(g->dl, 0, DSIZE * sizeof(float));
    memset(g->dr, 0, DSIZE * sizeof(float));
    memset(g->cl, 0, CSIZE * sizeof(float));
    memset(g->cr, 0, CSIZE * sizeof(float));
    for (int c = 0; c < 2; c++) for (int k = 0; k < 3; k++) line_clear(&g->dif[c][k]);
    memset(g->dps, 0, sizeof(g->dps));
    rv_clear(&g->rv);
}

static void reset_states(Eng *g) {
    clear_tails(g);
    Biquad *bs[] = {&g->lp, &g->hp, &g->amp_pre, &g->amp_bass, &g->amp_mid, &g->amp_treb, &g->cab[0], &g->cab[1], &g->cab[2], &g->cab[3]};
    for (unsigned k = 0; k < sizeof(bs) / sizeof(bs[0]); k++) bq_reset(bs[k]);
    memset(g->ph_x, 0, sizeof(g->ph_x)); memset(g->ph_y, 0, sizeof(g->ph_y));
    g->ph_fb[0] = g->ph_fb[1] = g->fl_fb[0] = g->fl_fb[1] = 0; g->tone_z[0] = g->tone_z[1] = 0;
    memset(g->amp_hp, 0, sizeof(g->amp_hp)); memset(g->amp_lp1, 0, sizeof(g->amp_lp1));
    memset(g->amp_hp2, 0, sizeof(g->amp_hp2)); memset(g->amp_dc, 0, sizeof(g->amp_dc));
    g->cmp_gr = g->cmp_env = g->cmp_rms = 0;
    memset(g->gl, 0, GSIZE * sizeof(float)); memset(g->gr, 0, GSIZE * sizeof(float));
    g->dcur[0] = g->dcur[1] = -1;
}

API void eng_clear(Eng *g) {
    pthread_mutex_lock(&g->lock);
    clear_tails(g);
    pthread_mutex_unlock(&g->lock);
}


/* ---------------------------------------------------------------- switching without clicks */
static inline float scurve(float x) { return x * x * (3.0f - 2.0f * x); }

/* One block of a section's fade. While `cur` differs from `target` the effect fades out; once silent at the
 * end of a block it changes over (the section's own code clears its state as before) and fades in.
 * Fills gs[] with the S-curve gain. Off (key 0) is never processed, so it changes over at once. */
static void xf_run(XF *x, int target, int n, float step, float *gs) {
    if (x->cur == 0) x->pos = 0;
    float pos = x->pos;
    for (int i = 0; i < n; i++) {
        if (x->cur != target) { pos -= step; if (pos < 0) pos = 0; }
        else { pos += step; if (pos > 1) pos = 1; }
        gs[i] = scurve(pos);
    }
    x->pos = pos;
    if (x->cur != target && pos <= 0) x->cur = target;
}
static inline float xf_step(Eng *g, float ms) { return 1.0f / (ms * 0.001f * g->sr); }

/* out = dry + (effect - dry) * gain; skipped when the gain is 1 for the whole block */
static void xf_mix(const float *gs, const float *dl, const float *dr, float *L, float *R, int n) {
    if (gs[0] >= 1.0f && gs[n - 1] >= 1.0f) return;
    for (int i = 0; i < n; i++) { L[i] = dl[i] + (L[i] - dl[i]) * gs[i]; R[i] = dr[i] + (R[i] - dr[i]) * gs[i]; }
}

/* Delay or reverb, with spillover. key: 0 = off, else the type (delay: type*8 + mode).
 * Switched off: the input closes over `in_ms`, the effect keeps running on what it has; it stops when its
 * output has stayed under -90 dB for `hold` samples (a never-ending tail fades out over 1 s after 20 s).
 * On again with the same key: the input opens again, nothing is cleared. Another key: crossfade. */
static void spill_block(Eng *g, Spill *s, int target, float *L, float *R, int n, int wet_only, int is_rev,
                        float in_ms, float x_ms, int hold, const float *gate) {
    int cur = s->xf.cur;
    if (cur == 0 && target == 0) { if (wet_only) { memset(L, 0, n * 4); memset(R, 0, n * 4); } return; }
    int spilling = cur != 0 && target == 0;
    float istep = xf_step(g, in_ms);
    float pin = s->pin;
    if (cur == 0) { s->pin = pin = 0.0f; s->tg = 1.0f; }   /* a new effect also gets its input faded in */
    /* gates: the input (XI) and the crossfade (XG) */
    if (spilling) {
        for (int i = 0; i < n; i++) { pin -= istep; if (pin < 0) pin = 0; g->XI[i] = scurve(pin); g->XG[i] = 1.0f; }
    } else {
        for (int i = 0; i < n; i++) { pin += istep; if (pin > 1) pin = 1; g->XI[i] = scurve(pin); }
        xf_run(&s->xf, target, n, xf_step(g, x_ms), g->XG);
        if (s->xf.cur != cur && s->xf.cur != 0) pin = 0;          /* changed over: open the input from zero */
    }
    s->pin = pin;
    int run = cur;                                               /* this block runs the key it started with */
    if (run == 0) { if (wet_only) { memset(L, 0, n * 4); memset(R, 0, n * 4); } return; }
    memcpy(g->TL, L, n * 4); memcpy(g->TR, R, n * 4);
    if (gate) for (int i = 0; i < n; i++) g->XI[i] *= gate[i];     /* routing change: the input follows the fade too */
    for (int i = 0; i < n; i++) { L[i] *= g->XI[i]; R[i] *= g->XI[i]; }
    if (is_rev) reverb_block(g, L, R, n, wet_only, run);
    else delay_block(g, L, R, n, wet_only, run / 8, run % 8);
    float pk = 0;
    if (spilling && pin <= 0) for (int i = 0; i < n; i++) pk = fmaxf(pk, fmaxf(fabsf(L[i]), fabsf(R[i])));
    /* out = in * (1 - gin) + effect(in * gin), then the crossfade, then the tail fade */
    float tg = s->tg, tstep = 1.0f / g->sr;
    for (int i = 0; i < n; i++) {
        float gi = g->XI[i], gx = g->XG[i];
        if (spilling && s->tail_n > 20 * g->sr) { tg -= tstep; if (tg < 0) tg = 0; }
        float dl = wet_only ? 0.0f : g->TL[i], dr = wet_only ? 0.0f : g->TR[i];
        float yl = (wet_only ? 0.0f : g->TL[i] * (1.0f - gi)) + L[i], yr = (wet_only ? 0.0f : g->TR[i] * (1.0f - gi)) + R[i];
        L[i] = dl + (yl - dl) * gx * tg;
        R[i] = dr + (yr - dr) * gx * tg;
    }
    s->tg = tg;
    if (spilling) {
        s->tail_n += n;
        if (pin <= 0) { if (pk < 3.2e-5f) s->quiet += n; else s->quiet = 0; }
        if ((pin <= 0 && s->quiet > hold) || tg <= 0) {       /* the tail is over: really off */
            s->xf.cur = 0; s->xf.pos = 0; s->quiet = 0; s->tail_n = 0; s->tg = 1;
            if (is_rev) g->rev_prev = 0; else g->dly_prev = 0;
        }
    } else { s->quiet = 0; s->tail_n = 0; s->tg = 1; }
}

static inline float dry_of(float mix) { return mix < 0.5f ? 1.0f : 2.0f * (1.0f - mix); }

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

    /* chain on/off: the input to the chain fades out, the dry signal fades in.
     * With trails the delay and reverb tails keep playing after bypass. */
    float on_t = p[P_ON] > 0.5f ? 1.0f : 0.0f;
    float ramp = 1.0f - expf(-1.0f / (0.012f * sr));
    float fr_t = p[P_FREEZE] > 0.5f ? 1.0f : 0.0f, fr_ramp = 1.0f - expf(-1.0f / (0.03f * sr));
    float on_s = g->on_s, frz = g->frz_s;
    for (int i = 0; i < n; i++) {
        on_s += (on_t - on_s) * ramp;
        frz += (fr_t - frz) * fr_ramp;
        g->FRZ[i] = frz > 0.999f ? 1.0f : frz;
        g->ON[i] = on_s;
        g->DL[i] = L[i]; g->DR[i] = R[i];
        L[i] *= on_s; R[i] *= on_s;
    }
    g->on_s = on_s; g->frz_s = frz;
    int trails = p[P_TRAILS] > 0.5f;
    if (!trails && on_s < 1e-4f && on_t < 0.5f) {
        if (!g->cleared) { clear_tails(g); g->cleared = 1; }
        for (int i = 0; i < n; i++) { L[i] = R[i] = 0; }
    } else {
        g->cleared = 0;
        /* COMP, AMP (type and cabinet), MOD: crossfade on any change of type */
        int k_cmp = (int)p[P_CMP_TYPE], k_amp = (int)p[P_AMP_TYPE], k_mod = (int)p[P_MOD_TYPE];
        if (k_amp) k_amp = k_amp * 8 + (int)clampf(p[P_AMP_CAB], 0, 4);
        int k_dly = (int)p[P_DLY_TYPE], k_rev = (int)p[P_REV_TYPE];
        if (k_dly) k_dly = k_dly * 8 + (k_dly == DLY_DUAL ? 0 : (int)clampf(p[P_DLY_MODE], 0, 2));
        {
            XF *xs[3] = {&g->xf_cmp, &g->xf_amp, &g->xf_mod};
            int ks[3] = {k_cmp, k_amp, k_mod};
            float ms[3] = {10.0f, 10.0f, 25.0f};
            for (int sct = 0; sct < 3; sct++) {
                XF *x = xs[sct];
                int run = x->cur;
                xf_run(x, ks[sct], n, xf_step(g, ms[sct]), g->XG);
                if (run == 0) { if (sct == 2) mod_block(g, L, R, n, 0); continue; }
                memcpy(g->TL, L, n * 4); memcpy(g->TR, R, n * 4);
                if (sct == 0) comp_block(g, L, R, n, run);
                else if (sct == 1) amp_block(g, L, R, n, run / 8, run % 8);
                else mod_block(g, L, R, n, run);
                xf_mix(g->XG, g->TL, g->TR, L, R, n);
            }
            if (g->xf_cmp.cur == 0) comp_block(g, L, R, 0, 0);
        }
        /* routing change: delay + reverb fade out, the routing changes over, they fade back in */
        int route_t = (int)p[P_ROUTING] == ROUTING_PARALLEL ? 2 : 1;
        if (g->xf_route.cur == 0) { g->xf_route.cur = route_t; g->xf_route.pos = 1; }
        int route = g->xf_route.cur;
        float rg[MAXBLOCK];
        {
            float pos = g->xf_route.pos, st = xf_step(g, 30.0f);
            for (int i = 0; i < n; i++) {
                if (route != route_t) { pos -= st; if (pos < 0) pos = 0; } else { pos += st; if (pos > 1) pos = 1; }
                rg[i] = scurve(pos);
            }
            g->xf_route.pos = pos;
            if (route != route_t && pos <= 0) g->xf_route.cur = route_t;
        }
        float ML[MAXBLOCK], MR[MAXBLOCK];
        memcpy(ML, L, n * 4); memcpy(MR, R, n * 4);
        int dhold = (int)((clampf(p[P_DLY_TIME], 0, 5) + 0.3f) * sr);
        int rhold = (int)((clampf(p[P_REV_PRE], 0, 0.3f) + 0.4f) * sr);
        if (route == 2) {
            /* split: delay and reverb each get the same signal, wet only; then balance and spread between them */
            memcpy(g->AL, L, n * sizeof(float)); memcpy(g->AR, R, n * sizeof(float));
            memcpy(g->BL, L, n * sizeof(float)); memcpy(g->BR, R, n * sizeof(float));
            spill_block(g, &g->sp_dly, k_dly, g->AL, g->AR, n, 1, 0, 30.0f, 30.0f, dhold, rg);
            spill_block(g, &g->sp_rev, k_rev, g->BL, g->BR, n, 1, 1, 80.0f, 80.0f, rhold, rg);
            float dry_t = fminf(k_dly ? dry_of(p[P_DLY_MIX]) : 1.0f, k_rev ? dry_of(p[P_REV_MIX]) : 1.0f);
            float b = clampf(p[P_BALANCE], 0, 1);
            float ga_t = fminf(1.0f, 2.0f * (1.0f - b)), gb_t = fminf(1.0f, 2.0f * b);
            float sp_t = clampf(p[P_SPREAD], 0, 1);
            float pd = g->pdry_s, ga = g->bal_a_s, gb = g->bal_b_s, sp = g->spread_s;
            for (int i = 0; i < n; i++) {
                pd += (dry_t - pd) * ramp; ga += (ga_t - ga) * ramp; gb += (gb_t - gb) * ramp; sp += (sp_t - sp) * ramp;
                /* spread: delay to the left, reverb to the right */
                float al = g->AL[i] + g->AR[i] * sp, ar = g->AR[i] * (1.0f - sp);
                float bl = g->BL[i] * (1.0f - sp), br = g->BR[i] + g->BL[i] * sp;
                L[i] = pd * L[i] + ga * al + gb * bl;
                R[i] = pd * R[i] + ga * ar + gb * br;
            }
            g->pdry_s = pd; g->bal_a_s = ga; g->bal_b_s = gb; g->spread_s = sp;
        } else {
            spill_block(g, &g->sp_dly, k_dly, L, R, n, 0, 0, 30.0f, 30.0f, dhold, rg);
            spill_block(g, &g->sp_rev, k_rev, L, R, n, 0, 1, 80.0f, 80.0f, rhold, rg);
        }
        if (!(rg[0] >= 1.0f && rg[n - 1] >= 1.0f))
            for (int i = 0; i < n; i++) { L[i] = ML[i] + (L[i] - ML[i]) * rg[i]; R[i] = MR[i] + (R[i] - MR[i]) * rg[i]; }
        float chk = 0;
        for (int i = 0; i < n; i++) chk += L[i] + R[i];
        if (!isfinite(chk)) {                       /* never let a bad number stick in the lines */
            reset_states(g);
            g->nan_resets++;
            for (int i = 0; i < n; i++) { L[i] = R[i] = 0; }
        }
    }
    float og0 = g->out_s, og1 = p[P_OUTGAIN];
    g->out_s = og1;
    float wd_t = clampf(p[P_WIDTH_OUT], 0, 2), wd = g->width_s;
    float opk0 = 0, opk1 = 0;
    for (int i = 0; i < n; i++) {
        float on = g->ON[i], wg = trails ? 1.0f : on;
        float og = og0 + (og1 - og0) * (i + 1) / n;
        wd += (wd_t - wd) * ramp;
        float l = L[i] * wg, r = R[i] * wg;
        float m = 0.5f * (l + r), s = 0.5f * (l - r) * wd;        /* output width: 0 mono, 1 as is, 2 extra wide */
        l = m + s; r = m - s;
        float ol = softclip((l + g->DL[i] * (1.0f - on)) * og, 0.9f);
        float orr = softclip((r + g->DR[i] * (1.0f - on)) * og, 0.9f);
        L[i] = ol; R[i] = orr;
        float al = fabsf(ol), ar = fabsf(orr);
        if (al > opk0) opk0 = al;
        if (ar > opk1) opk1 = ar;
    }
    g->width_s = wd;
    g->out_pk[0] = fmaxf(g->out_pk[0] * 0.9f, opk0);
    g->out_pk[1] = fmaxf(g->out_pk[1] * 0.9f, opk1);
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

/* Status: see STATUS_KEYS in brickambientfx.py */
#define NSTATUS 39
API int eng_status(Eng *g, float *o, int max) {
    float s[NSTATUS] = {
        g->in_pk[0], g->in_pk[1], g->out_pk[0], g->out_pk[1], g->load,
        (float)g->underruns, (float)g->cap_xruns, (float)g->rs_under, (float)g->fill_avg,
        (float)((g->ratio - 1.0) * 1e6), (float)g->cap_open, (float)g->play_open, (float)g->rt_play,
        (float)g->cap_period, (float)g->play_buffer, g->dcur[0] / g->sr, (float)g->blocks, (float)g->cap_rate,
        (float)g->target, (float)g->priming, (float)g->cap_channels, (float)g->rs_over, (float)g->play_period,
        (float)g->rt_cap, (float)g->cap_frames, (float)g->cap_reads, g->cap_raw_pk, (float)g->cap_buffer,
        (float)g->cap_opens, (float)g->wpos, (float)g->nan_resets,
        g->cmp_gr_show, (float)g->gev, (float)g->glitch_count, g->frz_s, g->rv.tank_pk, (float)g->rv.tank_clip,
        (float)(g->sp_dly.xf.cur != 0 && (int)g->p[P_DLY_TYPE] == 0), (float)(g->sp_rev.xf.cur != 0 && (int)g->p[P_REV_TYPE] == 0) };
    int n = max < NSTATUS ? max : NSTATUS;
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
        for (int chn = 2; chn >= 1; chn--) {
            void *pcm = NULL;
            int err = A.open(&pcm, dev, 1, 0);                         /* capture, blocking */
            if (err < 0) { snprintf(g->cap_err, sizeof(g->cap_err), "%s: %s", dev, A.strerror(err)); break; }
            err = A.setp(pcm, 2, 3, (unsigned)chn, (unsigned)g->sr, 1, 6000);   /* S16_LE, interleaved, 6 ms */
            if (err < 0) {
                snprintf(g->cap_err, sizeof(g->cap_err), "%s %dch: %s", dev, chn, A.strerror(err));
                A.close(pcm);
                continue;
            }
            unsigned long buf = 0, per = 0;
            A.getp(pcm, &buf, &per);
            if (per == 0 || per > MAXBLOCK) per = 128;
            g->ch = pcm; g->cap_period = (int)per; g->cap_channels = chn; g->cap_rate = g->sr; g->cap_buffer = (int)buf;
            g->cap_opens++;
            strncpy(g->cap_dev, dev, sizeof(g->cap_dev) - 1);
            g->cap_err[0] = 0;
            A.start(pcm);
            return 1;
        }
    }
    return 0;
}

static void *capturer(void *arg) {
    Eng *g = (Eng *)arg;
    g->rt_cap = set_rt(52);
    float fbuf[MAXBLOCK * 2];
    while (g->run) {
        if (g->cap_change) { c_close(g); g->cap_change = 0; }
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
        float pk = 0;
        for (long i = 0; i < r; i++) {
            if (chn == 2) { fbuf[2 * i] = g->cbuf[2 * i] / 32768.0f; fbuf[2 * i + 1] = g->cbuf[2 * i + 1] / 32768.0f; }
            else { fbuf[2 * i] = fbuf[2 * i + 1] = g->cbuf[i] / 32768.0f; }
            pk = fmaxf(pk, fmaxf(fabsf(fbuf[2 * i]), fabsf(fbuf[2 * i + 1])));
        }
        g->cap_raw_pk = fmaxf(g->cap_raw_pk * 0.97f, pk);     /* raw USB level, before Input gain */
        g->cap_frames += (double)r;
        ring_push(g, fbuf, (int)r);
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
        if (!g->ph && !p_open(g)) { usleep(500000); continue; }
        double c0 = cpu_now();
        pthread_mutex_lock(&g->lock);
        ring_pull(g, g->L, g->R, n);
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
    strncpy(g->cap_devs, cap_devs ? cap_devs : "", sizeof(g->cap_devs) - 1);
    g->cap_devs[sizeof(g->cap_devs) - 1] = 0;
    g->cap_change = 1;
}
API void eng_set_buffer(Eng *g, int frames) { g->want_frames = frames; }
API const char *eng_device(Eng *g) { return g->play_dev; }
API const char *eng_cap_device(Eng *g) { return g->cap_dev; }
API const char *eng_error(Eng *g) { return g->err; }
API const char *eng_cap_error(Eng *g) { return g->cap_err; }
API int bafx_version(void) { return 1; }
