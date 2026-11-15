/*
 * bsdsp - native DSP for BrickSampler (voice playback + FX chain).
 * Mirrors the NumPy engine in bricksampler.py sample for sample, so either can be used.
 * Build: see build.sh (zig cc cross-compiles for the Brick, Windows and Linux).
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
       P_VSIZE, P_VDAMP, P_VMIX, P_MASTER, P_DRV_ON, P_FLT_ON, P_DLY_ON, P_REV_ON, P_COUNT };

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
    /* click-free switching (version 5): S-curve fades per section, delay/reverb tails ring out */
    float x_drv, x_flt, g_dly, g_rev;   /* fade positions 0..1 */
    int flt_cur;                        /* filter type running now (0 = none) */
    int dly_act, rev_act;               /* still producing a tail */
    int dly_quiet, rev_quiet;           /* samples spent under -90 dB */
    float *dry;                         /* scratch copy of the input */
    int dry_n;
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
    f->dry = (float *)calloc(2 * 4096, sizeof(float));
    f->dry_n = 4096;
    return f;
}

API void fx_destroy(FX *f) {
    if (!f) return;
    for (int c = 0; c < 2; c++) {
        free(f->dbuf[c]);
        for (int i = 0; i < NCOMB; i++) free(f->comb[c][i].buf);
        for (int i = 0; i < NAP; i++) free(f->ap[c][i].buf);
    }
    free(f->dry);
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
    f->dly_act = f->rev_act = 0;
    f->g_dly = f->g_rev = 0;
}

static inline float scurve(float x) { return x * x * (3 - 2 * x); }

/* Move a fade position toward its target over `ms` milliseconds (one step per sample). */
static inline float fade_step(float x, int on, float step) {
    if (on) { x += step; return x > 1 ? 1 : x; }
    x -= step; return x < 0 ? 0 : x;
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

/* Process one stereo block in place. p holds P_COUNT parameters (see enum). */
API void fx_process(FX *f, float *L, float *R, int n, const float *p) {
    float *ch[2] = {L, R};
    float sr = f->sr;
    if (n > f->dry_n) n = f->dry_n;
    float *dL = f->dry, *dR = f->dry + f->dry_n;
    /* ---- DRIVE section (drive, bitcrush, downsample): fades in and out over 10 ms */
    float dr = p[P_DRIVE];
    int bits = (int)p[P_BITS], k = (int)p[P_DS];
    int drv_want = p[P_DRV_ON] > 0.5f && (dr > 0.001f || bits < 16 || k > 1);
    if (drv_want || f->x_drv > 0) {
        memcpy(dL, L, n * sizeof(float)); memcpy(dR, R, n * sizeof(float));
        if (dr > 0.001f) {
            float g = 1 + 30 * dr * dr, comp = 0.4f + 0.6f / g;
            for (int c = 0; c < 2; c++)
                for (int i = 0; i < n; i++) ch[c][i] = tanhf(ch[c][i] * g) * comp;
        }
        if (bits < 16) {
            float q = (float)(1 << (bits - 1));
            for (int c = 0; c < 2; c++)
                for (int i = 0; i < n; i++) ch[c][i] = roundf(ch[c][i] * q) / q;
        }
        if (k > 1) {
            int cnt = f->ds_count;
            for (int i = 0; i < n; i++) {
                if (cnt == 0) { f->ds_hold[0] = L[i]; f->ds_hold[1] = R[i]; }
                L[i] = f->ds_hold[0]; R[i] = f->ds_hold[1];
                if (++cnt >= k) cnt = 0;
            }
            f->ds_count = cnt;
        }
        float st = 1.0f / (0.010f * sr), x = f->x_drv;
        for (int i = 0; i < n; i++) {
            x = fade_step(x, drv_want, st);
            float w = scurve(x);
            L[i] = dL[i] + (L[i] - dL[i]) * w;
            R[i] = dR[i] + (R[i] - dR[i]) * w;
        }
        f->x_drv = x;
    }
    /* ---- FILTER: on/off and type changes fade out to dry, switch while silent, fade back in (10 ms) */
    int ft = p[P_FLT_ON] > 0.5f ? (int)p[P_FTYPE] : 0;
    if (ft || f->flt_cur || f->x_flt > 0) {
        float st = 1.0f / (0.010f * sr), x = f->x_flt;
        int i0 = 0;
        while (i0 < n) {
            if (f->flt_cur != ft && x <= 0) {          /* silent: change over */
                f->flt_cur = ft;
                f->z1[0] = f->z1[1] = f->z2[0] = f->z2[1] = 0;
                if (ft) set_filter(f, ft, p[P_CUTOFF], p[P_RESO]);
            }
            if (f->flt_cur && (f->flt_cur != f->f_type || fabsf(p[P_CUTOFF] - f->f_cut) > 0.05f
                               || fabsf(p[P_RESO] - f->f_reso) > 0.0005f))
                set_filter(f, f->flt_cur, p[P_CUTOFF], p[P_RESO]);
            int want = f->flt_cur != 0 && f->flt_cur == ft;
            /* process until the fade reaches zero (when a change is waiting) or the block ends */
            int i = i0;
            for (; i < n; i++) {
                x = fade_step(x, want, st);
                float w = scurve(x);
                for (int c = 0; c < 2; c++) {
                    float in = ch[c][i], out = in;
                    if (f->flt_cur) {
                        out = f->b0 * in + f->z1[c];
                        f->z1[c] = f->b1 * in - f->a1 * out + f->z2[c];
                        f->z2[c] = f->b2 * in - f->a2 * out;
                    }
                    ch[c][i] = in + (out - in) * w;
                }
                if (!want && x <= 0 && f->flt_cur != ft) { i++; break; }
            }
            i0 = i;
        }
        f->x_flt = x;
        f->f_on = f->flt_cur != 0;
    }
    /* ---- DELAY: switching off closes only the input (30 ms), the echoes ring out */
    float mix = p[P_DMIX];
    int dly_want = p[P_DLY_ON] > 0.5f && mix > 0.0005f;
    if (dly_want) f->dly_act = 1;
    if (f->dly_act) {
        float target = clampf(p[P_DTIME] * f->sr, 16.0f, (float)(DSIZE - 4));
        float gst = 1.0f / (0.030f * sr), gx = f->g_dly, peak = 0;
        if (f->dcur < 0) f->dcur = target;
        float c = 0.85f * (1 - p[P_DTONE]), fb = fminf(p[P_DFB], 0.95f);
        int ping = p[P_DPING] > 0.5f;
        float *b0 = f->dbuf[0], *b1 = f->dbuf[1];
        int w = f->dw;
        for (int i = 0; i < n; i++) {
            f->dcur += (target - f->dcur) * 0.0004f;          /* glide: no clicks when the time changes */
            float rp = (float)w - f->dcur;
            int r0 = (int)floorf(rp);
            float fr = rp - r0;
            int ia = r0 & DMASK, ib = (r0 + 1) & DMASK;
            float rl = b0[ia] + (b0[ib] - b0[ia]) * fr;
            float rr = b1[ia] + (b1[ib] - b1[ia]) * fr;
            f->tone[0] = c * f->tone[0] + (1 - c) * rl;
            f->tone[1] = c * f->tone[1] + (1 - c) * rr;
            float wl = f->tone[0], wr = f->tone[1];
            gx = fade_step(gx, dly_want, gst);
            float gin = scurve(gx), il = L[i] * gin, ir = R[i] * gin;
            if (ping) {
                b0[w] = 0.5f * (il + ir) + fb * wr;
                b1[w] = fb * wl;
            } else {
                b0[w] = il + fb * wl;
                b1[w] = ir + fb * wr;
            }
            L[i] += mix * wl;
            R[i] += mix * wr;
            float a = fabsf(wl) + fabsf(wr);
            if (a > peak) peak = a;
            w = (w + 1) & DMASK;
        }
        f->dw = w;
        f->g_dly = gx;
        if (!dly_want) {                     /* stop once the tail has been silent for delay time + 0.3 s */
            f->dly_quiet = peak * mix < 3.2e-5f && gx <= 0 ? f->dly_quiet + n : 0;
            if (f->dly_quiet > (int)(f->dcur + 0.3f * sr)) {
                f->dly_act = 0; f->dly_quiet = 0; f->dcur = -1;
                memset(f->dbuf[0], 0, DSIZE * sizeof(float)); memset(f->dbuf[1], 0, DSIZE * sizeof(float));
                f->tone[0] = f->tone[1] = 0;
            }
        } else f->dly_quiet = 0;
    }
    /* ---- REVERB (Freeverb topology): switching off closes only the input (80 ms), the tail rings out */
    float vm = p[P_VMIX];
    int rev_want = p[P_REV_ON] > 0.5f && vm > 0.0005f;
    if (rev_want) f->rev_act = 1;
    if (f->rev_act) {
        float fb = p[P_VSIZE] * 0.28f + 0.7f, d1 = p[P_VDAMP] * 0.4f, d0 = 1 - d1, wet = vm * 3.0f;
        float gst = 1.0f / (0.080f * sr), gx = f->g_rev, peak = 0;
        for (int i = 0; i < n; i++) {
            gx = fade_step(gx, rev_want, gst);
            float in = (L[i] + R[i]) * 0.015f * scurve(gx);
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
                float a = fabsf(out);
                if (a > peak) peak = a;
            }
        }
        f->g_rev = gx;
        if (!rev_want) {
            f->rev_quiet = peak * wet < 3.2e-5f && gx <= 0 ? f->rev_quiet + n : 0;
            if (f->rev_quiet > (int)(0.4f * sr)) {
                f->rev_act = 0; f->rev_quiet = 0;
                for (int c = 0; c < 2; c++) {
                    for (int j = 0; j < NCOMB; j++) { memset(f->comb[c][j].buf, 0, f->comb[c][j].len * sizeof(float)); f->comb[c][j].store = 0; }
                    for (int j = 0; j < NAP; j++) memset(f->ap[c][j].buf, 0, f->ap[c][j].len * sizeof(float));
                }
            }
        } else f->rev_quiet = 0;
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
enum { EV_ON = 1, EV_OFF, EV_CC, EV_BEND, EV_ALLOFF };

typedef struct { const float *d; int n; float p[SP_COUNT]; } ESlot;

/* ---- Wave sequences (version 4), after the Korg Wavestation: a list of steps, each one a sample
 * region with its own duration (or GATE), crossfade into the next step, level and tuning. A range of
 * steps can loop forwards or back-and-forth, a number of times or for ever. */
#define WS_MAX 64
enum { WH_ON, WH_NSTEPS, WH_START, WH_LSTART, WH_LEND, WH_MODE, WH_REPEATS, WH_KEYSYNC, WH_VELSTART, WH_WHEEL,
       WH_COUNT };
enum { WP_DUR, WP_XF, WP_LEVEL, WP_SEMIS, WP_START, WP_END, WP_LOOP, WP_COUNT };   /* dur/xf in samples, dur<0 = GATE */
typedef struct { const float *d; int n, s, e, loop; double dur, xf; float level, ratio; } WStep;
typedef struct { int on, nsteps, start, lstart, lend, mode, repeats, keysync, velstart, wheel; WStep st[WS_MAX]; } WSeq;
typedef struct {
    int cur, nxt, dir, reps, ndir, nreps, xing, ended;
    double t, pa, pb;               /* time in the current step; read positions in the current / next step */
} WState;

typedef struct {
    int slot, note, keytrack, s, e, reverse, oneshot, loop, stage, done, sustained;
    double pos;
    int ls, le;                     /* loop in playback coordinates (0 = region start in play direction) */
    double xf;                      /* crossfade length in samples */
    float amp, gl, gr, level, rel_rate;
    int ws;                         /* 1: this voice plays the slot's wave sequence */
    WState w;
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

typedef struct {
    int sr, block, kit, max_voices, kit_base, sustain;
    float bend, bend_range;
    ESlot slot[E_SLOTS];
    WSeq seq[E_SLOTS];
    WState free_run[E_SLOTS];      /* the shared clock of sequences with key sync off */
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
#if !defined(_WIN32)
    pthread_t th;
    volatile int run, paused, closed, want_frames, opened_frames, buffer_frames, period_frames, rt, open_fails;
    void *pcmh;
    char devs[256], dev[64], err[160];
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
/* A slot's wave sequence. hdr: WH_COUNT floats; datas/ns: each step's sample; sp: nsteps * WP_COUNT.
 * Python keeps the sample arrays alive. Playing voices keep going with the new settings. */
API void eng_set_seq(Eng *g, int idx, const float *hdr, const float **datas, const int *ns, const float *sp) {
    if (idx < 0 || idx >= E_SLOTS) return;
    LOCK(g->lock);
    WSeq *q = &g->seq[idx];
    int n = (int)hdr[WH_NSTEPS];
    if (n < 0) n = 0;
    if (n > WS_MAX) n = WS_MAX;
    q->nsteps = n;
    q->on = hdr[WH_ON] > 0.5f && n > 0;
    q->start = (int)hdr[WH_START]; q->lstart = (int)hdr[WH_LSTART]; q->lend = (int)hdr[WH_LEND];
    q->mode = (int)hdr[WH_MODE]; q->repeats = (int)hdr[WH_REPEATS]; q->keysync = hdr[WH_KEYSYNC] > 0.5f;
    q->velstart = hdr[WH_VELSTART] > 0.5f; q->wheel = hdr[WH_WHEEL] > 0.5f;
    if (q->start >= n) q->start = n ? n - 1 : 0;
    for (int i = 0; i < n; i++) {
        WStep *t = &q->st[i];
        const float *p = sp + i * WP_COUNT;
        t->d = datas[i]; t->n = t->d ? ns[i] : 0;
        t->s = (int)((double)p[WP_START] * t->n + 1e-3);
        t->e = (int)((double)p[WP_END] * t->n + 1e-3);
        if (t->e > t->n) t->e = t->n;
        if (t->s > t->e - 2) t->s = t->e - 2;
        if (t->s < 0) t->s = 0;
        t->dur = p[WP_DUR]; t->xf = p[WP_XF] > 0 ? p[WP_XF] : 0;
        t->level = p[WP_LEVEL]; t->ratio = exp2f(p[WP_SEMIS] / 12.0f); t->loop = p[WP_LOOP] > 0.5f;
    }
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

/* Which step follows `cur`, counting loop repeats and turning round in back-and-forth loops.
 * -1 = the sequence has ended (the last step then holds). */
static int ws_next(const WSeq *q, int cur, int *dir, int *reps) {
    int n = q->nsteps, ls = q->lstart, le = q->lend;
    if (ls < 0) ls = 0;
    if (le >= n) le = n - 1;
    int looping = ls <= le && cur >= ls && cur <= le && (q->repeats == 0 || *reps < q->repeats);
    if (looping && q->mode == 1) {                         /* back and forth */
        if (ls == le) { (*reps)++; return cur; }
        if (*dir > 0 && cur >= le) { *dir = -1; return cur - 1; }
        if (*dir < 0 && cur <= ls) {
            (*reps)++; *dir = 1;
            return cur + 1;
        }
        return cur + *dir;
    }
    if (*dir < 0) {                                        /* repeats used up while going backwards */
        if (cur <= 0) { *dir = 1; return n > 1 ? 1 : -1; }
        return cur - 1 >= ls ? cur - 1 : (*dir = 1, cur + 1 < n ? cur + 1 : -1);
    }
    if (looping && cur >= le) { (*reps)++; return ls; }
    return cur + 1 < n ? cur + 1 : -1;
}

static void ws_init(const WSeq *q, WState *w, int vel) {
    memset(w, 0, sizeof(*w));
    int st = q->start;
    if (q->velstart && q->nsteps > st) st += (int)(vel / 128.0f * (q->nsteps - st));
    if (st >= q->nsteps) st = q->nsteps - 1;
    if (st < 0) st = 0;
    w->cur = st; w->dir = 1; w->nxt = -2;
}

static inline float ws_read(const WStep *t, double *pos, double inc) {
    int len = t->e - t->s;
    if (!t->d || len < 2) return 0.0f;
    double x = *pos;
    *pos += inc;
    if (x >= len - 1) {
        if (!t->loop) return 0.0f;
        x = fmod(x, (double)(len - 1));
        if (*pos >= len - 1) *pos = fmod(*pos, (double)(len - 1));
    }
    int i0 = (int)x;
    float fr = (float)(x - i0);
    float a = t->d[t->s + i0], b = t->d[t->s + i0 + 1];
    return a + (b - a) * fr;
}

/* One sample of a sequence (or just its clock when out == NULL): plays the current step and, during
 * a crossfade, the next one; moves on when the step's time is up (GATE steps: when the key is let go). */
static float ws_tick(const WSeq *q, WState *w, double inc, int released, float mw, int audio) {
    int n = q->nsteps;
    if (n <= 0) return 0.0f;
    if (q->wheel) {                                        /* the mod wheel scans the sequence by hand */
        float f = mw * (n - 1);
        int a = (int)f, b = a + 1 < n ? a + 1 : a;
        float fr = f - a;
        if (a != w->cur) { w->cur = a; }
        const WStep *A = &q->st[a], *B = &q->st[b];
        float sa = ws_read(A, &w->pa, inc * A->ratio) * A->level;
        float sb = b != a ? ws_read(B, &w->pb, inc * B->ratio) * B->level : 0.0f;
        return audio ? sa * (1 - fr) + sb * fr : 0.0f;
    }
    if (w->cur >= n) w->cur = n - 1;
    const WStep *A = &q->st[w->cur];
    double dur = A->dur < 0 ? (released ? w->t : 1e18) : A->dur;
    if (w->ended) dur = 1e18;
    double xf = A->xf < dur ? A->xf : dur;
    if (!w->ended && !w->xing && w->t >= dur - xf) {       /* time to bring in the next step */
        w->ndir = w->dir; w->nreps = w->reps;
        w->nxt = ws_next(q, w->cur, &w->ndir, &w->nreps);
        if (w->nxt < 0 || w->nxt >= n) { w->ended = 1; dur = 1e18; }
        else { w->xing = 1; w->pb = 0; }
    }
    float out = ws_read(A, &w->pa, inc * A->ratio) * A->level;
    if (w->xing) {
        const WStep *B = &q->st[w->nxt];
        float wgt = xf > 0 ? (float)((w->t - (dur - xf)) / xf) : 1.0f;
        if (wgt > 1) wgt = 1;
        if (wgt < 0) wgt = 0;
        float b = ws_read(B, &w->pb, inc * B->ratio) * B->level;
        out = out * (1 - wgt) + b * wgt;
    }
    w->t += 1;
    if (w->xing && w->t >= dur) {                          /* the next step takes over */
        w->t = w->t - dur + xf;
        w->cur = w->nxt; w->dir = w->ndir; w->reps = w->nreps;
        w->pa = w->pb; w->xing = 0;
    }
    return audio ? out : 0.0f;
}

static int ws_render(Eng *g, EVoice *v, double inc, float e0, float e1, float *L, float *R, int n) {
    const WSeq *q = &g->seq[v->slot];
    if (!q->on) return 1;
    float de = (e1 - e0) / n, amp = v->amp;
    int rel = v->stage >= 3;
    for (int k = 0; k < n; k++) {
        float o = ws_tick(q, &v->w, inc, rel, g->mw, 1) * (e0 + de * (k + 1)) * amp;
        L[k] += o * v->gl;
        R[k] += o * v->gr;
    }
    return 0;
}

static void handle(Eng *g, EEv *ev) {
    int a = ev->a, b = ev->b;
    if (ev->kind == EV_ON) {
        int idx, keytrack;
        if (!g->kit) { idx = 0; keytrack = 1; }
        else {
            int pad = a - g->kit_base;
            if (pad < 0 || pad >= E_SLOTS - 1) return;
            idx = 1 + pad; keytrack = 0;
        }
        int seq = g->seq[idx].on;
        if (!seq && (!g->slot[idx].d || g->slot[idx].n < 2)) return;
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
        if (seq) {
            EVoice *v = &g->v[g->nv - 1];
            v->ws = 1; v->oneshot = 0;
            const WSeq *q = &g->seq[idx];
            if (q->keysync) ws_init(q, &v->w, b);
            else {
                v->w = g->free_run[idx];                   /* join the running sequence */
                if (v->w.cur >= q->nsteps) ws_init(q, &v->w, b);
            }
        }
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

/* Same ADSR as Voice._env in Python: advance n samples, return level at start and end. */
static void v_env(Eng *g, EVoice *v, int n, float *e0, float *e1) {
    const float *p = g->slot[v->slot].p;
    float t = (float)n / g->sr, lvl = v->level;
    int st = v->stage;
    while (t > 1e-12f && st < 4) {
        if (st == 0) {
            float rate = 1.0f / (p[SP_ATTACK] > 0.0005f ? p[SP_ATTACK] : 0.0005f);
            float need = (1.0f - lvl) / rate;
            if (need <= t) { lvl = 1; t -= need; st = 1; } else { lvl += rate * t; t = 0; }
        } else if (st == 1) {
            float sus = p[SP_SUSTAIN];
            if (lvl <= sus) { st = 2; continue; }
            float rate = (1.0f - sus) / (p[SP_DECAY] > 0.001f ? p[SP_DECAY] : 0.001f);
            float need = (lvl - sus) / rate;
            if (need <= t) { lvl = sus; t -= need; st = 2; } else { lvl -= rate * t; t = 0; }
        } else if (st == 2) {
            lvl = p[SP_SUSTAIN];
            if (lvl <= 1e-5f) { lvl = 0; st = 4; }
            t = 0;
        } else {
            float need = lvl / v->rel_rate;
            if (need <= t) { lvl = 0; st = 4; } else { lvl -= v->rel_rate * t; t = 0; }
        }
    }
    *e0 = v->level; *e1 = lvl;
    v->level = lvl; v->stage = st;
}

static inline float rd(const float *d, int dn, const EVoice *v, int i) {
    int a = v->reverse ? v->e - 1 - i : v->s + i;
    if (a < 0) a = 0; else if (a >= dn) a = dn - 1;
    return d[a];
}

/* One voice into L/R (added). Like voice_render, plus a loop between ls and le with a linear
 * crossfade: in the last xf samples before le, the material just before ls fades in, so the
 * jump back to ls is seamless. */
static int v_render(Eng *g, EVoice *v, double inc, float e0, float e1, float *L, float *R, int n) {
    const float *d = g->slot[v->slot].d;
    int dn = g->slot[v->slot].n, len = v->e - v->s;
    if (len < 2 || !d) return 1;
    double ps = v->pos, ls = v->ls, le = v->le, Lp = le - ls, X = v->xf;
    int iL = v->le - v->ls;
    float de = (e1 - e0) / n, amp = v->amp;
    int done = 0;
    for (int k = 0; k < n; k++) {
        double x = ps + inc * k;
        if (v->loop) {
            if (x >= le) x = ls + fmod(x - ls, Lp);
        } else if (x >= len - 1) { done = 1; break; }
        int i0 = (int)x, i1 = i0 + 1;
        float fr = (float)(x - i0);
        if (v->loop && i1 >= v->le) i1 -= iL;
        float a = rd(d, dn, v, i0), b = rd(d, dn, v, i1);
        float smp = a + (b - a) * fr;
        if (v->loop && X > 0 && x >= le - X) {
            float w = (float)((x - (le - X)) / X);
            float c = rd(d, dn, v, i0 - iL), e = rd(d, dn, v, i0 - iL + 1);
            smp = smp * (1 - w) + (c + (e - c) * fr) * w;
        }
        float o = smp * (e0 + de * (k + 1)) * amp;
        L[k] += o * v->gl;
        R[k] += o * v->gr;
    }
    ps += inc * n;
    if (v->loop && ps >= le) ps = ls + fmod(ps - ls, Lp);
    v->pos = ps;
    return done;
}

static float lfo_wave(Eng *g, double ph) {
    switch ((int)g->lfop[LP_WAVE]) {
        case 1: return 1.0f - 4.0f * fabsf((float)fmod(ph + 0.25, 1.0) - 0.5f);   /* triangle */
        case 2: return (float)(2.0 * ph - 1.0);                                  /* saw up */
        case 3: return ph < 0.5 ? 1.0f : -1.0f;                                   /* square */
        case 4: return g->lfo_sh;                                                 /* sample & hold */
        default: return sinf((float)(6.283185307179586 * ph));                    /* sine */
    }
}

/* Render one block into g->L / g->R. Returns 1 if anything was computed, 0 when idle. */
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
    if (g->nv == 0) {
        float tail = g->fxp[P_DMIX] > 0.0005f ? g->fxp[P_DTIME] * 1.5f : 0.0f;
        if (tail < 0.5f) tail = 0.5f;
        if ((float)g->quiet_blocks * n / g->sr > tail) {
            if (!g->idle) { fx_clear(g->fx); g->idle = 1; }
            g->peak[0] = g->peak[1] = 0;
            return 0;
        }
    }
    g->idle = 0;
    memset(g->L, 0, sizeof(float) * n);
    memset(g->R, 0, sizeof(float) * n);
    /* LFO: one global oscillator, read at the start and end of the block */
    float *lp = g->lfop;
    float sc = 1.0f;
    if (lp[LP_DELAY] > 0.001f) { sc = (float)(g->lfo_t / lp[LP_DELAY]); if (sc > 1) sc = 1; }
    if (lp[LP_MODWHEEL] > 0.5f) sc *= g->mw;
    float l0 = lfo_wave(g, g->lfo_phase);
    g->lfo_phase += lp[LP_RATE] * (double)n / g->sr;
    if (g->lfo_phase >= 1.0) {
        g->lfo_phase -= floor(g->lfo_phase);
        g->rng = g->rng * 1664525u + 1013904223u;
        g->lfo_sh = (float)(g->rng >> 8) / 8388608.0f - 1.0f;
    }
    g->lfo_t += (double)n / g->sr;
    float l1 = lfo_wave(g, g->lfo_phase);
    g->lfo_val = l0; g->lfo_scale = sc;
    float pmod = lp[LP_PITCH] * sc * l0;
    float vd = lp[LP_VOLUME] * sc;
    float g0 = 1.0f - vd * (1.0f - l0) * 0.5f, g1 = 1.0f - vd * (1.0f - l1) * 0.5f;

    for (int si = 0; si < E_SLOTS; si++) {             /* free-running sequences keep their own clock */
        const WSeq *q = &g->seq[si];
        if (!q->on || q->keysync || q->wheel) continue;
        WState *w = &g->free_run[si];
        if (w->cur >= q->nsteps || (w->dir == 0)) ws_init(q, w, 0);
        for (int j = 0; j < n; j++) ws_tick(q, w, 1.0, 0, g->mw, 0);
    }
    int k = 0;
    for (int i = 0; i < g->nv; i++) {
        EVoice *v = &g->v[i];
        const float *p = g->slot[v->slot].p;
        float semis = p[SP_TUNE] + p[SP_FINE] / 100.0f + pmod;
        if (v->keytrack) semis += v->note - p[SP_ROOT] + g->bend;
        double inc = pow(2.0, semis / 12.0);
        float e0, e1;
        v_env(g, v, n, &e0, &e1);
        int done = v->ws ? ws_render(g, v, inc, e0 * g0, e1 * g1, g->L, g->R, n)
                         : v_render(g, v, inc, e0 * g0, e1 * g1, g->L, g->R, n);
        if (!(done || v->stage >= 4)) g->v[k++] = *v;
    }
    g->nv = k;
    const float *fxp = g->fxp;
    if (lp[LP_CUTOFF] > 0.0005f && g->fxp[P_FTYPE] >= 1 && sc > 0) {
        memcpy(g->fxl, g->fxp, sizeof(g->fxl));
        g->fxl[P_CUTOFF] = clampf(g->fxp[P_CUTOFF] * exp2f(4.0f * lp[LP_CUTOFF] * sc * l0), 20.0f, 18000.0f);
        fxp = g->fxl;
    }
    fx_process(g->fx, g->L, g->R, n, fxp);
    float pk[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        float a = fabsf(g->L[i]), b = fabsf(g->R[i]);
        if (a > pk[0]) pk[0] = a;
        if (b > pk[1]) pk[1] = b;
    }
    g->quiet_blocks = (g->nv || pk[0] > 2e-4f || pk[1] > 2e-4f) ? 0 : g->quiet_blocks + 1;
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
        if (v->ws) {                                       /* sequence voices: slot + 100, step + fraction */
            const WSeq *q = &g->seq[v->slot];
            double fr = 0;
            if (q->nsteps && v->w.cur < q->nsteps) {
                double d = q->st[v->w.cur].dur;
                fr = d > 0 && !v->w.ended ? v->w.t / d : 0;
                if (fr > 0.999) fr = 0.999;
            }
            o[n++] = (float)(v->slot + 100);
            o[n++] = (float)(v->w.cur + fr);
            continue;
        }
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

#if !defined(_WIN32)
/* ---- ALSA writer thread (libasound loaded at run time, so the library builds without its headers) */
typedef int (*f_open)(void **, const char *, int, int);
typedef int (*f_setp)(void *, int, int, unsigned, unsigned, int, unsigned);
typedef int (*f_getp)(void *, unsigned long *, unsigned long *);
typedef long (*f_writei)(void *, const void *, unsigned long);
typedef int (*f_recover)(void *, int, int);
typedef int (*f_pcm)(void *);
typedef const char *(*f_strerror)(int);
static struct { f_open open; f_setp setp; f_getp getp; f_writei writei; f_recover recover; f_pcm drop, close; f_strerror strerror; } A;

static int alsa_load(void) {
    static int done = 0;
    if (done) return done > 0;
    void *h = dlopen("libasound.so.2", RTLD_NOW);
    if (!h) { done = -1; return 0; }
    A.open = (f_open)dlsym(h, "snd_pcm_open");
    A.setp = (f_setp)dlsym(h, "snd_pcm_set_params");
    A.getp = (f_getp)dlsym(h, "snd_pcm_get_params");
    A.writei = (f_writei)dlsym(h, "snd_pcm_writei");
    A.recover = (f_recover)dlsym(h, "snd_pcm_recover");
    A.drop = (f_pcm)dlsym(h, "snd_pcm_drop");
    A.close = (f_pcm)dlsym(h, "snd_pcm_close");
    A.strerror = (f_strerror)dlsym(h, "snd_strerror");
    done = (A.open && A.setp && A.getp && A.writei && A.recover && A.drop && A.close && A.strerror) ? 1 : -1;
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
    while (g->run) {
        if (g->paused) {
            a_close(g);
            g->closed = 1;
            usleep(10000);
            continue;
        }
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
