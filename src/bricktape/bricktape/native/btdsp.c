/*
 * btdsp - native DSP for BrickTape (4-track tape, FX, audio in/out). Built on BrickLooper's bsdsp.
 * Voices and FX mirror the NumPy engine in bricklooper.py sample for sample, so either can be used.
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
API int btdsp_version(void) { return 1; }


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

/* ---- tape (BrickTape): 4 mono tracks on one linear tape, like a cassette 4-track.
 * The tape position P counts frames. The block rendered now is heard comp_int frames later, and the input
 * arriving now was played comp_ext frames ago. So input frame k of this block belongs at P - comp_ext + k,
 * and a punch made "now" falls at P - comp_int (what the player was hearing when they pressed). */
#define NTT 4
#define TAPE_MIN 10                  /* minutes of tape per track (4 x 10 min x 16 bit = 230 MB, plus the undo copy) */
#define TFOREVER ((long long)1 << 62)
#define RING (1 << 15)               /* input ring, stereo frames */
#define TAPN (1 << 16)               /* mono tap of the input */
#define ENV_CH 2048                  /* frames per overview cell (the lanes on screen) */
enum { TS_STOP, TS_PLAY, TS_FF, TS_REW };
enum { TC_PLAY = 1, TC_STOP, TC_REC, TC_FF, TC_REW, TC_PAUSE, TC_LOCATE, TC_UNDO, TC_CLEAR };
enum { TP_COMPI, TP_COMPE, TP_MON, TP_INGAIN, TP_OUTVOL, TP_MET, TP_BPM, TP_BEATS, TP_CLICK, TP_WIND,
       TP_ARM, TP_IN = TP_ARM + NTT, TP_VOL = TP_IN + NTT, TP_PAN = TP_VOL + NTT, TP_MUTE = TP_PAN + NTT,
       TP_SCRUB = TP_MUTE + NTT, TP_COUNT };

typedef struct {
    short *d, *u;                   /* the track (16-bit mono) and the copy of what the last take replaced */
    unsigned char *env;             /* peak per ENV_CH frames, 0..255 */
    long long hi, uhi;              /* end of the recorded part (and before the last take) */
    long long rin, rout;            /* record window on the tape (rin < 0: none, rout = TFOREVER: still open) */
    long long blo, bhi;             /* the undo copy covers [blo, bhi) */
    long long ecell;                /* last overview cell this take started again */
    int take, ver;
    float pg, pk;                   /* playback fade (0 while it records), level meter */
} TTrack;

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
    /* tape */
    TTrack tt[NTT];
    long long G, P, maxt, ncell, click_p;
    int ts, tpaused, wts, wpaused, recm, engaged, take_no, undo_take, undo_st, click_on;
    float tg, sg, sz[2];            /* tape gain, scrub gain and its low-pass state */
    double wsp;                     /* winding speed now: starts slow, speeds up to TP_WIND while it winds */
    float tp[TP_COUNT];
    float *inL, *inR, *pL, *pR;
    float in_pk[2];
    float *ring;                    /* input frames from the capture thread (or eng_in_push) */
    volatile unsigned rh, rt_;      /* write / read counters */
    int ring_target, in_under, in_drop, in_started;
    double rfrac, fill_f, drift_i, ratio;     /* drift resampler: input frames read per output frame */
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
    /* tape: TAPE_MIN minutes per track. calloc'd memory is only really used once it is written. */
    g->maxt = (long long)sr * 60 * TAPE_MIN;
    g->ncell = g->maxt / ENV_CH + 1;
    for (int i = 0; i < NTT; i++) {
        TTrack *t = &g->tt[i];
        t->d = (short *)calloc((size_t)g->maxt, sizeof(short));
        t->u = (short *)calloc((size_t)g->maxt, sizeof(short));
        t->env = (unsigned char *)calloc((size_t)g->ncell, 1);
        t->rin = -1; t->take = -1; t->pg = 1;
        g->tp[TP_VOL + i] = 0.8f; g->tp[TP_IN + i] = 2;
    }
    g->undo_take = -1;
    g->inL = (float *)calloc(block, sizeof(float)); g->inR = (float *)calloc(block, sizeof(float));
    g->pL = (float *)calloc(block, sizeof(float)); g->pR = (float *)calloc(block, sizeof(float));
    g->ring = (float *)calloc(RING * 2, sizeof(float));
    g->tap = (float *)calloc(TAPN, sizeof(float));
    g->tp[TP_INGAIN] = 1; g->tp[TP_OUTVOL] = 1; g->tp[TP_MON] = 1; g->tp[TP_WIND] = 30; g->tp[TP_BPM] = 120; g->tp[TP_BEATS] = 4;
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

static void tape_cmd(Eng *g, int cmd, int arg);

static void handle(Eng *g, EEv *ev) {
    int a = ev->a, b = ev->b;
    if (ev->kind == EV_LOOP) { tape_cmd(g, a, b); return; }
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

/* The USB device and the Brick run on separate clocks (a USB synth or interface can be 0.2% fast or slow). Instead of skipping
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
    if (!g->in_started && avail >= (unsigned)(target + n)) {
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

/* =========================================================================================== tape */

static long long tcomp_i(Eng *g) { return (long long)g->tp[TP_COMPI]; }
static long long tcomp_e(Eng *g) { return g->mic_on ? (long long)g->tp[TP_COMPI] : (long long)g->tp[TP_COMPE]; }

static void env_rebuild(Eng *g, TTrack *t, long long a, long long b) {
    if (a < 0) a = 0;
    if (b > g->maxt) b = g->maxt;
    for (long long c = a / ENV_CH; c * ENV_CH < b && c < g->ncell; c++) {
        long long s = c * ENV_CH, e = s + ENV_CH;
        if (e > g->maxt) e = g->maxt;
        int m = 0;
        for (long long i = s; i < e; i++) { int v = t->d[i] < 0 ? -t->d[i] : t->d[i]; if (v > m) m = v; }
        m >>= 7;
        t->env[c] = (unsigned char)(m > 255 ? 255 : m);
    }
}

static int any_window(Eng *g) {
    for (int i = 0; i < NTT; i++) if (g->tt[i].rin >= 0) return 1;
    return 0;
}

/* A track records while REC is engaged, the tape plays (no change waiting) and the track is armed. */
static int rec_cond(Eng *g, int i) {
    return g->recm && g->ts == TS_PLAY && !g->tpaused && g->wts == TS_PLAY && !g->wpaused && g->tp[TP_ARM + i] > 0.5f;
}

static void open_window(Eng *g, TTrack *t, long long at) {
    if (!g->engaged) { g->engaged = 1; g->take_no++; g->undo_st = 0; }
    if (t->take != g->take_no) {    /* first punch of this take on this track: a new undo copy */
        t->take = g->take_no;
        t->blo = t->bhi = at < 0 ? 0 : at;
        t->uhi = t->hi;
    }
    t->rin = at; t->rout = TFOREVER; t->ecell = -1;
}

/* Input samples src for tape frames w0 .. w0+n into the track, only inside its record window, with 5 ms
 * fades at the punch in and out (old and new sound crossfade there). */
static float tape_write(Eng *g, TTrack *t, long long w0, int n, const float *src) {
    long long F = g->sr / 200;
    long long a = w0 > t->rin ? w0 : t->rin, b = w0 + n;
    if (t->rout != TFOREVER && b > t->rout) b = t->rout;
    if (a < 0) a = 0;
    if (b > g->maxt) b = g->maxt;
    if (b <= a) return 0;
    if (b > t->bhi) {
        memcpy(t->u + t->bhi, t->d + t->bhi, sizeof(short) * (size_t)(b - t->bhi));
        t->bhi = b;
    }
    float pk = 0;
    for (long long p = a; p < b; p++) {
        float r = 1.0f;
        long long e1 = p - t->rin;
        if (e1 < F) r = (float)e1 / (float)F;
        if (t->rout != TFOREVER) {
            long long e2 = t->rout - p;
            if (e2 < F && (float)e2 / (float)F < r) r = (float)e2 / (float)F;
        }
        float s = src[p - w0];
        float old = t->d[p] / 32768.0f;
        float v = clampf(old + (s - old) * r, -0.99997f, 0.99997f);
        t->d[p] = (short)lrintf(v * 32767.0f);
        long long c = p / ENV_CH;
        if (c != t->ecell) { t->ecell = c; t->env[c] = 0; }
        int m = (int)(fabsf(v) * 255.0f);
        if (m > t->env[c]) t->env[c] = (unsigned char)m;
        if (fabsf(s) > pk) pk = fabsf(s);
    }
    if (b > t->hi) t->hi = b;
    t->ver++;
    return pk;
}

static void tape_undo(Eng *g) {
    if (g->engaged || any_window(g) || !g->undo_st) return;
    for (int i = 0; i < NTT; i++) {
        TTrack *t = &g->tt[i];
        if (t->take != g->undo_take || t->bhi <= t->blo) continue;
        for (long long p = t->blo; p < t->bhi; p++) { short x = t->d[p]; t->d[p] = t->u[p]; t->u[p] = x; }
        long long h = t->hi; t->hi = t->uhi; t->uhi = h;
        env_rebuild(g, t, t->blo, t->bhi);
        t->ver++;
    }
    g->undo_st = g->undo_st == 1 ? 2 : 1;
}

static void tape_clear(Eng *g, int i) {
    if (i < 0 || i >= NTT) return;
    TTrack *t = &g->tt[i];
    if (t->rin >= 0) return;
    memset(t->d, 0, sizeof(short) * (size_t)t->hi);
    memset(t->env, 0, (size_t)(t->hi / ENV_CH + 1 < g->ncell ? t->hi / ENV_CH + 1 : g->ncell));
    t->hi = 0; t->take = -1; t->ver++;
}

static void tape_cmd(Eng *g, int cmd, int arg) {
    switch (cmd) {
    case TC_PLAY: g->recm = 0; g->wts = TS_PLAY; g->wpaused = 0; break;    /* from REC: punch out, keep playing */
    case TC_REC: g->recm = 1; g->wts = TS_PLAY; g->wpaused = 0; break;     /* from PLAY: punch in */
    case TC_STOP: g->recm = 0; g->wts = TS_STOP; g->wpaused = 0; break;
    case TC_FF: g->recm = 0; g->wts = g->wts == TS_FF ? TS_STOP : TS_FF; g->wpaused = 0; break;
    case TC_REW: g->recm = 0; g->wts = g->wts == TS_REW ? TS_STOP : TS_REW; g->wpaused = 0; break;
    case TC_PAUSE: if (g->wts == TS_PLAY) g->wpaused = !g->wpaused; break;
    case TC_LOCATE:
        if (g->ts == TS_STOP && g->wts == TS_STOP) {
            long long p = arg < 0 ? 0 : arg;
            g->P = p > g->maxt ? g->maxt : p;
        }
        break;
    case TC_UNDO: tape_undo(g); break;
    case TC_CLEAR: tape_clear(g, arg); break;
    }
}

/* Move the tape one block: record the armed tracks, then mix their playback and the input monitor into
 * g->L / g->R. Returns 1 when there is anything to hear. */
static int tape_process(Eng *g, int n) {
    long long ci = tcomp_i(g), ce = tcomp_e(g);
    float gin = g->tp[TP_INGAIN];
    /* input level per source: L, R, L+R */
    float ipk[3] = {0, 0, 0};
    for (int k = 0; k < n; k++) {
        float a = fabsf(g->inL[k] * gin), b = fabsf(g->inR[k] * gin), m = fabsf(0.5f * (g->inL[k] + g->inR[k]) * gin);
        if (a > ipk[0]) ipk[0] = a;
        if (b > ipk[1]) ipk[1] = b;
        if (m > ipk[2]) ipk[2] = m;
    }
    /* record windows follow REC, the arm switches and the transport */
    for (int i = 0; i < NTT; i++) {
        TTrack *t = &g->tt[i];
        int c = rec_cond(g, i);
        if (c) {
            if (t->rin < 0) open_window(g, t, g->P - ci);
            else if (t->rout != TFOREVER) t->rout = TFOREVER;        /* armed again before it finished */
        } else if (t->rin >= 0 && t->rout == TFOREVER) {
            long long o = g->P - ci;
            t->rout = o > t->rin ? o : t->rin;
        }
        if (t->rin >= 0 && t->rout != TFOREVER && g->P - ce >= t->rout) t->rin = -1;   /* the input caught up */
    }
    /* a change of transport waits for the fade out and for the last of the input to be recorded */
    int moving = g->ts == TS_PLAY && !g->tpaused;
    int pending = g->wts != g->ts || g->wpaused != g->tpaused;
    if (pending && !((moving && (g->tg > 0.0f || any_window(g))) || ((g->ts == TS_FF || g->ts == TS_REW) && g->sg > 0.0f))) {
        g->ts = g->wts; g->tpaused = g->wpaused;
        moving = g->ts == TS_PLAY && !g->tpaused;
        pending = 0;
        if (!moving) for (int i = 0; i < NTT; i++) g->tt[i].rin = -1;
    }
    if (g->engaged && !g->recm && !any_window(g)) { g->engaged = 0; g->undo_take = g->take_no; g->undo_st = 1; }
    float g0 = g->tg, tgt = moving && !pending ? 1.0f : 0.0f;
    float g1 = tgt > g0 ? fminf(1.0f, g0 + 0.25f) : fmaxf(0.0f, g0 - 0.25f);
    g->tg = g1;
    g->click_on = moving;
    g->click_p = g->P;
    int active = moving || g1 > 0.0f || pending;
    static float src[FXMAXN];
    for (int i = 0; i < NTT; i++) {
        TTrack *t = &g->tt[i];
        int sel = (int)g->tp[TP_IN + i];
        if (sel < 0 || sel > 2) sel = 2;
        float pk = g->tp[TP_ARM + i] > 0.5f ? ipk[sel] : 0.0f;   /* armed: the meter shows the input */
        if (moving) {
            float ptgt = (t->rin >= 0 && t->rout == TFOREVER) ? 0.0f : 1.0f;
            float p0 = t->pg, p1 = ptgt > p0 ? fminf(1.0f, p0 + 0.25f) : fmaxf(0.0f, p0 - 0.25f);
            t->pg = p1;
            float vol = g->tp[TP_MUTE + i] > 0.5f ? 0.0f : g->tp[TP_VOL + i];
            float ang = (clampf(g->tp[TP_PAN + i], -1, 1) + 1) * 3.14159265f / 4;
            float gl = cosf(ang) * 1.41421356f, gr = sinf(ang) * 1.41421356f;
            if (vol > 0.0f && (p0 > 0.0f || p1 > 0.0f) && (g0 > 0.0f || g1 > 0.0f)) {
                for (int k = 0; k < n; k++) {
                    long long p = g->P + k;
                    if (p >= g->maxt) break;
                    float w = (float)(k + 1) / (float)n;
                    float s = t->d[p] / 32768.0f * (g0 + (g1 - g0) * w) * (p0 + (p1 - p0) * w) * vol;
                    g->L[k] += s * gl; g->R[k] += s * gr;
                    if (fabsf(s) > pk) pk = fabsf(s);
                }
            }
            if (t->rin >= 0) {
                for (int k = 0; k < n; k++) {
                    float s = sel == 0 ? g->inL[k] : sel == 1 ? g->inR[k] : 0.5f * (g->inL[k] + g->inR[k]);
                    src[k] = s * gin;
                }
                tape_write(g, t, g->P - ce, n, src);
            }
        }
        t->pk = t->pk * 0.85f > pk ? t->pk * 0.85f : pk;
    }
    if (moving) {
        g->P += n;
        if (g->P >= g->maxt) { g->P = g->maxt; g->wts = TS_STOP; g->recm = 0; }
    }
    /* rewind / fast forward: the tracks are heard sped up ("cue"), each output sample the average of the
     * stretch of tape it passes over, low-passed and quieter. Fades in and out like play does. */
    int winding = !moving && !g->tpaused && (g->ts == TS_FF || g->ts == TS_REW) && g->wts == g->ts;
    {
        float s0 = g->sg, stgt = winding && g->tp[TP_SCRUB] > 0.5f ? 1.0f : 0.0f;
        float s1 = stgt > s0 ? fminf(1.0f, s0 + 0.1f) : fmaxf(0.0f, s0 - 0.25f);
        g->sg = s1;
        if ((s0 > 0.0f || s1 > 0.0f) && (g->ts == TS_FF || g->ts == TS_REW)) {
            double sp = winding ? g->wsp : 0.0;        /* fading out: the tape has already stopped */
            int dir = g->ts == TS_FF ? 1 : -1, w = (int)(g->wsp < 1.0 ? 1.0 : g->wsp);
            float tl[NTT], tr_[NTT];
            for (int i = 0; i < NTT; i++) {
                float vol = g->tp[TP_MUTE + i] > 0.5f ? 0.0f : g->tp[TP_VOL + i];
                float ang = (clampf(g->tp[TP_PAN + i], -1, 1) + 1) * 3.14159265f / 4;
                tl[i] = cosf(ang) * 1.41421356f * vol; tr_[i] = sinf(ang) * 1.41421356f * vol;
            }
            float a = 0.6f;
            for (int k = 0; k < n; k++) {
                long long p0 = g->P + (long long)(dir * sp * k);
                long long lo = dir > 0 ? p0 : p0 - w, hi = lo + w;
                if (lo < 0) lo = 0;
                if (hi > g->maxt) hi = g->maxt;
                float l = 0, r = 0;
                if (hi > lo) {
                    for (int i = 0; i < NTT; i++) {
                        TTrack *t = &g->tt[i];
                        if (tl[i] == 0.0f && tr_[i] == 0.0f) continue;
                        if (lo >= t->hi) continue;
                        int acc = 0;
                        for (long long q = lo; q < hi; q++) acc += t->d[q];
                        float x = (float)acc / (float)(hi - lo) / 32768.0f;
                        l += x * tl[i]; r += x * tr_[i];
                    }
                }
                float gk = (s0 + (s1 - s0) * (float)(k + 1) / (float)n) * 0.9f;
                g->sz[0] += a * (l - g->sz[0]); g->sz[1] += a * (r - g->sz[1]);
                g->L[k] += g->sz[0] * gk; g->R[k] += g->sz[1] * gk;
            }
            active = 1;
        } else if (s1 <= 0.0f) { g->sz[0] = g->sz[1] = 0; }
    }
    /* the tape winds only while nothing else is waiting: it stops at once when the button is let go */
    if (winding) {
        double top = g->tp[TP_WIND] < 2.0f ? 2.0 : g->tp[TP_WIND];
        if (g->wsp < 2.0) g->wsp = 2.0;
        g->wsp = g->wsp * 1.0045 > top ? top : g->wsp * 1.0045;   /* 2x to 30x in about 1.6 s */
        long long step = (long long)(g->wsp * n);
        if (g->ts == TS_FF) {
            g->P += step;
            if (g->P >= g->maxt) { g->P = g->maxt; g->wts = TS_STOP; }
        } else {
            g->P -= step;
            if (g->P <= 0) { g->P = 0; g->wts = TS_STOP; }
        }
        active = 1;
    } else if (g->ts != TS_FF && g->ts != TS_REW) {
        g->wsp = 0;                 /* next wind starts slow again */
    }
    /* input monitor */
    float pk2[2] = {0, 0}, mon = g->tp[TP_MON] > 0.5f ? gin : 0.0f;
    for (int k = 0; k < n; k++) {
        float a = g->inL[k] * gin, b = g->inR[k] * gin;
        if (fabsf(a) > pk2[0]) pk2[0] = fabsf(a);
        if (fabsf(b) > pk2[1]) pk2[1] = fabsf(b);
        if (mon != 0.0f) { g->L[k] += g->inL[k] * mon; g->R[k] += g->inR[k] * mon; }
    }
    for (int c = 0; c < 2; c++) g->in_pk[c] = g->in_pk[c] * 0.85f > pk2[c] ? g->in_pk[c] * 0.85f : pk2[c];
    if (g->cap_open || g->in_started) active = 1;
    if (g->mic_src) {               /* the mic is open only while REC is engaged on an armed track */
        int arm = 0;
        for (int i = 0; i < NTT; i++) if (g->tp[TP_ARM + i] > 0.5f) arm = 1;
        g->mic_want = (g->recm && arm) || any_window(g);
    }
    return active;
}

/* Click on every beat (higher on beat 1), from the start of the tape, only while it plays. Added after
 * the FX, never recorded. */
static void click(Eng *g, int n) {
    if (g->tp[TP_MET] < 0.5f || !g->click_on) return;
    int beats = (int)g->tp[TP_BEATS];
    if (beats < 1) beats = 4;
    double bpm = g->tp[TP_BPM] < 20.0f ? 120.0 : g->tp[TP_BPM];
    double beat = (double)g->sr * 60.0 / bpm;
    float vol = g->tp[TP_CLICK] * g->tg;
    double clen = 0.03 * g->sr;
    for (int k = 0; k < n; k++) {
        double x = (double)(g->click_p + k);
        double bi = floor(x / beat), ph = x - bi * beat;
        if (ph >= clen) continue;
        long long b = (long long)bi % beats;
        if (b < 0) b += beats;
        double f = b == 0 ? 1760.0 : 1100.0;
        float a = (float)(exp(-ph / (0.006 * g->sr)) * sin(6.283185307179586 * f * ph / g->sr)) * vol * (b == 0 ? 0.9f : 0.6f);
        g->L[k] += a; g->R[k] += a;
    }
}

/* Render one block into g->L / g->R: input -> tape -> FX -> volume. Returns 0 when idle. */
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
    int loop_on = tape_process(g, n);
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
    float ov = g->tp[TP_OUTVOL], pk[2] = {0, 0};
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


/* ---- tape calls from Python */
API void eng_tape(Eng *g, int cmd, int arg) { push(g, EV_LOOP, cmd, arg); }
API void eng_tape_params(Eng *g, const float *p) {
    LOCK(g->lock);
    memcpy(g->tp, p, sizeof(float) * TP_COUNT);
    UNLOCK(g->lock);
}
API int eng_tape_nparams(void) { return TP_COUNT; }
API long long eng_tape_max(Eng *g) { return g->maxt; }
API long long eng_tape_cells(Eng *g) { return g->ncell; }
API short *eng_tape_data(Eng *g, int i) { return (i >= 0 && i < NTT) ? g->tt[i].d : NULL; }
API unsigned char *eng_tape_env(Eng *g, int i) { return (i >= 0 && i < NTT) ? g->tt[i].env : NULL; }

/* Status for the screen, as doubles: [position, state, paused, rec engaged, wanted state, wanted pause,
 * tape frames, in peak L, in peak R, capture open, capture period, ring fill, input underruns, frames
 * skipped, ring target, mic, capture reads, capture peak, capture channels, drift ppm, undo (1 undo,
 * 2 redo), take running, tape gain, G], then 8 per track: [end, recording, peak, version, in the last
 * take, punch in, punch out (-1 open), take]. */
API int eng_tape_status(Eng *g, double *o, int max) {
    if (max < 24 + 8 * NTT) return 0;
    LOCK(g->lock);
    o[0] = (double)g->P; o[1] = g->ts; o[2] = g->tpaused; o[3] = g->recm; o[4] = g->wts; o[5] = g->wpaused;
    o[6] = (double)g->maxt; o[7] = g->in_pk[0]; o[8] = g->in_pk[1]; o[9] = g->cap_open; o[10] = g->cap_period;
    o[11] = (double)(unsigned)(g->rh - g->rt_); o[12] = g->in_under; o[13] = g->in_drop; o[14] = g->ring_target;
    o[15] = g->mic_on ? 1 : (g->mic_src ? 0.5 : 0);
#if !defined(_WIN32)
    o[16] = g->cap_reads; o[17] = g->cap_pk; o[18] = g->cap_ch;
#else
    o[16] = o[17] = o[18] = 0;
#endif
    o[19] = g->ratio > 0 ? (g->ratio - 1.0) * 1e6 : 0; o[20] = g->undo_st; o[21] = g->engaged; o[22] = g->tg; o[23] = (double)g->G;
    for (int i = 0; i < NTT; i++) {
        TTrack *t = &g->tt[i];
        double *q = o + 24 + 8 * i;
        q[0] = (double)t->hi; q[1] = t->rin >= 0; q[2] = t->pk; q[3] = t->ver;
        q[4] = t->take == g->undo_take && g->undo_st != 0; q[5] = (double)t->rin;
        q[6] = t->rout == TFOREVER ? -1.0 : (double)t->rout; q[7] = t->take;
    }
    UNLOCK(g->lock);
    return 24 + 8 * NTT;
}

/* Empty tape, at the start, stopped (loading another project). */
API void eng_tape_reset(Eng *g) {
    LOCK(g->lock);
    g->P = 0; g->ts = g->wts = TS_STOP; g->tpaused = g->wpaused = 0; g->recm = 0; g->engaged = 0; g->undo_st = 0;
    g->tg = 0;
    for (int i = 0; i < NTT; i++) {
        TTrack *t = &g->tt[i];
        memset(t->d, 0, sizeof(short) * (size_t)t->hi);
        memset(t->env, 0, (size_t)g->ncell);
        t->hi = 0; t->rin = -1; t->take = -1; t->pg = 1; t->ver++;
    }
    UNLOCK(g->lock);
}

/* A saved track back onto the tape, from the start. */
API void eng_tape_load(Eng *g, int i, const short *d, long long n) {
    if (i < 0 || i >= NTT || n < 1) return;
    LOCK(g->lock);
    TTrack *t = &g->tt[i];
    if (n > g->maxt) n = g->maxt;
    memcpy(t->d, d, sizeof(short) * (size_t)n);
    t->hi = n; t->take = -1;
    env_rebuild(g, t, 0, n);
    t->ver++;
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
static void *capture(void *arg) {
    Eng *g = (Eng *)arg;
    struct sched_param sp;
    sp.sched_priority = 52;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    void *pcm = NULL;
    int gen = -1, per = 128, chn = 2;
    char devs[512] = "", last_err[160] = "";
    static short buf[4096 * 2];
    static float st[4096 * 2], mono[4096];
    while (g->cap_run) {
        if (gen != g->cap_gen) {
            if (pcm) { A.drop(pcm); A.close(pcm); pcm = NULL; fprintf(stderr, "capture: closed\n"); }
            LOCK(g->qlock);
            strncpy(devs, g->cap_dev, sizeof(devs) - 1); devs[sizeof(devs) - 1] = 0;
            gen = g->cap_gen;
            UNLOCK(g->qlock);
            g->cap_open = 0; g->cap_err[0] = 0; last_err[0] = 0;
        }
        if (!devs[0]) { usleep(50000); continue; }
        if (!pcm) {
            /* each device name in turn, stereo then mono, 6 ms periods */
            char list[512], *save = NULL;
            strncpy(list, devs, sizeof(list) - 1); list[sizeof(list) - 1] = 0;
            for (char *dev = strtok_r(list, "\n", &save); dev && !pcm; dev = strtok_r(NULL, "\n", &save)) {
                for (int c = 2; c >= 1 && !pcm; c--) {
                    void *p = NULL;
                    int err = A.open(&p, dev, 1, 0);                      /* capture, blocking */
                    if (err < 0) { snprintf(g->cap_err, sizeof(g->cap_err), "%s: %s", dev, A.strerror(err)); break; }
                    err = A.setp(p, 2, 3, (unsigned)c, (unsigned)g->sr, 1, 6000);
                    if (err < 0) {
                        snprintf(g->cap_err, sizeof(g->cap_err), "%s %dch: %s", dev, c, A.strerror(err));
                        A.close(p);
                        continue;
                    }
                    unsigned long bsz = 0, p_ = 0;
                    A.getp(p, &bsz, &p_);
                    per = (p_ > 0 && p_ <= 4096) ? (int)p_ : 128;
                    chn = c; pcm = p;
                    strncpy(g->cap_used, dev, sizeof(g->cap_used) - 1);
                    A.start(pcm);
                    fprintf(stderr, "capture: opened %s, %d ch, period %d, buffer %lu\n", dev, c, per, bsz);
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
        }
        long r = A.readi(pcm, buf, (unsigned long)per);
        if (r < 0) {
            if (A.recover(pcm, (int)r, 1) < 0) {
                snprintf(g->cap_err, sizeof(g->cap_err), "read failed: %s", A.strerror((int)r));
                fprintf(stderr, "capture: %s (unplugged?)\n", g->cap_err);
                A.close(pcm); pcm = NULL; g->cap_open = 0; usleep(500000);
            } else A.start(pcm);
            continue;
        }
        if (r == 0) continue;
        g->cap_reads += 1;
        int mode = g->cap_mode;
        float pk = 0;
        for (long k = 0; k < r; k++) {
            float l, rr;
            if (chn == 2) { l = buf[2 * k] / 32768.0f; rr = buf[2 * k + 1] / 32768.0f; }
            else l = rr = buf[k] / 32768.0f;
            if (fabsf(l) > pk) pk = fabsf(l);
            if (fabsf(rr) > pk) pk = fabsf(rr);
            if (mode == 0) { st[2 * k] = st[2 * k + 1] = l; mono[k] = l; }
            else if (mode == 1) { st[2 * k] = st[2 * k + 1] = rr; mono[k] = rr; }
            else { st[2 * k] = l; st[2 * k + 1] = rr; mono[k] = 0.5f * (l + rr); }
        }
        g->cap_pk = fmaxf(g->cap_pk * 0.97f, pk);
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
