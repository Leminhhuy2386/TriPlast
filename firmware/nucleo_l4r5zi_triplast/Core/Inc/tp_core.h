/* tp_core.h - TriPlast on a microcontroller: value-domain features from raw sensor samples, the label-free store G,
 * the fast store F (normalized delta rule, or engram rehearsal), the consolidated store P with its spacing counter,
 * and pseudo-labelled engrams. The same file is compiled for the board (arm-none-eabi-gcc) and for the PC as the
 * fixed-point reference (src/device/reference.py builds it as a DLL), so both execute the same float32 operations
 * in the same order: the board's outputs must equal the reference's bit for bit.
 *
 * The math is src/models/triplast.py run_stream_reference with hard gates (deployment), applied to one person:
 *   per stream window: G's EMA update; F decays by (1 - delta); with pseudo-labels, an unlabelled window whose
 *   readout confidence >= tau joins the engram of its predicted class with weight omega; at a confirmation, F's write
 *   (one delta step, or the window joins its class's engram and R error-gated sweeps run over the engram class means),
 *   then the spacing counter and the F -> P transfer. After the stream: the final sweeps (pseudo-labels with
 *   rehearsal), then every scored window is classified with the frozen state.
 *
 * Stores W, F, P and the engram class means are rows of TP_BITS-bit integers with one float32 scale per row (block
 * floating point); TP_BITS = 32 keeps them float32 (scale 1). With integer stores the readout quantizes the
 * input u of each window to int16 with one scale and sums store x input products exactly in 64-bit integers
 * (SMLALD on the Cortex-M4, plain C on the PC: the same integers). Everything else is float32.
 *
 * Features (src/data/features.py, level3+stats, D = 88 x channels) are computed sample by sample, without a window
 * buffer: per channel, histograms of the sample's level bin (12 levels) for the whole window and its thirds, the
 * same for the first difference, crossing counts as difference arrays over bins, and the sums for mean and variance with
 * min and max (sums shifted by the window's first sample). The 12 levels are -2.5 + i * 5 / 11 in harness z units,
 * so a raw sample x lies above levels 0 .. ceil(t) - 1 with t = (z + 2.5) * 2.2 = a x + b; the host computes a, b
 * per channel from the harness z-score (and a for the difference, scaled by dx_std).
 *
 * Iteration 3, TriPlast-MR (src/models/multirate.py; header mr > 0, R = 0, no pseudo-labels, lam = 0): a
 * confirmation only stores its window's input u as an engram (TP_BITS-bit row with one scale, a ring of mr_per_class
 * engrams per class). tp_end_stream consolidates the bank: the engrams' Gram matrix and population logits (exact
 * integer dots), then for each store r with mr_rate[r] > 0, mr_epochs epochs of full-batch gradient descent in the
 * dual form (coefficients per engram and class, a bias change per class). A scored window's output is the mean of
 * the stores' probabilities (logits_out holds those probabilities); stream windows output the population logits.
 */
#ifndef TP_CORE_H
#define TP_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifndef TP_BITS
#define TP_BITS 16
#endif

#if TP_BITS == 8
typedef int8_t tp_q;
#define TP_QMAX 127.0f
#elif TP_BITS == 16
typedef int16_t tp_q;
#define TP_QMAX 32767.0f
#elif TP_BITS == 32
typedef float tp_q;
#else
#error "TP_BITS must be 8, 16 or 32"
#endif

#if defined(_WIN32) && defined(TP_HOST)
#define TP_API __declspec(dllexport)
#else
#define TP_API
#endif

#define TP_LEVELS 12
#define TP_BINS (TP_LEVELS + 1)
#define TP_VERSION 2u
#define TP_MAX_MR 8u

enum
{
  TP_POOL = 0u,       /* calibration-pool window, unlabelled                                     */
  TP_CONFIRM = 1u,    /* calibration-pool window confirmed with its label                        */
  TP_SCORE = 2u       /* scored window: classified with the frozen state, no update                */
};

/* Sent by the host (little-endian, packed as written here: 4 x u16, 4 x u8, 10 x f32, 2 x u16, 8 x f32 = 88 bytes). */
typedef struct
{
  uint16_t C, D, ch, L;            /* classes, features (88 ch), channels, samples per window               */
  uint8_t R, pseudo, hebbian, mr;  /* rehearsal sweeps (0: single delta step), pseudo-labels, Hebbian F;
                                      mr > 0: TriPlast-MR with mr stores (iteration 3)                       */
  float lam[3], alpha, eta, delta, kappa, Delta, tau, omega;
  uint16_t mr_epochs, mr_per_class; /* MR: replay epochs; engrams kept per class                            */
  float mr_rate[TP_MAX_MR];         /* MR: the stores' rates (0: the population readout itself)             */
} tp_header_t;

typedef struct
{
  float prev_x, prev_dx, k0, s1, s2, mn, mx;
  uint16_t hdx[TP_BINS], h3[3][TP_BINS];   /* the whole window's histogram is the sum of the thirds' */
  int16_t cx[TP_BINS], cdx[TP_BINS];
  uint8_t bx, bdx, pad[2];
} tp_acc_t;

typedef struct
{
  tp_header_t h;
  /* model constants (the blob, read only) */
  const float *Ws, *b, *aff, *nmean, *nstd, *sc_mean, *sc_inv;
  const tp_q *Wq;
  const uint8_t *groups;
  /* timing hook: DWT cycles of the last tp_window_end (features, readout, update); 0 on the host */
  uint32_t cyc[3];
  /* person state */
  float *m, *v, *Fs, *Ps, *Ms, *N, *cnt;
  tp_q *Fq, *Pq, *Mq;
  int32_t *last;
  float f_decay;
  int32_t t;
  /* window in progress */
  tp_acc_t *acc;
  uint16_t n_seen, third_end[3];
  uint8_t role, frozen;   /* frozen: after tp_end_stream, v holds the gain g and m holds lambda * m */
  uint16_t label;
  /* work buffers */
  float *z, *u, *row, *lg, *G, *A, *Ar, *L0, *wk;
  int16_t *uq;
  uint8_t *present;
  uint32_t bytes;
  float su;          /* scale of uq (the last readout's quantized input) */
  /* MR (iteration 3): engram rows Eq with scales Es, per-class write counts Ecnt, the engram list Ei; the bank:
   * Gram matrix Kg (n x n), population logits Le and errors Er (n x C), coefficients Mb (mr x nmax x C), bias
   * changes cb (mr x C), a scored window's engram products sv (nmax) and bank probabilities pb (C) */
  tp_q *Eq;
  float *Es, *Kg, *Le, *Er, *Mb, *cb, *sv, *pb;
  uint16_t *Ecnt, *Ei;
  uint16_t n_eng;
  uint8_t mr_done;
} tp_ctx_t;

/* Sizes for a header: returns the arena bytes (person state, work buffers, feature accumulators: the RAM the
 * method needs) and writes the model blob's bytes to *blob_bytes; 0 if the header is invalid. The blob holds the
 * constants: Ws[C], b[C], aff[ch*4] = (a, b, a_dx, 0), nmean[ch], nstd[ch], sc_mean[D], sc_inv[D] = 1 / scale (float32),
 * Wq[C*D] (tp_q), groups[D] (u8), each part 4-byte aligned. On the board the blob is in flash. */
TP_API uint32_t tp_layout(const tp_header_t *h, uint32_t *blob_bytes);
/* Points the context at the blob (read only) and lays the state out in `arena` (tp_layout bytes); returns 0 on
 * success. */
TP_API int32_t tp_init(tp_ctx_t *x, const tp_header_t *h, const uint8_t *blob, uint8_t *arena,
                       uint32_t arena_bytes);
/* A new person: m = 0, v = 1, F = P = engrams = 0, counters reset, stream position 0. */
TP_API void tp_reset(tp_ctx_t *x);
TP_API void tp_window_begin(tp_ctx_t *x, uint8_t role, uint16_t label);
/* n samples, time-major: samples[s * ch + c]. */
TP_API void tp_window_samples(tp_ctx_t *x, const float *samples, uint16_t n);
/* Finishes the window: features, then the stream step (TP_POOL / TP_CONFIRM) or the frozen-state logits
 * (TP_SCORE). logits_out (C floats): for TP_SCORE and TP_POOL the logits the readout gives the window (for a pool
 * window: before its update, with the decayed F, as the pseudo-label sees it); for TP_CONFIRM the logits before the
 * write. Returns the argmax (first maximum), or -1 if the window is incomplete. */
TP_API int32_t tp_window_end(tp_ctx_t *x, float *logits_out);
/* End of the calibration stream: the final rehearsal sweeps (pseudo-labels with R > 0), or the consolidation of
 * the MR bank; then G is frozen (the per-feature gain and offset are kept in place of v and m), so scored windows
 * skip the square roots. */
TP_API void tp_end_stream(tp_ctx_t *x);
/* CRC-32 of the person state (m, v, F, P, engrams, counters) and of the last features z, for equality checks. */
TP_API uint32_t tp_state_crc(const tp_ctx_t *x);
/* CRC-32 (IEEE 802.3, as zlib.crc32) of a byte range: the host's check of the blob written to flash. */
TP_API uint32_t tp_crc32(const void *data, uint32_t n);
TP_API const float *tp_features(const tp_ctx_t *x);
TP_API uint32_t tp_store_bits(void);

#endif /* TP_CORE_H */
