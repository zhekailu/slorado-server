/**
 * w8a8_fast.h
 *
 * Fused CUDA kernels for the slorado-w8a8-gpu-v1 model
 * dna_r10.4.1_e8.2_400bps_fast@v5.2.0-w8a8-perrowjoint-gpu-fp16scales-gxfp16-tanhsig
 * (W8A8 per-row-joint FAST v5.2, written by tools/FASTv5.2_quantForGPU/export_w8a8_gpu.py
 * --fp16-scales --gx-fp16 --tanh-sigmoid; any model with the same cell numerics works,
 * scales are data).  They compute exactly
 * what W8A8GpuCRFModel's LibTorch path computes -- same INT8 codes, same FP32 operations
 * in the same order -- but as three kernels instead of ~15 per LSTM timestep:
 *
 *   frontend  conv0 + conv1 + conv2 (+SiLU/clamp/quantize) for a tile of 64 output steps,
 *             all intermediates in shared memory; conv1/conv2 on INT8 tensor cores
 *   lstm      one LSTM layer, persistent: a group of 1, 2, 3 or 6 warps owns 16 chunks for the
 *             whole sequence (more warps per tile for small batches, chosen from the batch size
 *             and SM count), weights stay in shared memory, c stays in registers; the x-path
 *             GEMM is done inside the recurrence, so gx never goes to memory
 *   linear    96 -> 256 + clamp, FP16 scores written in TNC order for the decoder
 *
 * Activations between kernels are INT8 [T'][N][96] in a "fragment" channel order: within
 * each 96-byte row, byte 24c + 2j + b holds hidden unit 8j + 2c + b.  That is the order in
 * which the mma.m16n8k32 accumulator layout produces h, and it is also the operand layout
 * the next GEMM needs, so h never leaves registers between timesteps.  The pack functions
 * below permute the weight columns to match; nothing outside these kernels sees the order.
 *
 * Requires sm_80+ (mma.sync m16n8k32 s8).  Plain C API so it can be called from LibTorch
 * code (slorado) and from Python via ctypes (tests / benchmarks).
 */
#ifndef W8A8_FAST_H
#define W8A8_FAST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* packed sizes in bytes */
#define W8A8F_CONV1_FRAG_BYTES   1536    /* [2 ntile][3 kstep][32 lane] x 8 B  */
#define W8A8F_CONV2_FRAG_BYTES  30720    /* [12][10][32] x 8 B                 */
#define W8A8F_LSTM_FRAG_BYTES   73728    /* [12 hblock][4 gate][3 kstep][32 lane] x {x, h} 8 B each */
#define W8A8F_LSTM_EPI_FLOATS    1152    /* [12 hblock][4 quad] x 24: {dqx, bias} per gate, dqh */
#define W8A8F_LINEAR_FRAG_BYTES 24576    /* [32][3][32] x 8 B                  */

/* ---- host-side weight packing (plain host memory in and out) -------------------- */
/* conv weights in their natural [C_out][C_in][K] layout; LSTM/linear [rows][96] with
 * columns in natural unit order.  Outputs are the fragment-ordered device images. */
void w8a8f_pack_conv1(const int8_t *w /* [16][16][5] */, uint8_t *out);
void w8a8f_pack_conv2(const int8_t *w /* [96][16][19] */, uint8_t *out);
void w8a8f_pack_lstm(const int8_t *wih /* [384][96] */, const int8_t *whh /* [384][96] */,
                     const float *dqx, const float *dqh, const float *bias /* [384] each */,
                     uint8_t *frag_out, float *epi_out);
void w8a8f_pack_linear(const int8_t *w /* [256][96] */, uint8_t *out);

/* ---- device parameters (all pointers are device memory) ------------------------- */
typedef struct {
    const int8_t *w0;            /* conv0 weights [16][5]                         */
    const float  *dq0, *b0;      /* [16]                                          */
    const uint8_t *w1frag;       /* W8A8F_CONV1_FRAG_BYTES                        */
    const float  *dq1, *b1;      /* [16]                                          */
    const uint8_t *w2frag;       /* W8A8F_CONV2_FRAG_BYTES                        */
    const float  *dq2, *b2;      /* [96]                                          */
    float inv_in, inv_c1, inv_c2, inv_x0;  /* 1/s of conv0_in, conv1_in, conv2_in, lstm0_x */
    float conv_clamp;            /* 3.5                                           */
} w8a8f_frontend_params;

/* 1 if the current CUDA device can run these kernels (sm_80+: mma.sync m16n8k32 s8). */
int w8a8f_supported(void);

/* Each launcher returns 0 or a cudaError_t.  `stream` is a cudaStream_t (NULL = legacy). */

/* x: FP16 [N][T] normalised signal  ->  out: INT8 [Tp][N][96], Tp = (T - 1) / 6 + 1 */
int w8a8f_frontend(const void *x, int8_t *out, int N, int T, const w8a8f_frontend_params *p,
                   void *stream);

/* x, y: INT8 [Tp][N][96].  reverse = 1 runs t = Tp-1 .. 0.  inv_sh = 1 / s_h.
 * Cell numerics are fixed to the model these kernels are built for
 * (..-fp16scales-gxfp16-tanhsig): "elementwise": "fp32", "gx_dtype": "fp16",
 * "nonlinearity": "tanh_approx".  Callers must check the model declares exactly that. */
int w8a8f_lstm(const int8_t *x, int8_t *y, int N, int Tp, int reverse, const uint8_t *frag,
               const float *epi, float inv_sh, void *stream);

/* Warps per 16-chunk tile in w8a8f_lstm: 1, 2, 3 or 6 forces that split, 0 (default) picks
 * it from the batch size and SM count.  The output is identical for every split.  For tests
 * and benchmarks; the environment variable SLORADO_W8A8_LSTM_SPLIT does the same. */
void w8a8f_lstm_force_split(int split);

/* h: INT8 [rows][96]  ->  out: FP16 [rows][256] = clamp(acc * dq, lo, hi) */
int w8a8f_linear(const int8_t *h, void *out, int64_t rows, const uint8_t *frag, const float *dq,
                 float clamp_lo, float clamp_hi, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* W8A8_FAST_H */
