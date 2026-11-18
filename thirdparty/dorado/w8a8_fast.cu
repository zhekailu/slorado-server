/**
 * w8a8_fast.cu -- fused kernels for the slorado-w8a8-gpu-v1 FAST v5.2 model.  See w8a8_fast.h.
 *
 * Bit-exactness contract.  Every value these kernels produce equals what W8A8GpuCRFModel's
 * LibTorch path (and tools/FASTv5.2_quantForGPU/w8a8_gpu.forward) produces, because each
 * floating-point operation is the same IEEE operation in the same order:
 *   * INT8 x INT8 -> INT32 sums are exact in any order (|acc| < 2^22)
 *   * every mul / add is an explicit __fmul_rn / __fadd_rn: the LibTorch path runs them as
 *     separate kernels, so they must never be contracted into an FMA here either
 *   * activation quantization is x * (1/s) with 1/s rounded to FP32 -- what ATen does for
 *     "tensor / cpu-scalar" on CUDA -- then round-half-even, clamp to [-128, 127]
 *   * SiLU is x / (1 + expf(-x)), ATen's formula; tanh is tanh.approx.f32 and
 *     sigmoid(x) = 0.5 + 0.5 tanh.approx(0.5x), as the model's "nonlinearity" asks
 *   * gx is rounded to FP16 and back (the model's "gx_dtype": "fp16")
 */
#include "w8a8_fast.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace {

// ── primitives ──────────────────────────────────────────────────────────────

// D += A(16x32 s8, row) * B(32x8 s8, col), INT32 accumulate.
//   A regs: a0 = (row g, k 4c..4c+3), a1 = (row g+8, same k), a2 = (row g, k 16+4c..), a3 = (row g+8, ..)
//   B regs: b0 = (k 4c..4c+3, col g), b1 = (k 16+4c.., col g)
//   D regs: d0 = (row g, col 2c), d1 = (row g, col 2c+1), d2 = (row g+8, 2c), d3 = (row g+8, 2c+1)
// with g = lane / 4, c = lane % 4.
__device__ __forceinline__ void mma_s8(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint2 b) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
}

// Exact int -> float for |a| < 2^22: two full-rate ops instead of a quarter-rate I2F.
// Every K <= 96 GEMM here is within it (127 * 128 * 96 < 2^21).
__device__ __forceinline__ float i2f(int a) {
    return __fsub_rn(__int_as_float(a + 0x4B400000), 12582912.0f);
}

// round-half-even(v), clamped to [-128, 127]; v is clamped first so the magic-number
// rounding (exact for |v| < 2^22) is always valid.
__device__ __forceinline__ int q8(float v) {
    v = fminf(fmaxf(v, -4096.0f), 4096.0f);
    const int r = __float_as_int(__fadd_rn(v, 12582912.0f)) - 0x4B400000;
    return min(max(r, -128), 127);
}

__device__ __forceinline__ float tanh_approx(float x) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// 0.5 + 0.5 tanh(0.5x).  0.5 * t is exact (it can only lose bits when |t| < 2^-125, where
// adding 0.5 rounds to 0.5 either way), so one FMA equals the reference's mul-then-add.
__device__ __forceinline__ float sigmoid_tanh(float x) {
    return __fmaf_rn(0.5f, tanh_approx(__fmul_rn(x, 0.5f)), 0.5f);
}

// INT8 code of h = round_half_even(h / s_h) as the low byte of the returned word.  |h| <= 1
// so |h * inv_sh| < 127.5: no clamp is needed, and the magic-number rounding leaves the
// two's-complement code in the low byte of the float's bit pattern.
__device__ __forceinline__ uint32_t h_code_bits(float h, float inv_sh) {
    return (uint32_t)__float_as_int(__fadd_rn(__fmul_rn(h, inv_sh), 12582912.0f));
}

// ATen's silu: x / (1 + exp(-x)), then the model's clamp_max.
__device__ __forceinline__ float silu_clamp(float x, float hi) {
    return fminf(__fdiv_rn(x, __fadd_rn(1.0f, expf(-x))), hi);
}

__device__ __forceinline__ uint32_t lds32(const uint8_t *p) {
    return *reinterpret_cast<const uint32_t *>(p);
}

// A fragment of one 96-byte fragment-ordered activation row: the 24 bytes this lane's
// quad position owns, as six words (word 2s -> a0/a1 of k-step s, word 2s+1 -> a2/a3).
__device__ __forceinline__ void load_row(const int8_t *row, bool valid, int c, uint32_t (&w)[6]) {
    if (valid) {
        const uint2 *p = reinterpret_cast<const uint2 *>(row + 24 * c);
        const uint2 v0 = __ldg(p), v1 = __ldg(p + 1), v2 = __ldg(p + 2);
        w[0] = v0.x; w[1] = v0.y; w[2] = v1.x; w[3] = v1.y; w[4] = v2.x; w[5] = v2.y;
    } else {
#pragma unroll
        for (int i = 0; i < 6; ++i) w[i] = 0;
    }
}

__device__ __forceinline__ void store_row(int8_t *row, int c, const uint32_t (&w)[6]) {
    uint2 *p = reinterpret_cast<uint2 *>(row + 24 * c);
    p[0] = make_uint2(w[0], w[1]);
    p[1] = make_uint2(w[2], w[3]);
    p[2] = make_uint2(w[4], w[5]);
}

// Bytes 12*half .. 12*half+11 of a lane's 24-byte segment (words 3*half .. 3*half+2).
__device__ __forceinline__ void store_half_row(int8_t *row, int c, int half, const uint32_t (&w)[6]) {
    uint32_t *p = reinterpret_cast<uint32_t *>(row + 24 * c + 12 * half);
#pragma unroll
    for (int i = 0; i < 3; ++i) p[i] = w[3 * half + i];
}

// Place an INT8 code for (hidden block j, unit-in-pair b) into a lane's six-word segment.
__device__ __forceinline__ void put_code(uint32_t (&w)[6], int j, int b, int code) {
    w[j >> 1] |= (uint32_t)(code & 0xFF) << (16 * (j & 1) + 8 * b);
}

// ── frontend: conv0 -> conv1 -> conv2, 64 conv2 outputs per work item ──────────
constexpr int FT = 64;                // conv2 outputs per item
constexpr int FL0 = 6 * FT + 21;      // quantized input samples    [6t0-13, 6t0+6FT+8)
constexpr int FL1 = 6 * FT + 17;      // conv0 outputs               [6t0-11, ...)
constexpr int FL2 = 6 * FT + 13;      // conv1 outputs               [6t0-9, ...)
constexpr int F_XQ = 416;             // >= FL0
constexpr int F_C0_ROWS = 416;        // >= 25 m-tiles * 16 + 5 (conv1 reads 6 taps, the 6th is padding)
constexpr int F_C1_ROWS = 400;        // >= 6 * 63 + 20 (conv2 reads 20 taps, the 20th is padding)
constexpr int F_THREADS = 256;      // 8 warps, 2 CTAs per SM

struct FrontSmem {
    uint2 w2[W8A8F_CONV2_FRAG_BYTES / 8];
    uint2 w1[W8A8F_CONV1_FRAG_BYTES / 8];
    float dq0[16], b0[16], dq1[16], b1[16], dq2[96], b2[96];
    int w0[16][5];
    int8_t xq[F_XQ];
    int8_t c0[F_C0_ROWS * 16];
    int8_t c1[F_C1_ROWS * 16];
};

__global__ void __launch_bounds__(F_THREADS, 2)
frontend_kernel(const __half *__restrict__ x, int8_t *__restrict__ out, int N, int T, int Tp,
                w8a8f_frontend_params p) {
    extern __shared__ __align__(16) uint8_t smem_raw[];
    FrontSmem &s = *reinterpret_cast<FrontSmem *>(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;

    {
        const uint4 *src2 = reinterpret_cast<const uint4 *>(p.w2frag);
        uint4 *dst2 = reinterpret_cast<uint4 *>(s.w2);
        for (int i = tid; i < W8A8F_CONV2_FRAG_BYTES / 16; i += F_THREADS) dst2[i] = src2[i];
        const uint4 *src1 = reinterpret_cast<const uint4 *>(p.w1frag);
        uint4 *dst1 = reinterpret_cast<uint4 *>(s.w1);
        for (int i = tid; i < W8A8F_CONV1_FRAG_BYTES / 16; i += F_THREADS) dst1[i] = src1[i];
        if (tid < 16) {
            s.dq0[tid] = p.dq0[tid]; s.b0[tid] = p.b0[tid];
            s.dq1[tid] = p.dq1[tid]; s.b1[tid] = p.b1[tid];
            for (int k = 0; k < 5; ++k) s.w0[tid][k] = p.w0[tid * 5 + k];
        }
        if (tid < 96) { s.dq2[tid] = p.dq2[tid]; s.b2[tid] = p.b2[tid]; }
    }

    // conv0 work split: thread -> (output channel co0, block of 26 consecutive positions)
    const int co0 = tid & 15, blk0 = tid >> 4;
    int w0r[5];
    float dq0r, b0r;

    const int ntiles = (Tp + FT - 1) / FT;
    const int64_t items = (int64_t)N * ntiles;
    for (int64_t item = blockIdx.x; item < items; item += gridDim.x) {
        const int n = (int)(item / ntiles);
        const int t0 = (int)(item % ntiles) * FT;
        const int base0 = 6 * t0 - 13, base1 = 6 * t0 - 11, base2 = 6 * t0 - 9;
        __syncthreads();   // previous item finished with the buffers (and parameters loaded)
        if (item == blockIdx.x) {
#pragma unroll
            for (int k = 0; k < 5; ++k) w0r[k] = s.w0[co0][k];
            dq0r = s.dq0[co0];
            b0r = s.b0[co0];
        }

        // input -> INT8 codes; zero codes outside [0, T) are conv0's padding
        const __half *xn = x + (int64_t)n * T;
        for (int i = tid; i < FL0; i += F_THREADS) {
            const int gp = base0 + i;
            s.xq[i] = (gp >= 0 && gp < T) ? (int8_t)q8(__fmul_rn(__half2float(xn[gp]), p.inv_in)) : 0;
        }
        __syncthreads();

        // conv0 (1 -> 16, k5): scalar integer dot products, a sliding window of 5 inputs
        {
            constexpr int RUN = (FL1 + 15) / 16;                  // 26 positions per thread
            const int pp0 = blk0 * RUN;
            int win[5];
#pragma unroll
            for (int k = 0; k < 4; ++k) win[k] = (pp0 + k < F_XQ) ? s.xq[pp0 + k] : 0;
#pragma unroll 2
            for (int i = 0; i < RUN; ++i) {
                const int pp = pp0 + i;
                if (pp >= FL1) break;
                win[4] = s.xq[pp + 4];
                const int gp = base1 + pp;
                int code = 0;
                if (gp >= 0 && gp < T) {
                    const int acc = win[0] * w0r[0] + win[1] * w0r[1] + win[2] * w0r[2] + win[3] * w0r[3] +
                                    win[4] * w0r[4];
                    const float y = __fadd_rn(__fmul_rn(i2f(acc), dq0r), b0r);
                    code = q8(__fmul_rn(silu_clamp(y, p.conv_clamp), p.inv_c1));
                }
                s.c0[pp * 16 + co0] = (int8_t)code;
#pragma unroll
                for (int k = 0; k < 4; ++k) win[k] = win[k + 1];
            }
        }
        __syncthreads();

        // conv1 (16 -> 16, k5): rows = conv1 positions, K = tap * 16 + ci (80, padded to 96)
        for (int mt = warp; mt < (FL2 + 15) / 16; mt += F_THREADS / 32) {
            const uint8_t *ra = reinterpret_cast<const uint8_t *>(s.c0) + (mt * 16 + g) * 16 + 4 * c;
            const uint8_t *rb = ra + 8 * 16;
            int acc[2][4] = {};
#pragma unroll
            for (int ks = 0; ks < 3; ++ks) {
                const uint32_t a0 = lds32(ra + 32 * ks), a1 = lds32(rb + 32 * ks);
                const uint32_t a2 = lds32(ra + 32 * ks + 16), a3 = lds32(rb + 32 * ks + 16);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) mma_s8(acc[nt], a0, a1, a2, a3, s.w1[(nt * 3 + ks) * 32 + lane]);
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int pp = mt * 16 + g + (e >> 1) * 8, ch = nt * 8 + 2 * c + (e & 1);
                    if (pp < FL2) {
                        const int gp = base2 + pp;
                        int code = 0;
                        if (gp >= 0 && gp < T) {
                            const float y = __fadd_rn(__fmul_rn(i2f(acc[nt][e]), s.dq1[ch]), s.b1[ch]);
                            code = q8(__fmul_rn(silu_clamp(y, p.conv_clamp), p.inv_c2));
                        }
                        s.c1[pp * 16 + ch] = (int8_t)code;
                    }
                }
            }
        }
        __syncthreads();

        // conv2 (16 -> 96, k19, stride 6): warp -> m-tile (warp & 3), n-tiles 6*(warp >> 2) .. +5;
        // K = 304 -> 320.  A lane's 6 n-tiles are bytes 12*half .. +11 of its 24-byte segment.
        {
            const int tl = (warp & 3) * 16 + g;                  // local output index, rows g / g+8
            const int nh = warp >> 2;
            const uint8_t *ra = reinterpret_cast<const uint8_t *>(s.c1) + tl * 96 + 4 * c;
            const uint8_t *rb = ra + 8 * 96;
            int acc[6][4] = {};
#pragma unroll
            for (int ks = 0; ks < 10; ++ks) {
                const uint32_t a0 = lds32(ra + 32 * ks), a1 = lds32(rb + 32 * ks);
                const uint32_t a2 = lds32(ra + 32 * ks + 16), a3 = lds32(rb + 32 * ks + 16);
#pragma unroll
                for (int i = 0; i < 6; ++i)
                    mma_s8(acc[i], a0, a1, a2, a3, s.w2[((nh * 6 + i) * 10 + ks) * 32 + lane]);
            }
            uint32_t wa[6] = {}, wb[6] = {};
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                const int nt = nh * 6 + i;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int co = nt * 8 + 2 * c + (e & 1);
                    // K = 304: |acc| can exceed 2^22 in the worst case, so the exact cvt here
                    const float y = __fadd_rn(__fmul_rn(__int2float_rn(acc[i][e]), s.dq2[co]), s.b2[co]);
                    const int code = q8(__fmul_rn(silu_clamp(y, p.conv_clamp), p.inv_x0));
                    if (e < 2) put_code(wa, nt, e & 1, code); else put_code(wb, nt, e & 1, code);
                }
            }
            const int ta = t0 + tl, tb = ta + 8;
            if (ta < Tp) store_half_row(out + ((int64_t)ta * N + n) * 96, c, nh, wa);
            if (tb < Tp) store_half_row(out + ((int64_t)tb * N + n) * 96, c, nh, wb);
        }
    }
}

// ── LSTM layer: persistent, weights in shared memory ─────────────────────────────
// A 16-chunk row tile is owned for the whole sequence by a group of S warps (S = 1, 2, 3, 6).
// Warp s of the group computes hidden blocks [s * 12/S, (s+1) * 12/S): their x- and h-path
// GEMMs, cell epilogue and c state.  S = 1 is the throughput layout (h stays in registers);
// S > 1 cuts the serial work per timestep by S for small batches, where there are too few tiles
// to fill the GPU and every step is latency-bound.  The S warps exchange h codes through a
// double-buffered shared tile, so one named barrier per step suffices.  Every cell value is
// computed by exactly the same operations whatever S is, so the output does not depend on S.
#ifndef W8A8F_LSTM_WARPS
#define W8A8F_LSTM_WARPS 8           // warps per CTA (one CTA per SM: the weights fill shared memory)
#endif
#ifndef W8A8F_LSTM_PREFETCH
#define W8A8F_LSTM_PREFETCH 1        // load x_{t+1} during step t (12 more live registers)
#endif
constexpr int LSTM_MAX_WARPS = W8A8F_LSTM_WARPS;
constexpr int LSTM_SMEM = W8A8F_LSTM_FRAG_BYTES + W8A8F_LSTM_EPI_FLOATS * 4;
constexpr int LSTM_HTILE = 16 * 96;                       // one h tile, fragment-ordered rows
constexpr int LSTM_SMEM_MAX = LSTM_SMEM + (LSTM_MAX_WARPS / 2) * 2 * LSTM_HTILE;

// Epilogue constants of one hidden block j for quad position c (columns 8j + 2c, +1 of
// every gate): gate[q] = {dqx0, dqx1, b0, b1}, dqh[0] = {q0: dqh0, dqh1, q1: dqh0, dqh1},
// dqh[1] = the same for q2, q3.  Six LDS.128 per hidden block.
struct __align__(16) LstmEpi { float4 gate[4]; float4 dqh[2]; };

__device__ __forceinline__ void named_barrier(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

template <int S>
__global__ void __launch_bounds__(LSTM_MAX_WARPS * 32, 1)
lstm_kernel(const int8_t *__restrict__ x, int8_t *__restrict__ y, int N, int Tp, int reverse,
            const uint8_t *__restrict__ frag, const float *__restrict__ epi, float inv_sh) {
    constexpr int JB = 12 / S;               // hidden blocks per warp (even: whole h words)
    constexpr int NW = JB / 2;               // h words per row this warp produces
    static_assert(12 % S == 0 && JB % 2 == 0, "S must be 1, 2, 3 or 6");

    extern __shared__ __align__(16) uint8_t smem_raw[];
    const uint4 *s_w = reinterpret_cast<const uint4 *>(smem_raw);       // [12][4][3][32] {x, h}
    const LstmEpi *s_e = reinterpret_cast<const LstmEpi *>(smem_raw + W8A8F_LSTM_FRAG_BYTES);
    {
        const uint4 *src = reinterpret_cast<const uint4 *>(frag);
        uint4 *dst = reinterpret_cast<uint4 *>(smem_raw);
        for (int i = threadIdx.x; i < W8A8F_LSTM_FRAG_BYTES / 16; i += blockDim.x) dst[i] = src[i];
        const float4 *se = reinterpret_cast<const float4 *>(epi);
        float4 *de = reinterpret_cast<float4 *>(smem_raw + W8A8F_LSTM_FRAG_BYTES);
        for (int i = threadIdx.x; i < W8A8F_LSTM_EPI_FLOATS / 4; i += blockDim.x) de[i] = se[i];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int grp = warp / S, ws = warp % S;                 // group = tile; warp within it
    const int row0 = (blockIdx.x * ((blockDim.x >> 5) / S) + grp) * 16;
    if (row0 >= N) return;                                   // whole groups leave together
    const int ra = row0 + g, rb = ra + 8;
    const bool va = ra < N, vb = rb < N;
    const int j0 = ws * JB;                                  // first hidden block of this warp
    uint8_t *s_h = smem_raw + LSTM_SMEM + grp * 2 * LSTM_HTILE;   // [2][16 rows][96 B] (S > 1)
    (void)s_h;

    float cs[JB][4];
#pragma unroll
    for (int j = 0; j < JB; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) cs[j][e] = 0.0f;
    uint32_t ha[6] = {}, hb[6] = {};             // h_{t-1} codes, rows g and g+8 (A operand)

    const int dt = reverse ? -1 : 1;
    int t = reverse ? Tp - 1 : 0;
    uint32_t xa[6], xb[6];
#if W8A8F_LSTM_PREFETCH
    load_row(x + ((int64_t)t * N + ra) * 96, va, c, xa);
    load_row(x + ((int64_t)t * N + rb) * 96, vb, c, xb);
#endif

    for (int step = 0; step < Tp; ++step, t += dt) {
#if W8A8F_LSTM_PREFETCH
        uint32_t nxa[6], nxb[6];                 // prefetch x_{t+1}
        const bool more = step + 1 < Tp;
        load_row(x + ((int64_t)(t + dt) * N + ra) * 96, va && more, c, nxa);
        load_row(x + ((int64_t)(t + dt) * N + rb) * 96, vb && more, c, nxb);
#else
        load_row(x + ((int64_t)t * N + ra) * 96, va, c, xa);
        load_row(x + ((int64_t)t * N + rb) * 96, vb, c, xb);
#endif

        uint32_t na[NW] = {}, nb[NW] = {};       // h_t codes of this warp's hidden blocks
#pragma unroll
        for (int jj = 0; jj < JB; ++jj) {        // hidden block j: units 8j .. 8j+7, all 4 gates
            const int j = j0 + jj;
            int ax[4][4] = {}, ah[4][4] = {};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
                for (int ks = 0; ks < 3; ++ks) {
                    const uint4 w = s_w[((j * 4 + q) * 3 + ks) * 32 + lane];
                    mma_s8(ax[q], xa[2 * ks], xb[2 * ks], xa[2 * ks + 1], xb[2 * ks + 1], make_uint2(w.x, w.y));
                    mma_s8(ah[q], ha[2 * ks], hb[2 * ks], ha[2 * ks + 1], hb[2 * ks + 1], make_uint2(w.z, w.w));
                }
            }
            const LstmEpi &pe = s_e[j * 4 + c];
            float gate[4][4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 gq = pe.gate[q];
                const float4 dh = pe.dqh[q >> 1];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float dqx = (e & 1) ? gq.y : gq.x;
                    const float bb = (e & 1) ? gq.w : gq.z;
                    const float dqh = (q & 1) ? ((e & 1) ? dh.w : dh.z) : ((e & 1) ? dh.y : dh.x);
                    float gx = __fadd_rn(__fmul_rn(__int2float_rn(ax[q][e]), dqx), bb);   // I2F: one instruction, measured faster here
                    gx = __half2float(__float2half_rn(gx));             // gx stored as FP16
                    gate[q][e] = __fadd_rn(gx, __fmul_rn(__int2float_rn(ah[q][e]), dqh));
                }
            }
            uint32_t hc[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float ig = sigmoid_tanh(gate[0][e]);
                const float fg = sigmoid_tanh(gate[1][e]);
                const float gg = tanh_approx(gate[2][e]);
                const float og = sigmoid_tanh(gate[3][e]);
                cs[jj][e] = __fadd_rn(__fmul_rn(fg, cs[jj][e]), __fmul_rn(ig, gg));
                hc[e] = h_code_bits(__fmul_rn(og, tanh_approx(cs[jj][e])), inv_sh);
            }
            // codes of units (2c, 2c+1) for rows g and g+8 -> 16-bit halves of the h words
            // (j0 is even, so j & 1 == jj & 1)
            const uint32_t pa = __byte_perm(hc[0], hc[1], 0x0040), pb = __byte_perm(hc[2], hc[3], 0x0040);
            if (jj & 1) {
                na[jj >> 1] = __byte_perm(na[jj >> 1], pa, 0x5410);
                nb[jj >> 1] = __byte_perm(nb[jj >> 1], pb, 0x5410);
            } else {
                na[jj >> 1] = pa & 0xFFFFu;
                nb[jj >> 1] = pb & 0xFFFFu;
            }
        }
        if constexpr (S == 1) {
            if (va) store_row(y + ((int64_t)t * N + ra) * 96, c, na);
            if (vb) store_row(y + ((int64_t)t * N + rb) * 96, c, nb);
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                ha[i] = na[i];
                hb[i] = nb[i];
            }
        } else {
            // Publish this warp's words of h_t, wait for the group, read back all of h_t.
            // Buffer step & 1: a warp only rewrites it two steps later, after the next barrier,
            // by which time every warp of the group has finished reading it.
            uint8_t *hbuf = s_h + (step & 1) * LSTM_HTILE;
#pragma unroll
            for (int i = 0; i < NW; ++i) {
                *reinterpret_cast<uint32_t *>(hbuf + g * 96 + 24 * c + 4 * (ws * NW + i)) = na[i];
                *reinterpret_cast<uint32_t *>(hbuf + (g + 8) * 96 + 24 * c + 4 * (ws * NW + i)) = nb[i];
            }
            named_barrier(1 + grp, S * 32);
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const uint2 va2 = *reinterpret_cast<const uint2 *>(hbuf + g * 96 + 24 * c + 8 * i);
                const uint2 vb2 = *reinterpret_cast<const uint2 *>(hbuf + (g + 8) * 96 + 24 * c + 8 * i);
                ha[2 * i] = va2.x; ha[2 * i + 1] = va2.y;
                hb[2 * i] = vb2.x; hb[2 * i + 1] = vb2.y;
            }
            // the 16 rows of y_t are contiguous: 96 16-byte pieces, split over the group
            uint4 *yt = reinterpret_cast<uint4 *>(y + ((int64_t)t * N + row0) * 96);
            const uint4 *ht = reinterpret_cast<const uint4 *>(hbuf);
            for (int k = ws * 32 + lane; k < LSTM_HTILE / 16; k += S * 32)
                if (row0 + k / 6 < N) yt[k] = ht[k];
        }
#if W8A8F_LSTM_PREFETCH
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            xa[i] = nxa[i];
            xb[i] = nxb[i];
        }
#endif
    }
}

// ── linear 96 -> 256, clamp, FP16 out ─────────────────────────────────────────────
// Each warp computes a 16-row tile in two halves of 128 columns, stages a half in shared
// memory and writes it back with 16-byte stores (full 32-byte sectors, rows contiguous).
constexpr int LIN_WARPS = 4;
constexpr int LIN_STAGE = 64 + 4;     // half2 per staged row (+4 to spread banks)

__global__ void __launch_bounds__(LIN_WARPS * 32)
linear_kernel(const int8_t *__restrict__ h, __half *__restrict__ out, int64_t rows,
              const uint8_t *__restrict__ frag, const float *__restrict__ dq, float lo, float hi) {
    __shared__ __align__(16) uint2 s_w[W8A8F_LINEAR_FRAG_BYTES / 8];
    __shared__ __align__(16) float2 s_dq[128];
    __shared__ __align__(16) __half2 s_o[LIN_WARPS][16][LIN_STAGE];
    {
        const uint4 *src = reinterpret_cast<const uint4 *>(frag);
        uint4 *dst = reinterpret_cast<uint4 *>(s_w);
        for (int i = threadIdx.x; i < W8A8F_LINEAR_FRAG_BYTES / 16; i += blockDim.x) dst[i] = src[i];
        const float2 *d2 = reinterpret_cast<const float2 *>(dq);
        for (int i = threadIdx.x; i < 128; i += blockDim.x) s_dq[i] = d2[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    __half2 (*so)[LIN_STAGE] = s_o[warp];
    const int64_t tiles = (rows + 15) / 16;
    for (int64_t tile = (int64_t)blockIdx.x * LIN_WARPS + warp; tile < tiles;
         tile += (int64_t)gridDim.x * LIN_WARPS) {
        const int64_t ra = tile * 16 + g, rb = ra + 8;
        uint32_t xa[6], xb[6];
        load_row(h + ra * 96, ra < rows, c, xa);
        load_row(h + rb * 96, rb < rows, c, xb);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll 4
            for (int ntl = 0; ntl < 16; ++ntl) {
                const int nt = half * 16 + ntl;
                int acc[4] = {};
#pragma unroll
                for (int ks = 0; ks < 3; ++ks)
                    mma_s8(acc, xa[2 * ks], xb[2 * ks], xa[2 * ks + 1], xb[2 * ks + 1], s_w[(nt * 3 + ks) * 32 + lane]);
                const float2 d = s_dq[nt * 4 + c];
                const float v0 = fminf(fmaxf(__fmul_rn(i2f(acc[0]), d.x), lo), hi);
                const float v1 = fminf(fmaxf(__fmul_rn(i2f(acc[1]), d.y), lo), hi);
                const float v2 = fminf(fmaxf(__fmul_rn(i2f(acc[2]), d.x), lo), hi);
                const float v3 = fminf(fmaxf(__fmul_rn(i2f(acc[3]), d.y), lo), hi);
                so[g][ntl * 4 + c] = __floats2half2_rn(v0, v1);
                so[g + 8][ntl * 4 + c] = __floats2half2_rn(v2, v3);
            }
            __syncwarp();
#pragma unroll
            for (int it = 0; it < 8; ++it) {
                const int r = 2 * it + (lane >> 4);
                const int64_t row = tile * 16 + r;
                if (row < rows) {
                    const uint4 v = *reinterpret_cast<const uint4 *>(&so[r][(lane & 15) * 4]);
                    *reinterpret_cast<uint4 *>(out + row * 256 + half * 128 + (lane & 15) * 8) = v;
                }
            }
            __syncwarp();
        }
    }
}

// ── host helpers ──────────────────────────────────────────────────────────────

// Position k of a K=96 A operand -> byte in a fragment-ordered row -> natural unit index.
int k_to_pos(int k) {
    const int s = k / 32, r = k % 32, h = r / 16, c = (r % 16) / 4, e = r % 4;
    return 24 * c + 8 * s + 4 * h + e;
}
int pos_to_unit(int pos) {
    const int c = pos / 24, j = (pos % 24) / 2, b = pos % 2;
    return 8 * j + 2 * c + b;
}
int k_to_unit(int k) { return pos_to_unit(k_to_pos(k)); }

// One B fragment (8 bytes) for lane (g, c), k-step ks: column n's bytes at k = 32ks + 4c + e
// and 32ks + 16 + 4c + e.  `at(n, k)` returns the weight.
template <typename F>
void put_bfrag(uint8_t *dst, int n_base, int ks, F at) {
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, c = lane % 4;
        for (int e = 0; e < 4; ++e) {
            dst[lane * 8 + e] = (uint8_t)at(n_base + g, 32 * ks + 4 * c + e);
            dst[lane * 8 + 4 + e] = (uint8_t)at(n_base + g, 32 * ks + 16 + 4 * c + e);
        }
    }
}

struct DeviceInfo { int sms = 0; bool attrs = false; };
int g_lstm_force_split = -1;      // -1: not read yet; 0: automatic; 1, 2, 3, 6: forced
DeviceInfo g_dev[64];

DeviceInfo &device_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    DeviceInfo &d = g_dev[dev & 63];
    if (!d.attrs) {
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaFuncSetAttribute(lstm_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, LSTM_SMEM);
        cudaFuncSetAttribute(lstm_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, LSTM_SMEM_MAX);
        cudaFuncSetAttribute(lstm_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, LSTM_SMEM_MAX);
        cudaFuncSetAttribute(lstm_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, LSTM_SMEM_MAX);
        cudaFuncSetAttribute(frontend_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)sizeof(FrontSmem));
        d.attrs = true;
    }
    return d;
}

}  // namespace

// ── C API ─────────────────────────────────────────────────────────────────────

extern "C" void w8a8f_pack_conv1(const int8_t *w, uint8_t *out) {
    for (int nt = 0; nt < 2; ++nt)
        for (int ks = 0; ks < 3; ++ks)
            put_bfrag(out + ((nt * 3 + ks) * 32) * 8, nt * 8, ks, [&](int n, int k) -> int8_t {
                const int tap = k / 16, ci = k % 16;
                return tap < 5 ? w[(n * 16 + ci) * 5 + tap] : 0;
            });
}

extern "C" void w8a8f_pack_conv2(const int8_t *w, uint8_t *out) {
    for (int nt = 0; nt < 12; ++nt)
        for (int ks = 0; ks < 10; ++ks)
            put_bfrag(out + ((nt * 10 + ks) * 32) * 8, nt * 8, ks, [&](int n, int k) -> int8_t {
                const int tap = k / 16, ci = k % 16;
                return tap < 19 ? w[(n * 16 + ci) * 19 + tap] : 0;
            });
}

extern "C" void w8a8f_pack_lstm(const int8_t *wih, const int8_t *whh, const float *dqx,
                                const float *dqh, const float *bias, uint8_t *frag_out,
                                float *epi_out) {
    // [j][q][kstep][lane] x 16 B: bytes 0-7 the x-path fragment, 8-15 the h-path fragment
    uint8_t tmp[32 * 8];
    for (int j = 0; j < 12; ++j)
        for (int q = 0; q < 4; ++q)
            for (int ks = 0; ks < 3; ++ks)
                for (int path = 0; path < 2; ++path) {
                    const int8_t *w = path == 0 ? wih : whh;
                    put_bfrag(tmp, q * 96 + 8 * j, ks,
                              [&](int n, int k) -> int8_t { return w[n * 96 + k_to_unit(k)]; });
                    uint8_t *dst = frag_out + (((j * 4 + q) * 3 + ks) * 32) * 16 + path * 8;
                    for (int lane = 0; lane < 32; ++lane)
                        for (int b = 0; b < 8; ++b) dst[lane * 16 + b] = tmp[lane * 8 + b];
                }
    for (int j = 0; j < 12; ++j)
        for (int c = 0; c < 4; ++c) {
            float *e = epi_out + (j * 4 + c) * 24;
            for (int q = 0; q < 4; ++q) {
                const int n0 = q * 96 + 8 * j + 2 * c;
                e[q * 4 + 0] = dqx[n0]; e[q * 4 + 1] = dqx[n0 + 1];
                e[q * 4 + 2] = bias[n0]; e[q * 4 + 3] = bias[n0 + 1];
                e[16 + q * 2 + 0] = dqh[n0]; e[16 + q * 2 + 1] = dqh[n0 + 1];
            }
        }
}

extern "C" void w8a8f_pack_linear(const int8_t *w, uint8_t *out) {
    for (int nt = 0; nt < 32; ++nt)
        for (int ks = 0; ks < 3; ++ks)
            put_bfrag(out + ((nt * 3 + ks) * 32) * 8, nt * 8, ks,
                      [&](int n, int k) -> int8_t { return w[n * 96 + k_to_unit(k)]; });
}

extern "C" int w8a8f_supported(void) {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
    return major >= 8;
}

extern "C" int w8a8f_frontend(const void *x, int8_t *out, int N, int T,
                              const w8a8f_frontend_params *p, void *stream) {
    if (N <= 0 || T <= 0) return 0;
    DeviceInfo &d = device_info();
    const int Tp = (T - 1) / 6 + 1;
    const int64_t items = (int64_t)N * ((Tp + FT - 1) / FT);
    const int grid = (int)(items < (int64_t)d.sms * 2 ? items : (int64_t)d.sms * 2);
    frontend_kernel<<<grid, F_THREADS, sizeof(FrontSmem), (cudaStream_t)stream>>>(
        reinterpret_cast<const __half *>(x), out, N, T, Tp, *p);
    return (int)cudaGetLastError();
}

// Warps per tile (S) for a batch of `tiles` 16-chunk tiles on `sms` SMs, from the tiles each SM
// must hold, t = ceil(tiles / sms), with one CTA of <= LSTM_MAX_WARPS warps per SM:
//   t <= 1                  S = 6: one tile per SM, shortest per-step critical path
//   t <= LSTM_MAX_WARPS / 2 S = 2: still one wave, half the serial work per step
//   more                    S = 1: the original layout and launch, unchanged.  On Orin Nano
//                           S = 2 is 11-12% slower for t = 5..8 (one S = 1 wave).
static int lstm_split(int tiles, int sms) {
    if (g_lstm_force_split < 0) {
        const char *e = getenv("SLORADO_W8A8_LSTM_SPLIT");
        const int v = e ? atoi(e) : 0;
        g_lstm_force_split = (v == 1 || v == 2 || v == 3 || v == 6) ? v : 0;
    }
    if (g_lstm_force_split > 0) return g_lstm_force_split;
    const int t = (tiles + sms - 1) / sms;
    if (t <= 1) return 6;
    if (t <= LSTM_MAX_WARPS / 2) return 2;
    return 1;
}

template <int S>
static void launch_lstm(int tiles, int sms, const int8_t *x, int8_t *y, int N, int Tp, int reverse,
                        const uint8_t *frag, const float *epi, float inv_sh, cudaStream_t stream) {
    // tiles per CTA: spread over all SMs, at most LSTM_MAX_WARPS / S (the original launch for
    // S = 1).  Not spread evenly over waves: a CTA's time is a step function of its warps (one
    // per SM sub-partition is nearly free, measured on Orin: 1-4 tiles ~24 ms, 5-8 ~39 ms), so
    // full CTAs plus one small last CTA (39 + 24) beat two medium ones (39 + 39).
    int gpc = (tiles + sms - 1) / sms;
    gpc = gpc < 1 ? 1 : (gpc > LSTM_MAX_WARPS / S ? LSTM_MAX_WARPS / S : gpc);
    const int grid = (tiles + gpc - 1) / gpc;
    const int smem = LSTM_SMEM + (S > 1 ? gpc * 2 * LSTM_HTILE : 0);
    lstm_kernel<S><<<grid, gpc * S * 32, smem, stream>>>(x, y, N, Tp, reverse, frag, epi, inv_sh);
}

// Force the LSTM split (1, 2, 3 or 6 warps per tile); 0 restores the automatic choice.
// For tests and benchmarks; the environment variable SLORADO_W8A8_LSTM_SPLIT does the same.
extern "C" void w8a8f_lstm_force_split(int split) {
    g_lstm_force_split = (split == 1 || split == 2 || split == 3 || split == 6) ? split : 0;
}

extern "C" int w8a8f_lstm(const int8_t *x, int8_t *y, int N, int Tp, int reverse,
                          const uint8_t *frag, const float *epi, float inv_sh, void *stream) {
    if (N <= 0 || Tp <= 0) return 0;
    DeviceInfo &d = device_info();
    const int tiles = (N + 15) / 16;
    const cudaStream_t s = (cudaStream_t)stream;
    switch (lstm_split(tiles, d.sms)) {
        case 6: launch_lstm<6>(tiles, d.sms, x, y, N, Tp, reverse, frag, epi, inv_sh, s); break;
        case 3: launch_lstm<3>(tiles, d.sms, x, y, N, Tp, reverse, frag, epi, inv_sh, s); break;
        case 2: launch_lstm<2>(tiles, d.sms, x, y, N, Tp, reverse, frag, epi, inv_sh, s); break;
        default: launch_lstm<1>(tiles, d.sms, x, y, N, Tp, reverse, frag, epi, inv_sh, s); break;
    }
    return (int)cudaGetLastError();
}

extern "C" int w8a8f_linear(const int8_t *h, void *out, int64_t rows, const uint8_t *frag,
                            const float *dq, float clamp_lo, float clamp_hi, void *stream) {
    if (rows <= 0) return 0;
    DeviceInfo &d = device_info();
    const int64_t tiles = (rows + 15) / 16;
    int64_t grid = (tiles + LIN_WARPS - 1) / LIN_WARPS;
    if (grid > (int64_t)d.sms * 8) grid = (int64_t)d.sms * 8;
    linear_kernel<<<(int)grid, LIN_WARPS * 32, 0, (cudaStream_t)stream>>>(
        h, reinterpret_cast<__half *>(out), rows, frag, dq, clamp_lo, clamp_hi);
    return (int)cudaGetLastError();
}
