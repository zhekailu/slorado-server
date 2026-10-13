/**
 * w8a8_tanh_approx.cu
 *
 * The GPU's hardware tanh (MUFU.TANH, sm_75+) for the slorado-w8a8-gpu-v1 models whose
 * quantization.json asks for "nonlinearity": "tanh_approx".  It is the same instruction
 * tools/FASTv5.2_quantForGPU/hw_tanh.py runs through Triton, so slorado and the Python
 * reference evaluate identical tanh / sigmoid values.
 *
 *   FP32:  tanh.approx.f32
 *   FP16:  tanh.approx.f16x2 on pairs, tanh.approx.f16 on an odd tail or unaligned input
 *
 * Plain C entry point so the rest of slorado (g++, LibTorch) only needs the declaration.
 */
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void tanh_approx_f32(const float *x, float *y, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float r;
        asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x[i]));
        y[i] = r;
    }
}

__global__ void tanh_approx_f16x2(const uint32_t *x, uint32_t *y, int64_t n2) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n2) {
        uint32_t r;
        asm("tanh.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x[i]));
        y[i] = r;
    }
}

__global__ void tanh_approx_f16(const uint16_t *x, uint16_t *y, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint16_t r;
        asm("tanh.approx.f16 %0, %1;" : "=h"(r) : "h"(x[i]));
        y[i] = r;
    }
}

inline unsigned blocks(int64_t n) { return (unsigned)((n + kThreads - 1) / kThreads); }

}  // namespace

// y = tanh.approx(x) elementwise; x, y contiguous device buffers of n FP32 (is_half = 0)
// or FP16 (is_half = 1) values.  Returns 0 on success, otherwise a cudaError_t.
extern "C" int slorado_tanh_approx(const void *x, void *y, int64_t n, int is_half, void *stream) {
    if (n <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (!is_half) {
        tanh_approx_f32<<<blocks(n), kThreads, 0, s>>>((const float *)x, (float *)y, n);
    } else if (((uintptr_t)x | (uintptr_t)y) % 4 == 0) {
        const int64_t n2 = n / 2;
        if (n2 > 0) {
            tanh_approx_f16x2<<<blocks(n2), kThreads, 0, s>>>((const uint32_t *)x, (uint32_t *)y, n2);
        }
        if (n % 2) {
            tanh_approx_f16<<<1, 1, 0, s>>>((const uint16_t *)x + (n - 1), (uint16_t *)y + (n - 1), 1);
        }
    } else {
        tanh_approx_f16<<<blocks(n), kThreads, 0, s>>>((const uint16_t *)x, (uint16_t *)y, n);
    }
    return (int)cudaGetLastError();
}
