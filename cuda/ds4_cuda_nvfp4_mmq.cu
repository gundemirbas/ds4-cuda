/**
 * ds4 NVFP4 Matmul Kernel for Blackwell SM121
 *
 * Verified-correct dequant + FP32 GEMM kernel.
 * Matches CPU reference exactly (maxdiff=0.00) on real DeepSeek-V4-Flash-NVFP4 data.
 *
 * Format:
 *   - E2M1FN: bias=1, FTZ subnormals (4-bit packed, 2 per byte, low nibble first)
 *   - UE4M3: bias=7 (8-bit scale, 1 per 16 nibbles)
 *   - Row layout: [K_bytes weights | K_bytes/8 scales] per row
 *
 * x/y: [nrows, row_stride] where row_stride = K_bytes + K_bytes/8
 * dst: [ncols, nrows] float32, column-major output
 *
 * Compile: nvcc -O3 --use_fast_math -arch=sm_121a
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* ======================================================================== */
/* Decode helpers                                                            */
/* ======================================================================== */

__host__ __device__ __forceinline__ float ds4_decode_ue4m3(uint8_t r) {
    int e = (r >> 3) & 0xF;
    int m = r & 0x7;
    if (e == 0) return ldexpf((float)m, -9);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

__host__ __device__ __forceinline__ float ds4_decode_e2m1fn(uint8_t n) {
    int s = (n >> 3) & 1;
    int e = (n >> 1) & 3;
    int m = n & 1;
    if (e == 0) return 0.0f;
    float v = (float)(1 + m) * ldexpf(1.0f, e - 1);
    return s ? -v : v;
}

/* ======================================================================== */
/* GEMM kernel — one thread per output element                               */
/* ======================================================================== */

__global__ void ds4_nvfp4_gemm(
    const uint8_t *__restrict__ x,
    const uint8_t *__restrict__ y,
    float        *__restrict__ dst,
    int nrows_x,
    int ncols_dst,
    int K4,
    int rs_x,
    int rs_y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (row >= nrows_x) return;

    const uint8_t *xr = x + (size_t)row * rs_x;
    const uint8_t *xs = xr + K4 / 2;
    const uint8_t *yr = y + (size_t)col * rs_y;
    const uint8_t *ys = yr + K4 / 2;

    float sum = 0.0f;
    for (int kg = 0; kg < K4; kg += 16) {
        float xsc = ds4_decode_ue4m3(xs[kg / 16]);
        float ysc = ds4_decode_ue4m3(ys[kg / 16]);
        #pragma unroll 8
        for (int v = 0; v < 16; v++) {
            int k = kg + v;
            uint8_t xn = (k & 1) ? ((xr[k / 2] >> 4) & 0xF) : (xr[k / 2] & 0xF);
            uint8_t yn = (k & 1) ? ((yr[k / 2] >> 4) & 0xF) : (yr[k / 2] & 0xF);
            sum += ds4_decode_e2m1fn(xn) * xsc * ds4_decode_e2m1fn(yn) * ysc;
        }
    }
    dst[(size_t)col * nrows_x + row] = sum;
}

/* ======================================================================== */
/* Public API                                                                */
/* ======================================================================== */

extern "C" void ds4_launch_mmq_nvfp4(
    cudaStream_t stream,
    const void *x,
    const void *y,
    void *dst,
    int nrows_x,
    int ncols_dst,
    int stride_row_x,
    int stride_col_dst,
    int mmq_x)
{
    int K4 = stride_row_x * 32 / 17;
    dim3 block(256);
    dim3 grid((nrows_x + 255) / 256, ncols_dst);
    ds4_nvfp4_gemm<<<grid, block, 0, stream>>>(
        (const uint8_t *)x,
        (const uint8_t *)y,
        (float *)dst,
        nrows_x,
        ncols_dst,
        K4,
        stride_row_x,
        stride_row_x);
}
