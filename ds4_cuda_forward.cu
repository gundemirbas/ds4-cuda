/* ds4_cuda_forward.cu — CUDA kernels for NVFP4 forward pass */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

static __device__ __forceinline__ float f8e4m3_decode(uint8_t v) {
    if (!v) return 0.0f;
    unsigned s = (v >> 7) & 1u, e = (v >> 4) & 7u, m = v & 0xFu;
    float val = (e == 0) ? (float)m / 8.0f : (float)(m + 16) / 16.0f * ldexpf(1.0f, (int)e - 7);
    return s ? -val : val;
}

static __device__ __forceinline__ float e2m1_decode(uint8_t v) {
    const float t[16] = {0,0,1,1.5f,2,3,4,6, -0,-0,-1,-1.5f,-2,-3,-4,-6};
    return t[v & 0xFu];
}

__global__ void gemv_nvfp4_kernel(const float *x, const uint8_t *w, const uint8_t *ws,
                                   float *y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    float sum = 0.0f;
    int nk = K / 2;
    for (int i = 0; i < nk; i++) {
        uint8_t p = w[row * nk + i];
        float sc = f8e4m3_decode(ws[row * (K/16) + i/8]);
        sum += e2m1_decode(p & 0xF) * 2.f * sc * x[i*2];
        sum += e2m1_decode((p>>4)&0xF) * 2.f * sc * x[i*2+1];
    }
    y[row] = sum;
}

extern "C"
void launch_gemv_nvfp4(const float *x, const uint8_t *w, const uint8_t *ws,
                        float *y, int M, int K) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    gemv_nvfp4_kernel<<<blocks, threads>>>(x, w, ws, y, M, K);
}
