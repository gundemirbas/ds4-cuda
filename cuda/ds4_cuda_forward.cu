/* ds4_cuda_forward.cu — CUDA kernels for NVFP4 inference forward pass */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float d_f8e4m3(uint8_t v) {
    if (!v) return 0.0f;
    unsigned s = (v >> 7) & 1u, e = (v >> 4) & 7u, m = v & 0xFu;
    float val = (e == 0) ? (float)m / 8.0f : (float)(m + 16) / 16.0f * ldexpf(1.0f, (int)e - 7);
    return s ? -val : val;
}

__device__ __forceinline__ float d_e2m1(uint8_t v) {
    const float t[16] = {0,0,1,1.5f,2,3,4,6, -0,-0,-1,-1.5f,-2,-3,-4,-6};
    return t[v & 0xFu];
}

/* ========================================================================
 * 1. GEMV NVFP4 (UNPACKED U8): y[M] = W[M,K] × x[K]
 *    Each E2M1 value is stored as 1 byte (lower nibble = value)
 *    Scales: F8_E4M3, 1 per group of 8 values
 * ======================================================================== */
__global__ void gemv_nvfp4_kernel(const float *x, const uint8_t *w, const uint8_t *ws,
                                   float *y, int M, int K, float ws2) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    int scales_per_row = K / 8;  /* 1 scale per 8 values */
    for (int i = 0; i < K; i++) {
        uint8_t p = w[row * K + i];
        float sc = d_f8e4m3(ws[row * scales_per_row + i/8]);
        sum += d_e2m1(p) * ws2 * sc * x[i];
    }
    y[row] = sum;
}

extern "C"
void launch_gemv_nvfp4(const float *x, const uint8_t *w, const uint8_t *ws,
                        float *y, int M, int K, float ws2) {
    gemv_nvfp4_kernel<<<(M+255)/256, 256>>>(x, w, ws, y, M, K, ws2);
}

/* ========================================================================
 * 2. GEMV F8_E4M3: y[M] = W[M,K] × x[K]
 * ======================================================================== */
__global__ void gemv_f8e4m3_kernel(const float *x, const uint8_t *w,
                                    float *y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += d_f8e4m3(w[row * K + i]) * x[i];
    }
    y[row] = sum;
}

extern "C"
void launch_gemv_f8e4m3(const float *x, const uint8_t *w, float *y, int M, int K) {
    gemv_f8e4m3_kernel<<<(M+255)/256, 256>>>(x, w, y, M, K);
}

/* ========================================================================
 * 2b. GROUPED GEMV F8_E4M3: y[ngroups * os] = Σ_{g} W_g[os, gs] × x_g[gs]
 *     Splits input x[ngroups * gs] into ngroups groups of size gs,
 *     applies a per-group F8_E4M3 gemv with output size os per group.
 *     Weight layout: W[ngroups * os, gs] stored contiguously.
 * ======================================================================== */
__global__ void gemv_grouped_f8e4m3_kernel(const float *x, const uint8_t *w,
                                            float *y, int ngroups, int gs, int os) {
    int g = blockIdx.x;  /* one block per group */
    int tid = threadIdx.x;
    const uint8_t *wg = w + g * os * gs;  /* weight sub-block for this group */
    const float *xg = x + g * gs;         /* input sub-block for this group */
    for (int row = tid; row < os; row += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < gs; i++) {
            sum += d_f8e4m3(wg[row * gs + i]) * xg[i];
        }
        y[g * os + row] = sum;
    }
}

extern "C"
void launch_gemv_grouped_f8e4m3(const float *x, const uint8_t *w, float *y,
                                 int ngroups, int gs, int os) {
    gemv_grouped_f8e4m3_kernel<<<ngroups, 256>>>(x, w, y, ngroups, gs, os);
}

/* ========================================================================
 * 3. RMSNorm
 * ======================================================================== */
__global__ void rms_norm_kernel(const float *x, float *out, const float *weight,
                                 int n, float eps) {
    extern __shared__ float s[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[row * n + i];
        sum += v * v;
    }
    s[tid] = sum;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float rrms = rsqrtf(s[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[row * n + i] = x[row * n + i] * rrms * weight[i];
    }
}

extern "C"
void launch_rms_norm(const float *x, float *out, const float *weight,
                     int rows, int n, float eps) {
    int threads = 256;
    if (n < threads) threads = n;
    rms_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(x, out, weight, n, eps);
}

/* ========================================================================
 * 4. RoPE
 * ======================================================================== */
__global__ void rope_kernel(float *q, float *k, int n_heads, int head_dim,
                             int pos, float freq_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = n_heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    int dp = d % half;
    float freq = 1.0f / powf(freq_base, (float)(2 * dp) / (float)head_dim);
    float theta = (float)pos * freq;
    float ct = cosf(theta), st = sinf(theta);
    float *qb = q + h * head_dim;
    float a = qb[dp], b = qb[dp + half];
    qb[dp] = a * ct - b * st;
    qb[dp + half] = a * st + b * ct;
    if (h == 0 && k) {
        float a2 = k[dp], b2 = k[dp + half];
        k[dp] = a2 * ct - b2 * st;
        k[dp + half] = a2 * st + b2 * ct;
    }
}

extern "C"
void launch_rope(float *q, float *k, int n_heads, int head_dim, int pos, float freq_base) {
    int total = n_heads * head_dim;
    rope_kernel<<<(total+255)/256, 256>>>(q, k, n_heads, head_dim, pos, freq_base);
}

/* ========================================================================
 * 5. SiLU
 * ======================================================================== */
__global__ void silu_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}
extern "C"
void launch_silu(float *x, int n) { silu_kernel<<<(n+255)/256, 256>>>(x, n); }

/* ========================================================================
 * 6. Element-wise multiply
 * ======================================================================== */
__global__ void emul_kernel(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] *= b[i];
}
extern "C"
void launch_emul(float *a, const float *b, int n) { emul_kernel<<<(n+255)/256, 256>>>(a, b, n); }

/* ========================================================================
 * 7. Top-k from 256 (bitonic sort, k=6)
 * ======================================================================== */
__global__ void topk_256_kernel(const float *scores, int *indices, int k) {
    __shared__ float sv[256];
    __shared__ int si[256];
    int tid = threadIdx.x;
    if (tid < 256) { sv[tid] = scores[tid]; si[tid] = tid; }
    __syncthreads();
    /* Selection sort: find top k by iterating k times, each time thread 0 finds the max */
    for (int i = 0; i < k; i++) {
        __shared__ int best_idx_s;
        __shared__ float best_val_s;
        if (tid == 0) { best_idx_s = -1; best_val_s = -1e38f; }
        __syncthreads();
        /* Each thread checks its elements */
        float local_best = -1e38f;
        int local_best_idx = -1;
        for (int j = tid; j < 256; j += blockDim.x) {
            if (sv[j] > local_best) {
                local_best = sv[j];
                local_best_idx = j;
            }
        }
        /* Reduce to find global best */
        __shared__ float tval[256];
        __shared__ int tidx[256];
        tval[tid] = local_best;
        tidx[tid] = local_best_idx;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) {
                if (tval[tid] < tval[tid + s2]) {
                    tval[tid] = tval[tid + s2];
                    tidx[tid] = tidx[tid + s2];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            indices[i] = si[tidx[0]];
            sv[tidx[0]] = -1e38f; /* remove from future consideration */
        }
        __syncthreads();
    }
}
extern "C"
void launch_topk_256(const float *scores, int *indices, int k) {
    topk_256_kernel<<<1, 256>>>(scores, indices, k);
}

/* ========================================================================
 * 8. BF16 → F32
 * ======================================================================== */
__global__ void bf16_to_f32_kernel(const uint16_t *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t v = in[i];
    unsigned s = (v >> 15) & 1u, e = (v >> 7) & 0xFFu, m = v & 0x7Fu;
    float val;
    if (e == 0) val = ldexpf((float)m, -126);
    else if (e == 0xFF) val = (m == 0) ? INFINITY : NAN;
    else val = ldexpf(1.0f + (float)m / 128.0f, (int)e - 127);
    out[i] = s ? -val : val;
}
extern "C"
void launch_bf16_to_f32(const uint16_t *in, float *out, int n) {
    bf16_to_f32_kernel<<<(n+255)/256, 256>>>(in, out, n);
}

/* ========================================================================
 * 9. Residual add
 * ======================================================================== */
__global__ void residual_add_kernel(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}
extern "C"
void launch_residual_add(float *a, const float *b, int n) {
    residual_add_kernel<<<(n+255)/256, 256>>>(a, b, n);
}

/* ========================================================================
 * 10. Copy
 * ======================================================================== */
__global__ void copy_kernel(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}
extern "C"
void launch_copy(float *dst, const float *src, int n) {
    copy_kernel<<<(n+255)/256, 256>>>(dst, src, n);
}

/* ========================================================================
 * 11. GEMV F32
 * ======================================================================== */
__global__ void gemv_f32_kernel(const float *x, const float *w, float *y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) sum += w[row * K + i] * x[i];
    y[row] = sum;
}
extern "C"
void launch_gemv_f32(const float *x, const float *w, float *y, int M, int K) {
    gemv_f32_kernel<<<(M+255)/256, 256>>>(x, w, y, M, K);
}
