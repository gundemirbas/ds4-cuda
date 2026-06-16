/**
 * ds4_cuda_fp8_attention.cuh — FP8 Attention with Tensor Cores
 *
 * Attention computation with FP8 K/V cache:
 *   1. Q × K^T: Q is FP32, K is FP8 → scores FP32 (FP32 accumulate)
 *   2. softmax: FP32 (numerical stability)
 *   3. attn × V: attn is FP32, V is FP8 → output FP32
 *
 * For DeepSeek-V4 MLA (Multi-head Latent Attention):
 *   - n_heads = 64, head_dim = 512
 *   - n_kv_heads = 1 (shared across all heads)
 *   - GQA: each head shares the same K/V
 *
 * Tensor core usage:
 *   - FP8 GEMV: accumulate FP32 (not using tensor cores for GEMV)
 *   - For prefill (batched): use tensor core GEMM
 */

#ifndef DS4_CUDA_FP8_ATTENTION_CUH
#define DS4_CUDA_FP8_ATTENTION_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "ds4_kv_cache.h"

/* ========================================================================
 * FP8 Attention Kernels
 * ======================================================================== */

/**
 * Compute attention scores: scores[h][t] = Q[h] · K[t]
 *
 * For GQA: all heads share the same K cache.
 * Input Q is [n_heads][head_dim], K cache is [seq_len][head_dim] (FP8).
 * Output scores is [n_heads][seq_len] (FP32).
 *
 * Grid: (n_heads, 1, 1)
 * Block: (256, 1, 1)
 */
__global__ void fp8_attention_scores_kernel(
    const float *Q,             /* [n_heads][head_dim] */
    const uint8_t *K_cache,     /* [seq_len][head_dim] FP8 */
    float *scores,              /* [n_heads][seq_len] */
    int n_heads,
    int head_dim,
    int seq_len,
    int n_kv_heads              /* 1 for MLA, n_heads for MHA */
) {
    int h = blockIdx.x;  /* head index */
    if (h >= n_heads) return;

    int tid = threadIdx.x;
    const float *q_ptr = Q + h * head_dim;

    /* Shared memory for reduction */
    __shared__ float sdata[256];

    for (int t = 0; t < seq_len; t++) {
        /* K index: for GQA, all heads use same K */
        int kv_h = h % n_kv_heads;
        const uint8_t *k_ptr = K_cache + t * (n_kv_heads * head_dim) + kv_h * head_dim;

        /* Compute dot product: Q · K[t] */
        float sum = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float q_val = q_ptr[i];
            float k_val = ds4_f8e4m3_to_f32(k_ptr[i]);
            sum += q_val * k_val;
        }

        sdata[tid] = sum;
        __syncthreads();

        /* Parallel reduction */
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            /* Scale by sqrt(head_dim) */
            scores[h * seq_len + t] = sdata[0] / sqrtf((float)head_dim);
        }
        __syncthreads();
    }
}

/**
 * Apply causal mask and softmax.
 * scores[h][t] = softmax(scores[h][t]) for t <= pos, else -inf
 *
 * Grid: (n_heads, 1, 1)
 * Block: (256, 1, 1)
 */
__global__ void fp8_softmax_causal_kernel(
    float *scores,      /* [n_heads][seq_len] */
    int n_heads,
    int seq_len,
    int pos             /* current position (for causal mask) */
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;

    int tid = threadIdx.x;
    float *s = scores + h * seq_len;

    /* Find max for numerical stability */
    float max_val = -1e30f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        if (s[t] > max_val) max_val = s[t];
    }

    __shared__ float shared_max[256];
    shared_max[tid] = max_val;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + i]);
        __syncthreads();
    }
    max_val = shared_max[0];

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        s[t] = expf(s[t] - max_val);
        sum += s[t];
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) shared_sum[tid] += shared_sum[tid + i];
        __syncthreads();
    }
    sum = shared_sum[0];

    /* Normalize */
    for (int t = tid; t <= pos; t += blockDim.x) {
        s[t] /= sum;
    }

    /* Mask positions after pos */
    for (int t = pos + 1 + tid; t < seq_len; t += blockDim.x) {
        s[t] = -1e30f;  /* or 0 after softmax */
    }
    __syncthreads();
}

/**
 * Compute attention output: out[h] = sum_t(attn[h][t] * V[t])
 *
 * V cache is FP8, attn weights are FP32.
 * Output is FP32.
 *
 * Grid: (n_heads, 1, 1)
 * Block: (256, 1, 1)
 */
__global__ void fp8_attention_output_kernel(
    const float *attn,          /* [n_heads][seq_len] */
    const uint8_t *V_cache,     /* [seq_len][n_kv_heads * head_dim] FP8 */
    float *output,              /* [n_heads][head_dim] */
    int n_heads,
    int head_dim,
    int seq_len,
    int n_kv_heads
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;

    int tid = threadIdx.x;
    float *out = output + h * head_dim;
    const float *a = attn + h * seq_len;

    int kv_h = h % n_kv_heads;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float attn_val = a[t];
            uint8_t v_val = V_cache[t * (n_kv_heads * head_dim) + kv_h * head_dim + d];
            sum += attn_val * ds4_f8e4m3_to_f32(v_val);
        }
        out[d] = sum;
    }
}

/* ========================================================================
 * Launch Functions
 * ======================================================================== */

static inline void launch_fp8_attention(
    const float *d_q,           /* [n_heads][head_dim] */
    const uint8_t *d_k_cache,   /* [seq_len][n_kv_heads * head_dim] FP8 */
    const uint8_t *d_v_cache,   /* [seq_len][n_kv_heads * head_dim] FP8 */
    float *d_scores,            /* [n_heads][seq_len] scratch */
    float *d_output,            /* [n_heads][head_dim] */
    int n_heads,
    int head_dim,
    int seq_len,
    int n_kv_heads,
    int pos                     /* current position */
) {
    dim3 grid(n_heads);
    dim3 block(256);

    /* Step 1: Compute attention scores */
    fp8_attention_scores_kernel<<<grid, block>>>(
        d_q, d_k_cache, d_scores,
        n_heads, head_dim, seq_len, n_kv_heads);

    /* Step 2: Apply causal mask and softmax */
    fp8_softmax_causal_kernel<<<grid, block>>>(
        d_scores, n_heads, seq_len, pos);

    /* Step 3: Compute attention output */
    fp8_attention_output_kernel<<<grid, block>>>(
        d_scores, d_v_cache, d_output,
        n_heads, head_dim, seq_len, n_kv_heads);
}

/* ========================================================================
 * MLA-specific: Compressed Attention
 * ======================================================================== */

/**
 * MLA compressed attention: uses low-rank compression for Q and KV.
 *
 * In DeepSeek-V4 MLA:
 *   - Q is compressed: Q_compressed [n_lora_q] → Q [n_heads][head_dim]
 *   - KV is compressed: KV_compressed [n_lora_kv] → K [head_dim], V [head_dim]
 *   - KV cache stores the compressed representation (not expanded)
 *
 * This is more memory-efficient than standard GQA.
 */
__global__ void mla_decompress_k_kernel(
    const float *kv_compressed,  /* [n_lora_kv] (FP32) */
    const float *w_k,           /* [head_dim][n_lora_kv] weight */
    float *k_out,               /* [head_dim] */
    int head_dim,
    int n_lora_kv
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < n_lora_kv; i++) {
        sum += w_k[d * n_lora_kv + i] * kv_compressed[i];
    }
    k_out[d] = sum;
}

__global__ void mla_decompress_v_kernel(
    const float *kv_compressed,  /* [n_lora_kv] (FP32) */
    const float *w_v,           /* [head_dim][n_lora_kv] weight */
    float *v_out,               /* [head_dim] */
    int head_dim,
    int n_lora_kv
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < n_lora_kv; i++) {
        sum += w_v[d * n_lora_kv + i] * kv_compressed[i];
    }
    v_out[d] = sum;
}

#endif /* DS4_CUDA_FP8_ATTENTION_CUH */
