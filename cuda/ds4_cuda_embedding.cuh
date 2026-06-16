/**
 * ds4_cuda_embedding.cuh — Token Embedding Kernel
 *
 * Embedding lookup: token → F8 vector
 * The embedding table is stored in FP8 E4M3 format in the model weights.
 */

#ifndef DS4_CUDA_EMBEDDING_CUH
#define DS4_CUDA_EMBEDDING_CUH

#include <cuda_runtime.h>

/**
 * Token embedding lookup (FP8 → FP32).
 *
 * @param d_embedding   Embedding table [n_vocab][n_embd] (FP8 E4M3)
 * @param d_tokens      Token indices [n_tokens]
 * @param d_output      Output vectors [n_tokens][n_embd] (FP32)
 * @param n_vocab       Vocabulary size
 * @param n_embd        Embedding dimension
 * @param n_tokens      Number of tokens
 */
__global__ void token_embedding_fp8_kernel(
    const uint8_t *d_embedding,  /* [n_vocab][n_embd] FP8 */
    const int32_t *d_tokens,     /* [n_tokens] */
    float *d_output,             /* [n_tokens][n_embd] FP32 */
    int n_vocab,
    int n_embd,
    int n_tokens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_embd;
    if (idx >= total) return;

    int t = idx / n_embd;  /* token index */
    int d = idx % n_embd;  /* embedding dimension */

    int token = d_tokens[t];
    if (token < 0 || token >= n_vocab) {
        d_output[idx] = 0.0f;
        return;
    }

    /* FP8 → FP32 */
    uint8_t v = d_embedding[token * n_embd + d];
    if (!v) { d_output[idx] = 0.0f; return; }

    unsigned s = (v >> 7) & 1;
    unsigned e = (v >> 3) & 0xF;
    unsigned m = v & 0x7;
    float val = (e == 0) ? (float)m / 64.0f
                         : (float)(m + 8) / 8.0f * ldexpf(1.0f, (int)e - 7);
    d_output[idx] = s ? -val : val;
}

/**
 * Token embedding lookup (FP8 → FP8).
 * Stores result in FP8 format for subsequent layers.
 */
__global__ void token_embedding_fp8_out_kernel(
    const uint8_t *d_embedding,  /* [n_vocab][n_embd] FP8 */
    const int32_t *d_tokens,     /* [n_tokens] */
    uint8_t *d_output,           /* [n_tokens][n_embd] FP8 */
    int n_vocab,
    int n_embd,
    int n_tokens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_embd;
    if (idx >= total) return;

    int t = idx / n_embd;
    int d = idx % n_embd;

    int token = d_tokens[t];
    if (token < 0 || token >= n_vocab) {
        d_output[idx] = 0;
        return;
    }

    d_output[idx] = d_embedding[token * n_embd + d];
}

/**
 * Output projection: hidden states → logits.
 * This is typically a large matmul [n_vocab][n_embd].
 * For now, we use FP8 GEMV (one token at a time).
 */
__global__ void output_projection_fp8_kernel(
    const float *d_hidden,       /* [n_embd] FP32 */
    const uint8_t *d_output_w,   /* [n_vocab][n_embd] FP8 */
    float *d_logits,             /* [n_vocab] FP32 */
    int n_vocab,
    int n_embd)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vocab) return;

    float sum = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        uint8_t w = d_output_w[v * n_embd + i];
        float w_val = 0.0f;
        if (w) {
            unsigned s = (w >> 7) & 1;
            unsigned e = (w >> 3) & 0xF;
            unsigned m = w & 0x7;
            w_val = (e == 0) ? (float)m / 64.0f
                             : (float)(m + 8) / 8.0f * ldexpf(1.0f, (int)e - 7);
            if (s) w_val = -w_val;
        }
        sum += w_val * d_hidden[i];
    }
    d_logits[v] = sum;
}

/* Launch helpers */
static inline void launch_token_embedding(
    const uint8_t *d_embedding,
    const int32_t *d_tokens,
    float *d_output,
    int n_vocab,
    int n_embd,
    int n_tokens)
{
    int total = n_tokens * n_embd;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    token_embedding_fp8_kernel<<<blocks, threads>>>(
        d_embedding, d_tokens, d_output,
        n_vocab, n_embd, n_tokens);
}

static inline void launch_output_projection(
    const float *d_hidden,
    const uint8_t *d_output_w,
    float *d_logits,
    int n_vocab,
    int n_embd)
{
    int threads = 256;
    int blocks = (n_vocab + threads - 1) / threads;
    output_projection_fp8_kernel<<<blocks, threads>>>(
        d_hidden, d_output_w, d_logits,
        n_vocab, n_embd);
}

#endif /* DS4_CUDA_EMBEDDING_CUH */
