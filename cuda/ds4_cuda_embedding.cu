/**
 * ds4_cuda_embedding.cu — Token Embedding and Output Projection Kernels
 *
 * Embedding lookup: token -> F8 vector
 * Output projection: hidden -> logits
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

__global__ void token_embedding_fp8_kernel(
    const uint8_t *d_embedding,  /* [n_vocab][n_embd] FP8 */
    const int32_t *d_tokens,     /* [n_tokens] */
    float *d_output,             /* [n_tokens][n_embd] FP32 */
    int n_vocab,
    int n_embd,
    int n_tokens)
{
    int tid = blockIdx.x;
    if (tid >= n_tokens) return;
    int token = d_tokens[tid];
    if (token < 0 || token >= n_vocab) token = 0;
    const uint8_t *row = d_embedding + (size_t)token * n_embd;
    float *out = d_output + (size_t)tid * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        uint8_t w = row[i];
        float val;
        if (w == 0) val = 0.0f;
        else {
            unsigned s = (w >> 7) & 1;
            unsigned e = (w >> 3) & 0xF;
            unsigned m = w & 0x7;
            val = (e == 0) ? (float)m / 64.0f
                           : (float)(m + 8) / 8.0f * __powf(2.0f, (int)e - 7);
            if (s) val = -val;
        }
        out[i] = val;
    }
}

__global__ void output_projection_fp8_kernel(
    const float *d_hidden,   /* [n_embd] FP32 */
    const uint8_t *d_output_w, /* [n_vocab][n_embd] FP8 */
    float *d_logits,         /* [n_vocab] FP32 */
    int n_vocab,
    int n_embd)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vocab) return;
    float sum = 0.0f;
    const uint8_t *w_row = d_output_w + (size_t)v * n_embd;
    for (int i = 0; i < n_embd; i++) {
        uint8_t w = w_row[i];
        float w_val;
        if (w == 0) w_val = 0.0f;
        else {
            unsigned s = (w >> 7) & 1;
            unsigned e = (w >> 3) & 0xF;
            unsigned m = w & 0x7;
            w_val = (e == 0) ? (float)m / 64.0f
                             : (float)(m + 8) / 8.0f * __powf(2.0f, (int)e - 7);
            if (s) w_val = -w_val;
        }
        sum += w_val * d_hidden[i];
    }
    d_logits[v] = sum;
}

extern "C" {

void launch_token_embedding(
    const uint8_t *d_embedding, const int32_t *d_tokens,
    float *d_output, int n_vocab, int n_embd, int n_tokens)
{
    int threads = 256;
    int blocks = n_tokens;
    token_embedding_fp8_kernel<<<blocks, threads>>>(
        d_embedding, d_tokens, d_output, n_vocab, n_embd, n_tokens);
}

void launch_output_projection(
    const float *d_hidden, const uint8_t *d_output_w,
    float *d_logits, int n_vocab, int n_embd)
{
    int threads = 256;
    int blocks = (n_vocab + threads - 1) / threads;
    output_projection_fp8_kernel<<<blocks, threads>>>(
        d_hidden, d_output_w, d_logits, n_vocab, n_embd);
}

} // extern "C"
