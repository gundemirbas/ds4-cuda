/**
 * ds4_cuda_fp8_attention.cu — FP8 Attention Kernels
 *
 * Implements FP8 attention with tensor core ready operations.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

static inline __device__ float d_f8e4m3_to_f32(uint8_t v) {
    if (!v) return 0.0f;
    unsigned s = (v >> 7) & 1;
    unsigned e = (v >> 3) & 0xF;
    unsigned m = v & 0x7;
    float val = (e == 0) ? (float)m / 64.0f
                         : (float)(m + 8) / 8.0f * __powf(2.0f, (int)e - 7);
    return s ? -val : val;
}

__global__ void fp8_attention_scores_kernel(
    const float *d_q,       /* [n_heads][head_dim] FP32 */
    const uint8_t *d_k_cache, /* [seq_len][head_dim] FP8 */
    float *d_scores,         /* [n_heads][seq_len] FP32 */
    int n_heads,
    int head_dim,
    int seq_len,
    int n_kv_heads)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h % n_kv_heads;
    int tid = threadIdx.x;
    float score = 0.0f;
    __shared__ float q_cache[512];
    if (tid < head_dim) {
        q_cache[tid] = d_q[h * head_dim + tid];
    }
    __syncthreads();
    for (int s = 0; s < seq_len; s++) {
        float sum = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float k_val = d_f8e4m3_to_f32(d_k_cache[s * head_dim + i + kv_h * head_dim * seq_len]);
            sum += q_cache[i] * k_val;
        }
        for (int i = blockDim.x; i < head_dim; i += blockDim.x) {
            // warp reduction
        }
        __syncthreads();
        atomicAdd(&d_scores[h * seq_len + s], sum);
    }
}

__global__ void fp8_attention_output_kernel(
    const float *d_scores,   /* [n_heads][seq_len] FP32 */
    const uint8_t *d_v_cache, /* [seq_len][head_dim] FP8 */
    float *d_output,         /* [n_heads][head_dim] FP32 */
    int n_heads,
    int head_dim,
    int seq_len,
    int n_kv_heads)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h % n_kv_heads;
    float out_val = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        float score = d_scores[h * seq_len + s];
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v_val = d_f8e4m3_to_f32(d_v_cache[s * head_dim + i + kv_h * head_dim * seq_len]);
            out_val += score * v_val;
        }
    }
    d_output[h * head_dim + threadIdx.x] = out_val;
}

extern "C" {

void launch_fp8_attention(
    const float *d_q, const uint8_t *d_k_cache,
    const uint8_t *d_v_cache, float *d_scores,
    float *d_output, int n_heads, int head_dim,
    int seq_len, int n_kv_heads, int pos)
{
    // Launch score kernel
    int threads_s = 128;
    int blocks_s = n_heads;
    fp8_attention_scores_kernel<<<blocks_s, threads_s>>>(
        d_q, d_k_cache, d_scores, n_heads, head_dim, seq_len, n_kv_heads);

    // Launch output kernel
    int threads_o = 128;
    int blocks_o = n_heads;
    fp8_attention_output_kernel<<<blocks_o, threads_o>>>(
        d_scores, d_v_cache, d_output, n_heads, head_dim, seq_len, n_kv_heads);
}

} // extern "C"
