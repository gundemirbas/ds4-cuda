# AGENTS.md — ds4-cuda: NVFP4 Port for DeepSeek-V4-Flash-NVFP4

## 1. Proje Amacı

**ds4-cuda**, antirez/ds4 (DwarfStar) projesinin NVIDIA **DeepSeek-V4-Flash-NVFP4** modeliyle çalışacak bir **CUDA-only portudur**.  
Hedef donanım: **DGX Spark (NVIDIA GB10, sm_121a)**.

### Temel Yaklaşım
**antirez/ds4 orijinal kodunu temel alıyoruz. Eski hiçbir özellik kaybolmaz.**
Yeni özellikler **ek olarak** eklenir, mevcut kod **değiştirilmez**:

```
KORUNAN (antirez/ds4):
  ✅ GGUF loading       → ds4_gguf_load() çalışmaya devam eder
  ✅ Metal backend      → ds4_metal.m macOS'ta çalışır
  ✅ ROCm backend       → ds4_rocm.cu AMD'de çalışır
  ✅ CUDA backend       → ds4_cuda.cu Q8_0 çalışır
  ✅ Dağıtık hesaplama  → ds4_distributed.c korunur
  ✅ CPU fallback       → ds4.c içinde korunur
  ✅ Q8_0/FP16/IQ2/Q2_K → Tüm quantization formatları korunur
  ✅ SSD streaming      → Expert cache korunur

EKLENEN YENİ ÖZELLİKLER:
  🆕 Safetensors loading → GGUF'e alternatif (mmap + SSD streaming korunur)
  🆕 FP8 KV cache        → FP16'e alternatif (2× memory tasarrufu)
  🆕 NVFP4 matmul        → Q8_0'e alternatif (tensor core, dequant yok)
  🆕 FP8 matmul          → Q8_0'e alternatif (tensor core)
  🆕 FP8 attention       → FP16 attention'a alternatif
```

**Seçim mekanizması:** Model formatına göre otomatik seçim
- GGUF model → GGUF loading + Q8_0 + FP16 attention
- Safetensors model → Safetensors loading + NVFP4/FP8 + FP8 attention

---

## 2. Mevcut Durum

| Aşama | Durum | Açıklama |
|-------|-------|----------|
| Safetensors parser | ✅ Tamamlandı | `ds4.c` içine merge edildi (eski: `ds4_safetensors.c` 434 satır) |
| FP8 KV cache CPU | ✅ Tamamlandı | `ds4.c` içine merge edildi (eski: `ds4_kv_cache.c` 104 satır) |
| FP8 KV cache GPU | ✅ Tamamlandı | `ds4_cuda.cu` içine merge edildi + `fp32_to_fp8_e4m3_kernel` implemente edildi |
| config.json parser | ✅ Tamamlandı | `ds4.c` içine merge edildi (eski: `ds4_model_config.c` 270 satır) |
| Expert LRU cache | ✅ Tamamlandı | `ds4.c` içine merge edildi (eski: `ds4_expert_cache.c` 188 satır) |
| NVFP4 MMQ kernel | ✅ Tamamlandı | `ds4_cuda.cu` içine merge edildi (eski: `ds4_cuda_nvfp4_mmq.cu` 106 satır) |
| Forward kernels | ✅ Tamamlandı | `ds4_cuda.cu` içine merge edildi (eski: `ds4_cuda_forward.cu` 287 satır) |
| FP8 attention | ✅ Tamamlandı | `ds4_cuda.cu` içine merge edildi (eski: `ds4_cuda_fp8_attention.cu` 98 satır) |
| Embedding kernels | ✅ Tamamlandı | `ds4_cuda.cu` içine merge edildi (eski: `ds4_cuda_embedding.cu` 92 satır) |
| **Merge → ds4.c** | ✅ Tamamlandı | Safetensors loader + FP8 KV cache + NVFP4 dispatch + 4 kaynak merge |
| **Merge → ds4_cuda.cu** | ✅ Tamamlandı | NVFP4/FP8 kernel wrapper'ları + 6 kaynak merge |
| **Merge → ds4.h** | ✅ Tamamlandı | Safetensors/FP8/NVFP4 typedef'leri + 4 header merge |
| **Merge → ds4_gpu.h** | ✅ Tamamlandı | NVFP4/FP8 fonksiyon prototipleri |
| **SSD streaming (safetensors)** | ✅ Tamamlandı | PROT_READ + cudaHostRegister fallback + hmm_direct |
| **FP8 KV cache (auto)** | ✅ Tamamlandı | Safetensors algılayınca otomatik etkin |
| **Distributed (safetensors)** | ✅ Tamamlandı | Aynı SSD streaming mekanizması |
| **FP8 KV cache (GPU)** | ✅ Tamamlandı | `fp32_to_fp8_e4m3_kernel` + `ds4_fp8_kv_cache_append_gpu` + `ds4_gpu_kv_fp8_quantize_append_tensor` |
| **End-to-end test** | ⏳ Bekliyor | Model yükleme + inference |

### ⚠️ Tespit Edilen Sorunlar (Debug Sonucu)

| # | Sorun | Durum | Önem |
|---|-------|-------|------|
| 1 | **Q8_0 hardcoded dispatch** — Safetensors modelde tüm matmul çağrıları Q8_0 formatı varsayıyordu | ✅ Çözüldü | Kritik |
| 2 | **F8/NVFP4 weight pointer** — `model_map + offset` yerine `cuda_model_range_ptr` kullanılmıyordu | ✅ Çözüldü | Kritik |
| 3 | **E4M3 decode hatası** — `d_f8e4m3` E3M4 formatında kodlanmıştı (3-bit exp, 4-bit mantissa) | ✅ Çözüldü | Kritik |
| 4 | **F8_E4M3 block scales eksik** — Attention weight'lerinde F8_E8M0 block_scale tensor'leri var ama GEMV kernel'ı bunları kullanmıyor | ⏳ Çözülmedi | Yüksek |
| 5 | **NVFP4 scale tensor ayrılığı** — Safetensors'ta weight ve scale ayrı tensor olarak saklanıyor ama kernel interleave bekliyor | ⏳ Çözülmedi | Kritik |

### Safetensors Weight Formatları (Model Analizi)

```
Attention weights:
  layers.X.attn.wq_a.weight:  F8_E4M3, shape=[1024, 4096]
  layers.X.attn.wq_a.scale:   F8_E8M0, shape=[8, 32]     ← AYRI tensor!
  layers.X.attn.wq_b.weight:  F8_E4M3, shape=[32768, 1024]
  layers.X.attn.wq_b.scale:   F8_E8M0, shape=[32, 64]     ← AYRI tensor!
  layers.X.attn.wkv.weight:   F8_E4M3, shape=[512, 4096]
  layers.X.attn.wkv.scale:    F8_E8M0, shape=[4, 32]      ← AYRI tensor!
  layers.X.attn.wo_a.weight:  F8_E4M3, shape=[8192, 4096]
  layers.X.attn.wo_a.scale:   F8_E8M0, shape=[64, 32]     ← AYRI tensor!
  layers.X.attn.wo_b.weight:  F8_E4M3, shape=[4096, 8192]
  layers.X.attn.wo_b.scale:   F8_E8M0, shape=[32, 64]     ← AYRI tensor!

Expert weights:
  layers.X.ffn.experts.Y.w1.weight:       U8, shape=[2048, 2048]        ← NVFP4 packed
  layers.X.ffn.experts.Y.w1.weight_scale: F8_E4M3, shape=[2048, 256]    ← AYRI tensor!
  layers.X.ffn.experts.Y.w1.weight_scale_2: F32, shape=[]               ← Global scale
  layers.X.ffn.experts.Y.w1.input_scale:  F32, shape=[]                  ← Input scale
```

### Çözüm Gereksinimleri

**F8_E4M3 block scale desteği:**
- GEMV kernel'ına `scale_offset` parametresi ekle
- F8_E8M0 decode fonksiyonu yaz (8-bit exponent-only format)
- Weight + scale birlikte okunmalı: `output = sum(w[i] * scale[block] * x[i])`

**NVFP4 separate scale desteği:**
- `ds4_tensor` yapısına `scale_offset` field'ı ekle
- Safetensors loader'da scale tensor offset'ini bul
- GEMV wrapper'ına scale pointer'ı geçir
- Kernel'de `ws` pointer'ı ayrı tensor'dan okunmalı

---

## 3. antirez/ds4 Orijinal Yapısı

**Flat structure** — tüm kaynak dosyalar root dizinde.

```
antirez/ds4/
├── ds4.c                # 27791 satır — Ana engine
├── ds4.h                # Ana API typedef'leri
├── ds4_gpu.h            # GPU API (~100+ prototip)
├── ds4_cuda.cu          # 13256 satır — CUDA kernel'lar
├── ds4_metal.m          # Metal backend
├── ds4_rocm.cu          # ROCm backend
├── ds4_distributed.c    # Dağıtık hesaplama
├── ds4_cli.c            # CLI main()
├── ds4_server.c         # HTTP server
├── ds4_bench.c          # Benchmark
├── ds4_eval.c           # Evaluation
├── ds4_agent.c          # Agent
├── ds4_web.c            # Web UI
├── ds4_help.c / .h      # Help text
├── ds4_kvstore.c / .h   # KV store
├── ds4_ssd.c / .h       # SSD helpers
├── ds4_iq2_tables_cuda.inc
├── ds4_streaming_hotlist.inc
├── rax.c / rax.h / rax_malloc.h
├── linenoise.c / linenoise.h
├── metal/
├── Makefile
└── README.md
```

### Kritik Noktalar
- `ds4.c` içinde **214 static GPU/Metal fonksiyonu** var
- `ds4_gpu.h` **~100+ fonksiyon prototipi** içerir
- `ds4_cuda.cu` ds4_gpu.h'deki fonksiyonların CUDA implementasyonlarını içerir
- `ds4.c` → `ds4_gpu_*` çağrıları → `ds4_cuda.cu` implementasyonları

---

## 4. Değişiklik Stratejisi

### 4.1 Değişiklik Haritası (sadece 4 dosya)

| Dosya | Ne değişiyor | Satır |
|-------|-------------|-------|
| `ds4.h` | Safetensors + FP8 API typedef'leri + header merge | ~50 |
| `ds4_gpu.h` | NVFP4/FP8 fonksiyon prototipleri | ~30 |
| `ds4.c` | Safetensors loader + FP8 KV cache + NVFP4 dispatch + kaynak merge | ~500 |
| `ds4_cuda.cu` | NVFP4/FP8 kernel wrapper'ları + kaynak merge | ~300 |

### 4.2 Merge Stratejisi (16 → 4 dosya) — Tamamlandı

```
YENİ KOD                    → HEDEF DOSYA         → DURUM
────────────────────────────────────────────────────────────
ds4_safetensors.c (434)     → ds4.c içine          ✅ Merge edildi, kaynak silindi
ds4_safetensors.h (87)      → ds4.h içine          ✅ Merge edildi, kaynak silindi
ds4_kv_cache.c (104)        → ds4.c içine          ✅ Merge edildi, kaynak silindi
ds4_kv_cache.cu (124)       → ds4_cuda.cu içine    ✅ Merge edildi, kaynak silindi
ds4_kv_cache.h (67)         → ds4.h içine          ✅ Merge edildi, kaynak silindi
ds4_model_config.c (270)    → ds4.c içine          ✅ Merge edildi, kaynak silindi
ds4_model_internal.h        → ds4.h içine          ✅ Merge edildi, kaynak silindi
ds4_expert_cache.c (188)    → ds4.c içine          ✅ Merge edildi, kaynak silindi
ds4_ssd_streaming.h (35)    → ds4.h içine          ✅ Merge edildi, kaynak silindi
ds4_cuda_nvfp4_mmq.cu       → ds4_cuda.cu içine    ✅ Merge edildi, kaynak silindi
ds4_cuda_forward.cu (287)   → ds4_cuda.cu içine    ✅ Merge edildi, kaynak silindi
ds4_cuda_fp8_attention.cu   → ds4_cuda.cu içine    ✅ Merge edildi, kaynak silindi
ds4_cuda_embedding.cu (92)   → ds4_cuda.cu içine    ✅ Merge edildi, kaynak silindi
ds4_cuda_fp8_attention.cuh  → (kullanılmıyor)      🗑️ Silindi (içerik ds4_cuda.cu'ya merge)
ds4_cuda_embedding.cuh      → (kullanılmıyor)      🗑️ Silindi (içerik ds4_cuda.cu'ya merge)
ds4_layer_forward.cuh       → 🗑️ Silindi (ölü kod, engine kullanmıyor)
```

### 4.3 Korunan Orijinal Kod

**Hiç değiştirilmez:**
```
ds4_cli.c, ds4_server.c, ds4_bench.c, ds4_eval.c, ds4_agent.c
ds4_help.c, ds4_kvstore.c, ds4_web.c, ds4_ssd.c
ds4_distributed.c, ds4_distributed.h
rax.c, linenoise.c
ds4_help.h, ds4_kvstore.h, ds4_ssd.h, ds4_web.h
linenoise.h, rax.h, rax_malloc.h
ds4_cuda_runtime.cuh, ds4_iq2_tables_cuda.inc, ds4_streaming_hotlist.inc
metal/
```

---

## 5. Adım Adım Execution Planı

### ~~Adım 1: ds4.h Değişiklikleri~~ ✅ Tamamlandı

**Nerede:** `ds4.h` — mevcut typedef'lerin yanına (dosya sonu)

**Eklenecek kod (referans):**
```c
/* =========================================================================
 * Safetensors support (ds4-cuda)
 * ========================================================================= */
#ifndef DS4_SAFETensors_H
#define DS4_SAFETensors_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    SST_DTYPE_F32 = 0,
    SST_DTYPE_F16,
    SST_DTYPE_BF16,
    SST_DTYPE_F8_E4M3,
    SST_DTYPE_NVFP4,
    SST_DTYPE_MXFP4
} sst_dtype;

typedef struct {
    const char *name;
    uint32_t ndim;
    uint64_t shape[4];
    sst_dtype dtype;
    uint64_t data_offset;
    uint64_t data_size;
} sst_tensor;

typedef struct {
    char *name;
    int fd;
    void *map;
    uint64_t map_size;
    uint64_t data_offset;
    uint64_t n_tensors;
    sst_tensor *tensors;
} sst_model;

typedef struct {
    sst_model **models;
    uint64_t n_models;
} sst_sharded_model;

sst_sharded_model *sst_sharded_model_load(const char *model_dir);
void sst_sharded_model_free(sst_sharded_model *shard);
sst_model *sst_model_load(const char *path);
void sst_model_free(sst_model *m);
sst_tensor *sst_model_find_tensor(sst_model *m, const char *name);
uint64_t sst_tensor_elements(const sst_tensor *t);
uint64_t sst_tensor_bytes(const sst_tensor *t);

#endif /* DS4_SAFETensors_H */

/* =========================================================================
 * FP8 KV Cache (ds4-cuda)
 * ========================================================================= */
#ifndef DS4_KV_CACHE_H
#define DS4_KV_CACHE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    uint8_t *fp8_data;     // FP8 E4M3 KV data
    float *scales;         // Per-head scales
    uint32_t capacity;     // Max sequence length
    uint32_t head_dim;     // Head dimension
    uint32_t n_heads;      // Number of heads
    uint32_t pos;          // Current position
} ds4_fp8_kv_cache;

ds4_fp8_kv_cache *ds4_kv_cache_create(uint32_t capacity, uint32_t head_dim, uint32_t n_heads);
void ds4_kv_cache_free(ds4_fp8_kv_cache *cache);
void ds4_kv_cache_reset(ds4_fp8_kv_cache *cache);
void ds4_kv_cache_append(ds4_fp8_kv_cache *cache, const float *kv, uint32_t head);
void ds4_kv_cache_quantize_fp8(const float *src, uint8_t *dst, uint32_t n, float *scale_out);
void ds4_kv_cache_get_fp8_ptrs(const ds4_fp8_kv_cache *cache, uint32_t head,
                                const uint8_t **k_ptr, const uint8_t **v_ptr);
void ds4_kv_cache_stats(const ds4_fp8_kv_cache *cache, FILE *fp);

#endif /* DS4_KV_CACHE_H */

/* =========================================================================
 * Model Config (ds4-cuda)
 * ========================================================================= */
#ifndef DS4_MODEL_CONFIG_H
#define DS4_MODEL_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int32_t n_embd;
    int32_t n_head;
    int32_t n_head_kv;
    int32_t n_layer;
    int32_t n_expert;
    int32_t n_expert_used;
    int32_t n_rot;
    int32_t n_ctx;
    int32_t vocab_size;
    float rope_freq_base;
    float rope_scaling_factor;
    float rms_norm_eps;
    float expert_weight_scale;
    const char *architectures;
    const char *model_type;
} ds4_model_config;

bool ds4_model_config_load(const char *model_dir, ds4_model_config *cfg);

#endif /* DS4_MODEL_CONFIG_H */

/* =========================================================================
 * SSD Streaming Expert Cache (ds4-cuda)
 * ========================================================================= */
#ifndef DS4_SSD_STREAMING_H
#define DS4_SSD_STREAMING_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint32_t layer;
    uint32_t expert_id;
    void *data;
    uint64_t size;
    uint64_t last_access;
    uint32_t priority;
} ssd_expert_entry;

typedef struct {
    ssd_expert_entry *entries;
    uint32_t capacity;
    uint32_t count;
    uint64_t total_bytes;
    uint64_t max_bytes;
    uint64_t hits;
    uint64_t misses;
} ssd_expert_cache;

ssd_expert_cache *ssd_expert_cache_create(uint32_t capacity, uint64_t max_bytes);
void ssd_expert_cache_free(ssd_expert_cache *cache);
int ssd_expert_cache_put(ssd_expert_cache *cache, uint32_t layer, uint32_t expert,
                          const void *data, uint64_t size);
void *ssd_expert_cache_get(ssd_expert_cache *cache, uint32_t layer, uint32_t expert);
void ssd_expert_cache_stats(ssd_expert_cache *cache, uint64_t *hits, uint64_t *misses, double *hit_rate);

#endif /* DS4_SSD_STREAMING_H */
```

---

### ~~Adım 2: ds4_gpu.h Değişiklikleri~~ ✅ Tamamlandı

**Nerede:** `ds4_gpu.h` — mevcut prototiplerin sonuna (`#endif`'den önce)

**Eklenecek kod:**
```c
/* =========================================================================
 * NVFP4/FP8 Operations (ds4-cuda)
 * ========================================================================= */

/* NVFP4 GEMV — weight: 4-bit packed + block scales */
int ds4_gpu_matmul_nvfp4_tensor(
    ds4_gpu_tensor       *out,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                weight_offset,
    uint64_t                in_dim,
    uint64_t                out_dim,
    const ds4_gpu_tensor *x,
    uint64_t                n_tok);

/* FP8 E4M3 GEMV — weight: 8-bit float */
int ds4_gpu_matmul_f8e4m3_tensor(
    ds4_gpu_tensor       *out,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                weight_offset,
    uint64_t                in_dim,
    uint64_t                out_dim,
    const ds4_gpu_tensor *x,
    uint64_t                n_tok);

/* FP8 attention — KV cache FP8 E4M3 */
int ds4_gpu_attention_fp8_heads_tensor(
    ds4_gpu_tensor       *heads,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                sinks_offset,
    const ds4_gpu_tensor *q,
    const uint8_t         *k_fp8_cache,
    const uint8_t         *v_fp8_cache,
    uint32_t                n_raw,
    uint32_t                raw_cap,
    uint32_t                raw_start,
    uint32_t                n_head,
    uint32_t                head_dim);

/* FP8 KV cache quantize + append */
int ds4_gpu_kv_fp8_quantize_append_tensor(
    ds4_gpu_tensor       *kv,
    ds4_gpu_tensor       *raw_cache,
    uint32_t                raw_cap,
    uint32_t                row,
    uint32_t                head_dim,
    uint32_t                n_rot);
```

---

### ~~Adım 3: ds4.c Değişiklikleri~~ ✅ Tamamlandı

#### 3.1 Safetensors Loader (satır ~1900 civarı, `ds4_gguf_load` yanına)

**Eklenecek fonksiyon:**
```c
/* Safetensors model yükleme */
static bool ds4_safetensors_load(ds4_model *m, const char *model_dir) {
    // 1. ds4_model_config_load(model_dir, &cfg) çağır
    // 2. sst_sharded_model_load(model_dir) çağır
    // 3. Her shard'ı mmap() ile aç
    // 4. Tensor name → offset mapping oluştur
    // 5. ds4_tensor array'ini doldur
    // 6. Tokenizer JSON yükle
    return true;
}
```

#### 3.2 FP8 KV Cache (satır ~8805, `kv_cache_push_raw` yanına)

**Eklenecek fonksiyon:**
```c
/* FP8 E4M3 KV cache'e ekleme */
static void kv_cache_push_fp8(ds4_layer_cache *cache, const float *kv) {
    uint8_t fp8_data[DS4_N_HEAD_DIM * 2];
    float scale;
    ds4_kv_cache_quantize_fp8(kv, fp8_data, DS4_N_HEAD_DIM * 2, &scale);
    // FP8 olarak sakla
}
```

**Değişiklik:** Satır 9524, 9758, 10007'deki `kv_cache_push_raw` çağrıları:
```c
// ESKİ:
kv_cache_push_raw(cache, kv);

// YENİ:
if (use_fp8_kv_cache) {
    kv_cache_push_fp8(cache, kv);
} else {
    kv_cache_push_raw(cache, kv);
}
```

#### 3.3 NVFP4/FP8 Dispatch (satır ~5236, `matvec_q8_0_worker` yanına)

**Eklenecek fonksiyonlar:**
```c
/* NVFP4 GEMV worker */
static void matvec_nvfp4_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_nvfp4_ctx *ctx = vctx;
    ds4_gpu_matmul_nvfp4_tensor(ctx->out, ctx->model_map, ctx->model_size,
                                 ctx->weight_offset, ctx->in_dim, ctx->out_dim,
                                 ctx->x, ctx->n_tok);
}

/* FP8 GEMV worker */
static void matvec_f8e4m3_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_f8e4m3_ctx *ctx = vctx;
    ds4_gpu_matmul_f8e4m3_tensor(ctx->out, ctx->model_map, ctx->model_size,
                                   ctx->weight_offset, ctx->in_dim, ctx->out_dim,
                                   ctx->x, ctx->n_tok);
}
```

**Değişiklik:** Weight dispatch mantığı (matmul çağrılarında):
```c
// ESKİ:
if (weight->type == DS4_TENSOR_Q8_0) {
    ds4_gpu_matmul_q8_0_tensor(...);
}

// YENİ:
if (weight->type == DS4_TENSOR_NVFP4) {
    ds4_gpu_matmul_nvfp4_tensor(...);
} else if (weight->type == DS4_TENSOR_F8_E4M3) {
    ds4_gpu_matmul_f8e4m3_tensor(...);
} else if (weight->type == DS4_TENSOR_Q8_0) {
    ds4_gpu_matmul_q8_0_tensor(...);  // fallback
}
```

#### 3.4 Kaynak Merge

Aşağıdaki dosyaların içeriği `ds4.c`'nin sonuna eklenir:
```c
/* =========================================================================
 * Safetensors Parser (merge: ds4_safetensors.c)
 * ========================================================================= */
// ds4_safetensors.c içeriği buraya

/* =========================================================================
 * FP8 KV Cache (merge: ds4_kv_cache.c)
 * ========================================================================= */
// ds4_kv_cache.c içeriği buraya

/* =========================================================================
 * Model Config (merge: ds4_model_config.c)
 * ========================================================================= */
// ds4_model_config.c içeriği buraya

/* =========================================================================
 * Expert Cache (merge: ds4_expert_cache.c)
 * ========================================================================= */
// ds4_expert_cache.c içeriği buraya
```

---

### ~~Adım 4: ds4_cuda.cu Değişiklikleri~~ ✅ Tamamlandı

#### 4.1 Kernel Wrapper'ları (dosya sonu)

**Eklenecek wrapper fonksiyonlar:**
```c
/* =========================================================================
 * NVFP4/FP8 Kernel Wrappers (ds4-cuda)
 * ========================================================================= */

#include "ds4_cuda_forward.cuh"
#include "ds4_cuda_fp8_attention.cuh"
#include "ds4_cuda_embedding.cuh"

/* NVFP4 GEMV wrapper */
int ds4_gpu_matmul_nvfp4_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok) {
    const void *w = (const char*)model_map + weight_offset;
    launch_gemv_nvfp4((const float*)x->data, (const uint8_t*)w, NULL,
                      (float*)out->data, out_dim, in_dim);
    return 0;
}

/* FP8 GEMV wrapper */
int ds4_gpu_matmul_f8e4m3_tensor(...) {
    launch_gemv_f8e4m3((const float*)x->data, (const uint8_t*)w,
                       (float*)out->data, out_dim, in_dim);
    return 0;
}

/* FP8 attention wrapper */
int ds4_gpu_attention_fp8_heads_tensor(...) {
    launch_fp8_attention((const float*)q->data, k_fp8_cache, v_fp8_cache,
                         (float*)heads->data, n_head, head_dim, n_raw,
                         1.0f / sqrtf(head_dim));
    return 0;
}
```

#### 4.2 Kaynak Merge

Aşağıdaki dosyaların içeriği `ds4_cuda.cu`'nun sonuna eklenir:
```c
/* =========================================================================
 * NVFP4 MMQ Kernel (merge: ds4_cuda_nvfp4_mmq.cu)
 * ========================================================================= */
// ds4_cuda_nvfp4_mmq.cu içeriği buraya

/* =========================================================================
 * Forward Kernels (merge: ds4_cuda_forward.cu)
 * ========================================================================= */
// ds4_cuda_forward.cu içeriği buraya

/* =========================================================================
 * FP8 Attention (merge: ds4_cuda_fp8_attention.cu)
 * ========================================================================= */
// ds4_cuda_fp8_attention.cu içeriği buraya

/* =========================================================================
 * Embedding (merge: ds4_cuda_embedding.cu)
 * ========================================================================= */
// ds4_cuda_embedding.cu içeriği buraya

/* =========================================================================
 * FP8 KV Cache GPU (merge: ds4_kv_cache.cu)
 * ========================================================================= */
// ds4_kv_cache.cu içeriği buraya
```

---

## 6. Build Sistemi

### Makefile Yapısı

```makefile
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -g -std=c99 -Wall -Wextra -I.
NVFLAGS = -O3 -g --use_fast_math -gencode arch=compute_121a,code=sm_121a

# Orijinal antirez/ds4 kaynakları
SRC = ds4.c ds4_cli.c ds4_server.c ds4_bench.c \
      ds4_eval.c ds4_agent.c ds4_web.c ds4_help.c \
      ds4_kvstore.c ds4_ssd.c ds4_distributed.c \
      rax.c linenoise.c

# CUDA kernel'ları (hepsi ds4_cuda.cu içinde merge edildi)
CUDA_SRC = ds4_cuda.cu
```

### Hedefler

| Hedef | Binary | Açıklama |
|-------|--------|----------|
| `make` | `ds4` | CLI inference |
| `make ds4-server` | `ds4-server` | HTTP server |
| `make ds4-bench` | `ds4-bench` | Benchmark |
| `make ds4-eval` | `ds4-eval` | Evaluation |
| `make ds4-agent` | `ds4-agent` | Agent |

### Build Talimatı

```bash
# Yerel build (nix-shell)
nix-shell --command 'make clean && make'

# Remote build (DGX Spark)
git commit -m "update" && git push
ssh xexnaor@10.0.0.2 './run.sh' > log.txt
```

---

## 7. SSD Streaming Mekanizması

### Orijinal akış (korunuyor):
```
1. mmap(model.gguf) → OS sayfalama ile diskten RAM'e
2. ds4_gguf_load() → Header parse, tensor offset'leri
3. ds4_gpu_cache_model_range() → mmap'dan GPU'ya kopyalama
4. ds4_expert_profile_cache_use() → Sıcak expert'leri takip
```

### Yeni akış (safetensors):
```
1. mmap(shard_0.safetensors) → OS sayfalama ile diskten RAM'e
2. ds4_safetensors_load() → JSON header parse, tensor offset'leri
3. ds4_gpu_cache_model_range() → mmap'dan GPU'ya kopyalama (AYNI)
4. ds4_expert_profile_cache_use() → Sıcak expert'leri takip (AYNI)
```

**Kritik:** Safetensors mmap ile GGUF mmap **aynı mekanizma**. SSD streaming otomatik çalışır.

---

## 8. Fonksiyon Haritası

### Hangi fonksiyon nerede tanımlanacak:

| Fonksiyon | Tanım Yeri | Kaynak |
|-----------|-----------|--------|
| `sst_sharded_model_load()` | `ds4.c` (merge) | `ds4_safetensors.c` |
| `sst_model_load()` | `ds4.c` (merge) | `ds4_safetensors.c` |
| `sst_model_find_tensor()` | `ds4.c` (merge) | `ds4_safetensors.c` |
| `ds4_kv_cache_create()` | `ds4.c` (merge) | `ds4_kv_cache.c` |
| `ds4_kv_cache_append()` | `ds4.c` (merge) | `ds4_kv_cache.c` |
| `ds4_kv_cache_quantize_fp8()` | `ds4.c` (merge) | `ds4_kv_cache.c` |
| `ds4_model_config_load()` | `ds4.c` (merge) | `ds4_model_config.c` |
| `ssd_expert_cache_create()` | `ds4.c` (merge) | `ds4_expert_cache.c` |
| `ssd_expert_cache_put()` | `ds4.c` (merge) | `ds4_expert_cache.c` |
| `launch_gemv_nvfp4()` | `ds4_cuda.cu` (merge) | `ds4_cuda_forward.cu` |
| `launch_gemv_f8e4m3()` | `ds4_cuda.cu` (merge) | `ds4_cuda_forward.cu` |
| `launch_fp8_attention()` | `ds4_cuda.cu` (merge) | `ds4_cuda_fp8_attention.cu` |
| `launch_token_embedding()` | `ds4_cuda.cu` (merge) | `ds4_cuda_embedding.cu` |
| `ds4_launch_mmq_nvfp4()` | `ds4_cuda.cu` (merge) | `ds4_cuda_nvfp4_mmq.cu` |
| `ds4_gpu_matmul_nvfp4_tensor()` | `ds4_cuda.cu` | Yeni wrapper |
| `ds4_gpu_matmul_f8e4m3_tensor()` | `ds4_cuda.cu` | Yeni wrapper |
| `ds4_gpu_attention_fp8_heads_tensor()` | `ds4_cuda.cu` | Yeni wrapper |
| `ds4_safetensors_load()` | `ds4.c` | Yeni fonksiyon |
| `kv_cache_push_fp8()` | `ds4.c` | Yeni fonksiyon |
| `matvec_nvfp4_worker()` | `ds4.c` | Yeni fonksiyon |
| `matvec_f8e4m3_worker()` | `ds4.c` | Yeni fonksiyon |

---

## 9. Kritik Tasarım Kararları

### 9.1 Safetensors vs GGUF
- **Seçim:** Safetensors (DeepSeek modelleri bu formatta)
- **Fallback:** GGUF desteği korunur
- **Mekanizma:** mmap tabanlı (aynı SSD streaming)

### 9.2 FP8 KV Cache
- **Format:** E4M3 (8-bit float)
- **Quantize:** CPU-side FP32→FP8
- **GPU:** FP8 attention kernel
- **Memory:** FP16'nın 2× daha az yer kaplar

### 9.3 NVFP4 Weight Dispatch
- **Format:** NVFP4 (4-bit, tensor core native)
- **Kernel:** MMA-based (sm_121a)
- **Fallback:** Q8_0 (antirez'in mevcut kodu)

### 9.4 FP8 Attention
- **Format:** E4M3 (KV cache FP8)
- **Dequant:** Yok (FP8 doğrudan tensor core'da)

---

## 10. Test Stratejisi

### Birim Testleri (yapılacak)
```
test_safetensors:    Safetensors parser (shard okuma, tensor bulma)
test_fp8_kv_cache:   FP8 quantize accuracy (maxdiff < 1e-3)
test_nvfp4_gemm:     NVFP4 GEMV (CPU vs GPU, maxdiff < 1e-4)
test_fp8_attention:  FP8 attention (CPU vs GPU, maxdiff < 1e-4)
```

### Entegrasyon Testleri (yapılacak)
```
test_model_load:     Model yükleme (safetensors + tokenizer)
test_single_token:   Tek token üretimi (greedy decoding)
test_generation:     Kısa metin üretimi (50 token)
test_benchmark:      Throughput testi (tokens/sec)
```

---

## 11. Dosya Kaynakları

### AS IS (antirez/ds4 — hiç değiştirilmeden)
```
ds4_cli.c, ds4_server.c, ds4_bench.c, ds4_eval.c, ds4_agent.c
ds4_help.c, ds4_kvstore.c, ds4_web.c, ds4_ssd.c
ds4_distributed.c, ds4_distributed.h
rax.c, linenoise.c
ds4_help.h, ds4_kvstore.h, ds4_ssd.h, ds4_web.h
linenoise.h, rax.h, rax_malloc.h
ds4_cuda_runtime.cuh, ds4_iq2_tables_cuda.inc, ds4_streaming_hotlist.inc
metal/
```

### DEĞİŞTİ (merge tamamlandı)
```
ds4.h              ← Safetensors/FP8/NVFP4 typedef'leri + 4 header merge ✅
ds4_gpu.h          ← NVFP4/FP8 fonksiyon prototipleri ✅
ds4.c              ← Safetensors loader + FP8 KV cache + NVFP4 dispatch + 4 kaynak merge ✅
ds4_cuda.cu        ← NVFP4/FP8 kernel wrapper'ları + 6 kaynak merge ✅
```

### YENİ KOD (merge edildi, kaynaklar silindi)
```
ds4_safetensors.c (434 satır)     → ds4.c ✅
ds4_safetensors.h (87 satır)      → ds4.h ✅
ds4_kv_cache.c (104 satır)        → ds4.c ✅
ds4_kv_cache.cu (124 satır)       → ds4_cuda.cu ✅
ds4_kv_cache.h (67 satır)         → ds4.h ✅
ds4_model_config.c (270 satır)    → ds4.c ✅
ds4_expert_cache.c (188 satır)    → ds4.c ✅
ds4_cuda_nvfp4_mmq.cu (106 satır) → ds4_cuda.cu ✅
ds4_cuda_forward.cu (287 satır)   → ds4_cuda.cu ✅
ds4_cuda_fp8_attention.cu (98 satır) → ds4_cuda.cu ✅
ds4_cuda_embedding.cu (92 satır)  → ds4_cuda.cu ✅
ds4_cuda_fp8_attention.cuh       → 🗑️ Silindi
ds4_cuda_embedding.cuh           → 🗑️ Silindi
ds4_layer_forward.cuh            → 🗑️ Silindi (ölü kod)
```
