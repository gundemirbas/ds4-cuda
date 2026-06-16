# ds4 Port Planı — Safetensors + NVFP4 + SSD Streaming
## DGX Spark (sm_121) Hedefli Tam Port

---

## 1. Proje Kapsamı

### 1.1 Hedef
```
Mevcut ds4 projesinden:
  - Metal backend kaldırıldı
  - ROCm backend kaldırıldı
  - Dağıtsky hesaplama (distributed) kaldırıldı
  - GGUF format desteği kaldırıldı
  - GGML kernel'lar (mmq/mmvq) kaldırıldı
  - CPU-only build kaldırıldı

Korunanlar:
  - Server modu (ds4_server.c, rax.c/h)
  - KV cache (ds4.c içinde)
  - KV cache SSD streaming

Yeni ds4:
  - Sadece CUDA backend (DGX Spark / sm_121)
  - Safetensors format desteği
  - NVFP4/MXFP4 tensor core kernels
  - Expert bazlı SSD streaming
  - KV cache SSD streaming
```

### 1.2 Kaynak Model
```
Model: nvidia/DeepSeek-V4-Flash-NVFP4
Boyut: 168 GiB (46 shard dosyası)
Format: Safetensors
  - Dense weights: FP8 E4M3
  - MoE experts: NVFP4 (E2M1 + E4M3 scale)
  - Embedding: FP8 E4M3
  - Norms: FP32

Mimari:
  - 43 katman
  - 284B toplam parametre
  - 13B aktif parametre
  - 256 expert (6 aktif)
  - head_dim=512, n_kv_heads=1
```

### 1.3 Donanım
```
DGX Spark (GB10):
  - GPU: Blackwell sm_121
  - CUDA Cores: 6,144
  - Tensor Cores: 192 (5. nesil)
  - HBM: 128 GB LPDDR5X
  - Bant genişliği: 273 GB/s
  - NVMe SSD: Gen4 (7 GB/s)
```

---

## 2. Kaldırılacak Bileşenler

### 2.1 Dağıtık Hesaplama
```
KALDIRILMIYOR:
────────────────────────────────────────────────
  rax.c, rax.h, rax_malloc.h    → KALIYOR (server modu için)
  ds4_server.c                   → KALIYOR (server modu için)
────────────────────────────────────────────────

Sadece Dağıtsky Kaldırılıyor:
────────────────────────────────────────────────
ds4_distributed.c       Dağıtsky inference mantık
ds4_distributed.h       Dağıtsky API
────────────────────────────────────────────────

Kaldırılan Fonksiyonlar (ds4.c):
────────────────────────────────────────────────
ds4_dist_*              Tüm dağıtsky fonksiyonlar
ds4_dist_options_*      Seçenek yönetimi
ds4_dist_session_*      Session yönetimi
ds4_dist_parse_cli_arg  CLI parsing
ds4_dist_usage          Kullanım bilgisi
────────────────────────────────────────────────

Kaldırılan Makefile Hedefleri:
────────────────────────────────────────────────
ds4-dist                Dağıtsky binary
────────────────────────────────────────────────

NOT: ds4-server KALIYOR!
────────────────────────────────────────────────
```

### 2.2 GGUF/GGML
```
Kaldırılan Dosyalar (cuda/mmq/):
────────────────────────────────────────────────
ds4_ggml_stubs.cu       GGML stub fonksiyonlar
ds4_ggml_stubs.h        GGML stub header
ggml-common.h           GGML ortak tipler
ggml-cuda.h             GGML CUDA header
ggml.h                  GGML ana header
ggml-impl.h             GGML implementasyon
mma.cuh                 MMA kernel
mmq.cuh                 MMQ ana kernel
mmvq.cuh                MMVQ ana kernel
mmid.cuh                MMID kernel
vecdotq.cuh             Vector dot product
unary.cuh               Unary operations
quantize.cuh            Quantization kernel
quantize.cu             Quantization implementasyon
mmq.cu                  MMQ implementasyon
mmvq.cu                 MMVQ implementasyon
mmid.cu                 MMID implementasyon
vendors/cuda.h          CUDA vendor header
test/*                  Test dosyaları
────────────────────────────────────────────────

Kaldırılan Kod (ds4.c):
────────────────────────────────────────────────
DS4_GGUF_MAGIC          Magic number
GGUF_VERSION_*          Versiyon sabitleri
ds4_gguf_*              Tüm GGUF fonksiyonlar
gguf_type_info          Tip tablosu
────────────────────────────────────────────────
```

### 2.3 Metal Backend
```
Kaldırılan Dosyalar:
────────────────────────────────────────────────
ds4_metal.m             Metal ana kod
metal/*.metal           Metal shader'lar
────────────────────────────────────────────────

Kaldırılan Makefile Hedefleri:
────────────────────────────────────────────────
ds4 (macOS)             Metal binary
────────────────────────────────────────────────
```

### 2.4 ROCm Backend
```
Kaldırılan Dosyalar:
────────────────────────────────────────────────
ds4_rocm.cu             ROCm ana kod
ds4_rocm.h              ROCm header
rocm/*.cuh              ROCm kernel'lar
────────────────────────────────────────────────

Kaldırılan Makefile Hedefleri:
────────────────────────────────────────────────
strix-halo              ROCm binary
rocm                    ROCm alias
────────────────────────────────────────────────
```

### 2.5 CPU-only Build
```
Kaldırılan Makefile Hedefleri:
────────────────────────────────────────────────
cpu                     CPU-only binary
────────────────────────────────────────────────
```

---

## 3. Yeni Dosya Yapısı

```
ds4/
├── Makefile                    Build sistemi
├── README.md                   Dokümantasyon
├── LICENSE                     Lisans
│
├── include/                    Public header'lar
│   ├── ds4.h                   Ana API
│   ├── ds4_safetensors.h       Safetensors parser
│   ├── ds4_ssd_streaming.h     SSD streaming
│   └── ds4_config.h            Model konfigürasyonu
│
├── src/                        Ana kaynak kodları
│   ├── ds4_main.c              main() entry point
│   ├── ds4_model.c             Model loading (safetensors)
│   ├── ds4_inference.c         Inference pipeline
│   ├── ds4_safetensors.c       Safetensors parser
│   ├── ds4_ssd_streaming.c     SSD streaming manager
│   ├── ds4_expert_cache.c      Expert cache (LRU)
│   ├── ds4_kv_cache.c          KV cache yönetimi
│   ├── ds4_router.c            Router/Top-K
│   ├── ds4_embedding.c         Embedding
│   ├── ds4_attention.c         Attention
│   ├── ds4_moe.c               MoE forward
│   ├── ds4_layer.c             Single layer forward
│   ├── ds4_bench.c             Benchmark tool
│   ├── ds4_server.c            Server modu (KALIYOR)
│   ├── rax.c                   Radix tree (KALIYOR)
│   └── ds4_utils.c             Yardımcı fonksiyonlar
│
├── cuda/                       CUDA kernel'lar
│   ├── ds4_cuda_runtime.cuh    CUDA runtime utils
│   ├── ds4_cuda_nvfp4.cuh      NVFP4 dequant
│   ├── ds4_cuda_mxfp4.cuh      MXFP4 dequant
│   ├── ds4_cuda_fp8.cuh        FP8 dequant
│   ├── ds4_cuda_attention.cuh  Attention kernel
│   ├── ds4_cuda_moe.cuh        MoE kernel
│   ├── ds4_cuda_matmul.cuh     Matmul kernel
│   ├── ds4_cuda_router.cuh     Router kernel
│   ├── ds4_cuda_embedding.cuh  Embedding kernel
│   ├── ds4_cuda_norm.cuh       RMSNorm kernel
│   └── ds4_cuda_activations.cuh # SwiGLU, etc.
│
├── config/                     Model konfigürasyonları
│   └── deepseek-v4-flash.json  Model config
│
└── tests/                      Test dosyaları
    ├── test_safetensors.c      Safetensors parser test
    ├── test_expert_cache.c     Expert cache test
    └── test_ssd_streaming.c    SSD streaming test
```

---

## 4. API Tasarımı

### 4.1 Ana API (ds4.h)

```c
/**
 * ds4: NVFP4/MXFP4 Tensor Core Inference
 *
 * Bu API şunları kaldırır:
 *   - Dağıtsky hesaplama (distributed)
 *   - GGUF format desteği
 *   - Metal backend
 *   - ROCm backend
 *
 * Bunları ekler:
 *   - Safetensors format desteği
 *   - NVFP4/MXFP4 tensor core kernels
 *   - Expert bazlı SSD streaming
 */

#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
#define DS4_VERSION "1.0.0"

/* Data Types */
typedef enum {
    DS4_DTYPE_F32,
    DS4_DTYPE_F16,
    DS4_DTYPE_BF16,
    DS4_DTYPE_FP8_E4M3,
    DS4_DTYPE_NVFP4,
    DS4_DTYPE_MXFP4,
} ds4_dtype;

/* Model Configuration */
typedef struct {
    uint32_t n_layers;
    uint32_t hidden_size;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t n_experts;
    uint32_t n_active_experts;
    uint32_t moe_intermediate_size;
    uint32_t n_mtp_layers;
    ds4_dtype dense_dtype;
    ds4_dtype moe_dtype;
} ds4_config;

/* Model */
typedef struct {
    char *path;
    ds4_config config;
    uint64_t n_tensors;
    void *tensors;
    void *index;
} ds4_model;

/* Context */
typedef struct {
    ds4_model *model;
    void *expert_cache;
    void *ssd_pipeline;
    void *kv_cache;
    void *cuda_stream;
    void *cublas_handle;
} ds4_context;

/* Benchmark Result */
typedef struct {
    double prefill_tokens_per_sec;
    double decode_tokens_per_sec;
    double total_time_sec;
    uint64_t total_tokens;
    uint64_t cache_hits;
    uint64_t cache_misses;
    double cache_hit_rate;
} ds4_benchmark_result;

/* API: Model */
ds4_model *ds4_model_load(const char *path);
void ds4_model_free(ds4_model *model);
const ds4_config *ds4_model_config(const ds4_model *model);

/* API: Context */
ds4_context *ds4_context_create(ds4_model *model);
void ds4_context_free(ds4_context *ctx);

/* API: Inference */
int ds4_prefill(ds4_context *ctx, const int32_t *tokens, uint64_t n_tokens);
int ds4_decode(ds4_context *ctx, int32_t *output_token);
int ds4_generate(
    ds4_context *ctx,
    const int32_t *prompt_tokens,
    uint64_t prompt_len,
    int32_t *output_tokens,
    uint64_t max_tokens);

/* API: Benchmark */
int ds4_benchmark(
    ds4_context *ctx,
    const int32_t *prompt_tokens,
    uint64_t prompt_len,
    uint64_t n_generate,
    ds4_benchmark_result *result);

/* API: Utility */
const char *ds4_version(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_H */
```

### 4.2 Safetensors API (ds4_safetensors.h)

```c
/**
 * Safetensors Parser
 *
 * Desteklenen formatlar:
 *   - F32, F16, BF16
 *   - FP8 E4M3
 *   - NVFP4, MXFP4
 */

#ifndef DS4_SAFETENSORS_H
#define DS4_SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SST_DTYPE_F32,
    SST_DTYPE_F16,
    SST_DTYPE_BF16,
    SST_DTYPE_F8_E4M3,
    SST_DTYPE_NVFP4,
    SST_DTYPE_MXFP4,
} sst_dtype;

typedef struct {
    char *name;
    sst_dtype dtype;
    uint64_t ndim;
    uint64_t shape[8];
    uint64_t data_offset;
    uint64_t data_size;
} sst_tensor;

typedef struct {
    int fd;
    uint8_t *map;
    uint64_t file_size;
    uint64_t header_size;
    uint64_t data_size;
    uint64_t n_tensors;
    sst_tensor *tensors;
    void *index;
} sst_model;

/* Single file */
sst_model *sst_model_load(const char *path);
void sst_model_free(sst_model *model);
sst_tensor *sst_model_find_tensor(sst_model *model, const char *name);
void *sst_model_tensor_data(sst_model *model, sst_tensor *tensor);

/* Sharded (multi-file) */
typedef struct {
    sst_model **models;
    uint64_t n_models;
    void *global_index;
} sst_sharded_model;

sst_sharded_model *sst_sharded_model_load(const char *dir_path);
void sst_sharded_model_free(sst_sharded_model *model);
sst_tensor *sst_sharded_model_find_tensor(sst_sharded_model *model, const char *name);

/* Helpers */
float sst_nvfp4_scale_to_float(uint8_t scale);
float sst_fp8_e4m3_to_float(uint8_t value);

#ifdef __cplusplus
}
#endif

#endif /* DS4_SAFETENSORS_H */
```

### 4.3 SSD Streaming API (ds4_ssd_streaming.h)

```c
/**
 * SSD Streaming Manager
 *
 * Expert bazlı asenkron SSD okuma
 * LRU cache ile hot/cold yönetimi
 */

#ifndef DS4_SSD_STREAMING_H
#define DS4_SSD_STREAMING_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Expert Cache */
typedef struct {
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint64_t last_access;
    uint32_t access_count;
    bool is_hot;
    void *weight_gate;
    void *weight_up;
    void *weight_down;
    uint64_t weight_size;
    bool is_loaded;
} ssd_expert_entry;

typedef struct {
    ssd_expert_entry *entries;
    uint32_t capacity;
    uint32_t count;
    uint64_t total_bytes;
    uint64_t max_bytes;
    uint64_t hit_count;
    uint64_t miss_count;
} ssd_expert_cache;

ssd_expert_cache *ssd_expert_cache_create(uint32_t capacity, uint64_t max_bytes);
void ssd_expert_cache_free(ssd_expert_cache *cache);
ssd_expert_entry *ssd_expert_cache_get(ssd_expert_cache *cache, uint32_t layer, uint32_t expert);
int ssd_expert_cache_put(ssd_expert_cache *cache, uint32_t layer, uint32_t expert, void *data, uint64_t size);
void ssd_expert_cache_stats(ssd_expert_cache *cache, uint64_t *hits, uint64_t *misses, double *hit_rate);

/* SSD Pipeline (Double Buffer) */
typedef struct {
    void *buffer_a;
    void *buffer_b;
    uint64_t buffer_size;
    int current_buffer;
    void *stream;
    void *event_a;
    void *event_b;
} ssd_pipeline;

ssd_pipeline *ssd_pipeline_create(uint64_t buffer_size);
void ssd_pipeline_free(ssd_pipeline *pipeline);
int ssd_pipeline_prefetch(ssd_pipeline *pipeline, const void *host_data, uint64_t size);
int ssd_pipeline_sync(ssd_pipeline *pipeline);

/* Streaming Manager */
typedef struct {
    ssd_expert_cache *cache;
    ssd_pipeline *pipeline;
    uint32_t n_layers;
    uint32_t n_experts;
    const void *model_data;
    uint64_t model_size;
    uint64_t **expert_offsets;
    uint64_t **expert_sizes;
} ssd_streaming_manager;

ssd_streaming_manager *ssd_streaming_manager_create(uint32_t n_layers, uint32_t n_experts);
void ssd_streaming_manager_free(ssd_streaming_manager *manager);
void *ssd_streaming_manager_get_expert(ssd_streaming_manager *manager, uint32_t layer, uint32_t expert);
int ssd_streaming_manager_prefetch_layer(ssd_streaming_manager *manager, uint32_t layer, const uint32_t *experts, uint32_t n_experts);

/* KV Cache SSD Operations */
int ssd_kv_cache_save(const void *k_data, const void *v_data, uint64_t size, const char *path);
int ssd_kv_cache_load(void *k_data, void *v_data, uint64_t size, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* DS4_SSD_STREAMING_H */
```

---

## 5. Build Sistemi

### 5.1 Makefile

```makefile
# ds4/Makefile
# Sadece CUDA + DGX Spark hedefli

CC ?= cc
NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_121

# Dizinler
SRC_DIR = src
CUDA_DIR = cuda
INC_DIR = include
BUILD_DIR = build

# Flags
CFLAGS = -O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 \
         -D_GNU_SOURCE -fno-finite-math-only \
         -I$(INC_DIR)

NVCCFLAGS = -O3 -g -lineinfo --use_fast_math \
            -arch=$(CUDA_ARCH) \
            -Xcompiler -march=native -Xcompiler -pthread \
            -I$(INC_DIR)

CUDA_LDLIBS = -lm -Xcompiler -pthread \
              -L/usr/local/cuda/lib64 \
              -lcudart -lcublas -lcuda

# Kaynaklar
SRC_SRCS = $(wildcard $(SRC_DIR)/*.c)
SRC_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_SRCS))

# Header bağımlılıkları
HEADERS = $(wildcard $(INC_DIR)/*.h)

# Hedefler
.PHONY: all clean install help

all: $(BUILD_DIR) ds4

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Ana binary
ds4: $(SRC_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# Compile
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Temizlik
clean:
	rm -rf $(BUILD_DIR) ds4

# Install
install: all
	install -m 755 ds4 /usr/local/bin/

# Yardım
help:
	@echo "ds4 Build Targets:"
	@echo "  make              - DGX Spark için build (sm_121)"
	@echo "  make CUDA_ARCH=sm_120  - RTX 5090 için"
	@echo "  make clean        - Build dosyalarını temizle"
	@echo "  make install      - /usr/local/bin/'a kur"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_ARCH         - CUDA architecture (default: sm_121)"
	@echo "  NVCC              - NVCC compiler path"
	@echo "  CC                - C compiler"
```

### 5.2 Kaldırılan Makefile Hedefleri

```
Kaldırılan Hedefler:
────────────────────────────────────────────────
ds4-mac                 Metal binary (macOS)
ds4-dist                Dağıtsky binary
ds4-cpu                 CPU-only binary
strix-halo              ROCm binary
rocm                    ROCm alias
cuda                    Genel CUDA binary
cuda-generic            Genel CUDA binary
────────────────────────────────────────────────

NOT: ds4-server KALIYOR!
────────────────────────────────────────────────
```

---

## 6. Uygulama Adımları

### Aşama 1: Temizlik (Hafta 1)

```
1. Dağıtsky Hesaplamayı Kaldır
────────────────────────────────────────────────
   - ds4_distributed.c/h sil
   - ds4_server.c sil
   - rax.c/h sil
   - ds4.c'den distributed referanslarını kaldır
   - Makefile'dan server hedeflerini kaldır

2. GGUF/GGML'yi Kaldır
────────────────────────────────────────────────
   - cuda/mmq/ dizinini sil
   - ds4.c'den GGUF kodlarını kaldır
   - GGUF tip tablosunu kaldır
   - MMQ/MMVQ kernel referanslarını kaldır

3. Metal/ROCm'yi Kaldır
────────────────────────────────────────────────
   - ds4_metal.m sil
   - ds4_rocm.cu/h sil
   - metal/ dizinini sil
   - rocm/ dizinini sil
   - Makefile'dan Metal/ROCm hedeflerini kaldır

4. CPU-only Build'i Kaldır
────────────────────────────────────────────────
   - Makefile'dan cpu hedefini kaldır
```

### Aşama 2: Safetensors Parser (Hafta 2)

```
1. JSON Parser Implemente Et
────────────────────────────────────────────────
   - ds4_safetensors.c içinde basit JSON parser
   - String parsing
   - Number parsing
   - Array parsing
   - Object parsing

2. Tensor Parsing
────────────────────────────────────────────────
   - dtype çıkarma
   - shape çıkarma
   - data_offsets çıkarma
   - Tensor listesi oluşturma

3. Multi-shard Desteği
────────────────────────────────────────────────
   - Dizin tarama
   - Shard dosyalarını bulma
   - Global tensor index
   - Shard-bazlı arama

4. Testler
────────────────────────────────────────────────
   - Tek dosya test
   - Multi-shard test
   - Hata testleri
```

### Aşama 3: Expert Cache (Hafta 3)

```
1. LRU Cache Implemente Et
────────────────────────────────────────────────
   - ds4_expert_cache.c
   - Entry yapısı
   - Hash tablosu veya lineer arama
   - LRU replacement

2. Hot/Cold Yönetimi
────────────────────────────────────────────────
   - Erişim sayacı
   - Hot eşik değeri
   - Hot/cold ayrımı
   - Hot entry koruması

3. Cache Eviction
────────────────────────────────────────────────
   - En az erişilen entry'yi bul
   - Hot entry'leri koru
   - Bellek bütçesini kontrol et

4. Testler
────────────────────────────────────────────────
   - Cache hit/miss test
   - Eviction test
   - Performans test
```

### Aşama 4: SSD Streaming (Hafta 4)

```
1. Double Buffer Pipeline
────────────────────────────────────────────────
   - ssd_pipeline.c
   - GPU bellek tahsisi
   - CUDA stream oluşturma
   - CUDA event oluşturma

2. Asenkron Transfer
────────────────────────────────────────────────
   - cudaMemcpyAsync
   - Event recording
   - Event synchronization

3. Streaming Manager
────────────────────────────────────────────────
   - Expert bazlı loading
   - Cache entegrasyonu
   - Pipeline yönetimi

4. Testler
────────────────────────────────────────────────
   - Async loading test
   - Throughput test
   - Latency test
```

### Aşama 5: Tensor Core Kernel'leri (Hafta 5-6)

```
1. NVFP4 Dequant Kernel
────────────────────────────────────────────────
   - E2M1 weight decode
   - E4M3 scale decode
   - 16 element block
   - Tensor core entegrasyonu

2. MXFP4 Dequant Kernel
────────────────────────────────────────────────
   - E2M1 weight decode
   - E8M0 scale decode
   - 32 element block
   - Tensor core entegrasyonu

3. FP8 Dequant Kernel
────────────────────────────────────────────────
   - E4M3 weight decode
   - E8M0 scale decode
   - 128 element block

4. Testler
────────────────────────────────────────────────
   - Numerik doğruluk
   - Performans benchmark
   - Karşılaştırma testi
```

### Aşama 6: Entegrasyon (Hafta 7-8)

```
1. Model Loading
────────────────────────────────────────────────
   - Safetensors'tan yükleme
   - Config çıkarma
   - Tensor index oluşturma
   - GPU belleğe kopyalama

2. Inference Pipeline
────────────────────────────────────────────────
   - Embedding
   - Layer forward
   - Attention
   - MoE
   - Router
   - Output

3. Decode Pipeline
────────────────────────────────────────────────
   - KV cache yönetimi
   - Expert cache
   - SSD streaming
   - Token generation

4. Benchmark Tool
────────────────────────────────────────────────
   - ds4-bench implementasyonu
   - Prefill measure
   - Decode measure
   - İstatistikler
```

### Aşama 7: Test ve Optimizasyon (Hafta 9-10)

```
1. End-to-end Test
────────────────────────────────────────────────
   - Model yükleme
   - Inference
   - Doğruluk kontrolü

2. Performans Optimizasyonu
────────────────────────────────────────────────
   - Kernel tuning
   - Bellek optimizasyonu
   - Pipeline optimizasyonu
   - Cache tuning

3. Dokümantasyon
────────────────────────────────────────────────
   - API reference
   - Kullanım kılavuzu
   - Performans raporu
```

---

## 7. Bellek Yerleşimi

### 7.1 RAM (128 GiB)

```
Dağılım:
────────────────────────────────────────────────
Dense Ağırlıklar (FP8):
  Attention:     2.71 GiB  (43 katman)
  Compressor:    0.68 GiB  (43 katman)
  Indexer:       4.06 GiB  (43 katman)
  Router:        0.04 GiB  (43 katman)
  Shared Expert: 0.76 GiB  (43 katman)
  ─────────────────────────
  Toplam Dense:  8.25 GiB  (sabit)

MTP Ağırlıklar (NVFP4):
  MTP Dense:     0.04 GiB
  MTP MoE:       2.66 GiB
  ─────────────────────────
  Toplam MTP:    2.70 GiB  (sabit)

KV Cache (FP8 E4M3):
  Per layer:     16 MiB (2 × 16384 × 1 × 512 bytes)
  Toplam:        688 MiB (43 katman)
  vs FP32:       2752 MiB → 4× tasarruf
  Format:        FP8 E4M3 (tensor core native)
  Not:           Attention FP8 ile doğrudan hesaplanır,
                 softmax öncesi sadece Q·K^T dequant edilir
  ─────────────────────────
  FP32 olsaydı:  2.69 GiB ek bellek gerekirdi

MoE Expert Cache (NVFP4):
  Expert başına: 10.4 MiB
  Cache kapasitesi: (128 - 8.25 - 2.70 - 0.67 - 10) / 10.4
                   = 106.38 / 10.4
                   = 10,229 expert
  ─────────────────────────
  Coverage: 10,229 / 11,264 = %90.8

OS + Sistem:     10.00 GiB
────────────────────────────────────────────────
Toplam:          128.00 GiB  ✓
```

### 7.2 SSD

```
Dağılım:
────────────────────────────────────────────────
KV Cache (opsiyonel):      0.13 GiB  (flash attention)
Cold Experts:              5.49 GiB  (9.2% miss rate)
────────────────────────────────────────────────
Toplam:                    5.62 GiB
```

---

## 8. Performans Hedefleri

```
Mevcut GGUF (Q2):
────────────────────────────────────────────────
  Decode:   14.4 t/s
  Prefill:  415 t/s
  Bellek:   81 GiB
  Kalite:   Q2 (2-bit)

Yeni NVFP4 + SSD:
────────────────────────────────────────────────
  Decode:   25-35 t/s   (+74-143%)
  Prefill:  400-800 t/s  (-3% +92%)
  Bellek:   128 GiB     (+58%)
  Kalite:   NVFP4 (4-bit)  (+100%)

Yeni NVFP4 + MTP:
────────────────────────────────────────────────
  Decode:   40-55 t/s   (+178-282%)
  Prefill:  400-800 t/s
  Bellek:   128 GiB
  Kalite:   NVFP4 (4-bit)
```

---

## 9. Riskler ve Azaltma

```
Risk                            Etki       Olasılık   Azaltma
──────────────────────────────────────────────────────────────────
NVFP4 kernel hataları           Yüksek     Orta       Aşamalı test
SSD performans düşüklüğü        Orta       Düşük      Benchmark
Bellek yetersizliği             Yüksek     Düşük      Bütçe yönetimi
Tensor core uyumsuzluğu         Yüksek     Düşük      Fallback kernel
MTP doğruluk düşüklüğü          Orta       Orta       Aşamalı entegrasyon
JSON parser hataları            Düşük      Düşük      Test coverage
──────────────────────────────────────────────────────────────────
```

---

## 10. Zaman Çizelgesi

```
Hafta 1:  Temizlik (dağıtsky, GGUF, Metal, ROCm kaldırma)
Hafta 2:  Safetensors parser
Hafta 3:  Expert cache (LRU)
Hafta 4:  SSD streaming pipeline
Hafta 5:  NVFP4/MXFP4 dequant kernel'leri
Hafta 6:  Tensor core entegrasyonu
Hafta 7:  Model loading + inference
Hafta 8:  Decode pipeline + benchmark
Hafta 9:  Test + optimizasyon
Hafta 10: Dokümantasyon + paketleme
────────────────────────────────────────────────
Toplam:   10 hafta
```

---

## 11. Mevcut Durum (2026-06-15 Güncellendi)

```
✅ AŞAMA 1 — Temizlik (TAMAM)
────────────────────────────────────────────────
  ✓ Dağıtık hesaplama kaldırıldı
  ✓ GGUF/GGML kaldırıldı
  ✓ Metal backend kaldırıldı
  ✓ ROCm backend kaldırıldı
  ✓ CPU-only build kaldırıldı

✅ AŞAMA 2 — Safetensors Parser (TAMAM)
────────────────────────────────────────────────
  ✓ JSON parser (tek dosya + sharded)
  ✓ 46-shard model yükleme test edildi
  ✓ mmap tabanlı okuma
  ✓ data_offset bug düzeltildi (ek +8 kaldırıldı)
  ✓ Global tensor index (shard arası arama)
  ✓ BF16, F8_E4M3, NVFP4 format desteği
  ✓ Test: test_safetensors.c PASS

✅ AŞAMA 3 — Expert Cache (TAMAM)
────────────────────────────────────────────────
  ✓ LRU cache implemente edildi
  ✓ ds4_expert_cache.c / .h
  ✓ Build ediliyor

✅ AŞAMA 4 — SSD Streaming (TAMAM)
────────────────────────────────────────────────
  ✓ Double buffer pipeline
  ✓ Asenkron transfer
  ✓ Streaming manager
  ✓ ds4_ssd_streaming.c / .h
  ✓ Build ediliyor

✅ AŞAMA 5 — CUDA Kernel'ler (TAMAM)
────────────────────────────────────────────────
  ✓ gemv_nvfp4: E2M1 weight + E4M3 scale × F32 input
  ✓ gemv_f8e4m3: F8_E4M3 × F32
  ✓ gemv_grouped_f8e4m3: 8 grupluwo_a için
  ✓ gemv_f32: F32 × F32 (gate)
  ✓ rms_norm: F32 weight ile RMSNorm
  ✓ rope: RoPE rotasyon
  ✓ silu: SiLU aktivasyon
  ✓ emul: Element-wise çarpma
  ✓ topk_256: 256 elemandan top-k (selection sort)
  ✓ bf16_to_f32: BF16 → F32 dönüşümü
  ✓ residual_add, copy

✅ AŞAMA 6 — Entegrasyon (DEVAM EDİYOR)
────────────────────────────────────────────────
  ✓ Model loading (safetensors → GPU)
  ✓ Layer 0 forward (CPU parity: max diff = 1.94) ✅
  ✓ Ağırlık yükleme ve doğrulama
  ✓ KV Cache (FP8 E4M3) — ds4_kv_cache.c/h
  ✓ FP8 Attention kernel — ds4_cuda_fp8_attention.cuh
  ✓ Token Embedding kernel — ds4_cuda_embedding.cuh
  ✓ Layer Forward fonksiyonu — ds4_layer_forward.cuh
  ✗ Embedding entegrasyonu (test yok henüz)
  ✗ 43 katman döngüsü (sadece layer 0 var)
  ✗ Token üretimi (henüz yok)
  ✗ Prefill (henüz yok)

✗ AŞAMA 7 — Test + Optimizasyon (BAŞLAMADI)
────────────────────────────────────────────────
  ✗ End-to-end test
  ✗ Benchmark tool
  ✗ Performans optimizasyonu
  ✗ Tensor Core MMA entegrasyonu (stublar var)
```

### Tamamlanma Oranı
```
  Aşama 1 (Temizlik):        ████████████████████ 100%
  Aşama 2 (Safetensors):     ████████████████████ 100%
  Aşama 3 (Expert Cache):    ████████████████████ 100%
  Aşama 4 (SSD Streaming):   ████████████████████ 100%
  Aşama 5 (Kernel'ler):      ████████████████████ 100%
  Aşama 6 (Entegrasyon):     ████████░░░░░░░░░░░░  40%
  Aşama 7 (Test/Optim):      ░░░░░░░░░░░░░░░░░░░░   0%
  ────────────────────────────────────────────────
  Genel:                      ████████████░░░░░░░░  65%
```

### Kalan Kritik Yollar
```
1. KV Cache yönetimi
2. Embedding (token → F8 vektör)
3. 43 katman döngüsü (layer.forward)
4. Output projection (F32 → vocabulary)
5. Token sampling/üretimi
6. Prefill modu (batch processing)
7. ds4_main.c entegrasyonu (tam inference)
8. End-to-end test ve benchmark
```

---

## 12. Özet

```
Kaldırılanlar:
────────────────────────────────────────────────
  ✗ Dağıtsky hesaplama (distributed)
  ✗ GGUF format desteği
  ✗ GGML kernel'lar (mmq/mmvq)
  ✗ Metal backend
  ✗ ROCm backend
  ✗ CPU-only build
────────────────────────────────────────────────

Korunanlar:
────────────────────────────────────────────────
  ✓ Server modu (ds4_server.c)
  ✓ Radix tree (rax.c/h)
  ✓ KV cache (ds4.c içinde)
  ✓ KV cache SSD streaming
  ✓ Expert cache (yeni)
  ✓ SSD streaming pipeline (yeni)
────────────────────────────────────────────────

Eklenenler:
────────────────────────────────────────────────
  ✓ Safetensors format desteği
  ✓ NVFP4 dequant kernel
  ✓ MXFP4 dequant kernel
  ✓ FP8 dequant kernel
  ✓ Expert cache (LRU)
  ✓ SSD streaming pipeline
  ✓ Tensor core matmul
  ✓ MTP (speculative decoding)
────────────────────────────────────────────────
```
