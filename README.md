# ds4-cuda — NVFP4 MMQ Kernel for DGX Spark (Blackwell GB10 sm_121)

## Architecture Target

DGX Spark uses NVIDIA GB10 (Blackwell consumer, sm_121). Unlike datacenter Blackwell (sm_100), consumer Blackwell lacks tcgen05 / Tensor Memory (TMEM). However, it does support **native FP4 compute** via the `block_scale` MMA instructions when compiled with the correct architecture variant.

## Critical: sm_121 vs sm_121a

| Variant  | `__CUDA_ARCH` | PTX Version | a flag | f flag | NVFP4 MMA | Shared Memory |
|----------|---------------|-------------|--------|--------|-----------|---------------|
| sm_121   | 1210          | 6           | 0      | 0      | ❌ No     | 49 KB         |
| sm_121a  | 1210          | 7           | 1      | 0      | ✅ Yes    | 102 KB        |
| sm_121f  | 1210          | 7           | 1      | 1      | ❌ No     | 102 KB        |

**Only `sm_121a` enables native NVFP4/MXFP4 block_scale MMA instructions.**

## Compilation Flags

Use `-gencode arch=compute_121a,code=sm_121a` to generate PTX for the virtual architecture `compute_121a` and assemble for the real target `sm_121a`.

```makefile
NVFLAGS = -O3 --use_fast_math -gencode arch=compute_121a,code=sm_121a
```

## Family vs Specific Targets

| Target      | Covers                             |
|-------------|------------------------------------|
| sm_120f     | SM120 + SM121 (family)             |
| sm_120a     | SM120 only                         |
| sm_121a     | SM121 only                         |

Prebuilt wheels (PyTorch, vLLM, FlashInfer) typically ship `sm_120f` → no native NVFP4 MMA.  
JIT compilation on an SM121 machine produces `sm_121a` → unlocks native FP4 tensor cores.

## NVFP4 Format

The NVFP4 block format (DeepSeek-V4-Flash-NVFP4) uses:
- **E2M1** 4-bit values (2 exponent bits, 1 mantissa bit)
- **UE4M3** per-subgroup scales (4 exponent bits, 3 mantissa bits, unsigned)
- Block size: 64 elements
- Sub-block: 16 elements (4 scales per block)
- Packed size: 36 bytes per block (4 scale bytes + 32 value bytes)

## Kernel Design

The kernel uses the standard MMQ (matrix multiplication with quantized operands) approach:
1. Load NVFP4 blocks into shared memory tiles
2. Convert to packed 4-bit format for tensor core consumption
3. Execute `mma.sync.aligned.kind::mxf4nvf4.block_scale` instructions
4. Accumulate results and write back

## Building

```bash
# On DGX Spark
make cuda
```
