let
  pkgs = import <nixpkgs> {
    config = {
      rocmSupport = true;
    };
  };

  rocmPkgs = pkgs.rocmPackages;

in pkgs.mkShell rec {
  name = "ds4-rocm-dev-env";

  packages = with pkgs; [
    # Core build tools
    gcc
    gnumake
    coreutils
    bison
    which
    bash

    # ROCm / HIP packages
    rocmPkgs.clr                    # HIP compiler, ROCm device libs, etc.
    rocmPkgs.hipblas                # HIP accelerated BLAS
    rocmPkgs.hipblas-common         # HIPBLAS common headers
    rocmPkgs.rocwmma                # ROCm Warp Matrix Multiply-Accumulate
    rocmPkgs.hipcub                 # HIP CUB wrappers
    rocmPkgs.rocprim                # ROCm C++ Parallel Primitives
    rocmPkgs.rocm-runtime           # ROCm Runtime (HSA)
    rocmPkgs.rocm-core              # ROCm Core (rocm-core)
    rocmPkgs.rocminfo               # rocminfo tool
    rocmPkgs.hipblaslt              # HIP BLAS LT (required for ROCM_LDLIBS)
    rocmPkgs.rocm-comgr             # ROCm Code Object Manager
    rocmPkgs.rocm-device-libs       # ROCm device libraries (bitcode)
  ];

  ROCM_PATH = "${rocmPkgs.clr}";
  ROCM_ARCH = "gfx1151";
  ROCM_VERSION = "${rocmPkgs.rocm-core.version}";
  HIP_PATH = "${rocmPkgs.clr}";
  HIP_PLATFORM = "amd";
  HSA_PATH = "${rocmPkgs.rocm-runtime}";
  DEVICE_LIB_PATH = "${rocmPkgs.rocm-device-libs}/amdgcn/bitcode";
  GPU_BACKEND = "rocm";

  ROCM_INCLUDE_PATH = "${rocmPkgs.clr}/include:"
    + "${rocmPkgs.hipblas}/include:"
    + "${rocmPkgs.hipblas-common}/include:"
    + "${rocmPkgs.rocwmma}/include:"
    + "${rocmPkgs.hipcub}/include:"
    + "${rocmPkgs.rocprim}/include";

  ROCM_CFLAGS = "-O3 -ffast-math -g -fno-finite-math-only -pthread"
    + " -D__HIP_PLATFORM_AMD__"
    + " -Wno-unused-command-line-argument"
    + " --offload-arch=${ROCM_ARCH}"
    + " -I${rocmPkgs.clr}/include"
    + " -I${rocmPkgs.rocwmma}/include"
    + " -I${rocmPkgs.hipcub}/include"
    + " -I${rocmPkgs.hipblas}/include"
    + " -I${rocmPkgs.hipblas-common}/include"
    + " -I${rocmPkgs.rocprim}/include"
    + " -I${rocmPkgs.hipblaslt}/include";

  ROCM_LDLIBS = "-lm -pthread"
    + " -L${rocmPkgs.hipblas}/lib"
    + " -L${rocmPkgs.hipblaslt}/lib"
    + " -lhipblas -lhipblaslt";

  shellHook = ''
    echo "=== DS4 ROCm (Strix Halo) Development Environment ==="
    echo "ROCm target: make strix-halo (ROCM_ARCH=${ROCM_ARCH})"
    echo "CPU target:  make cpu"
    echo ""
    echo "Environment variables:"
    echo "  ROCM_PATH=$ROCM_PATH"
    echo "  ROCM_ARCH=$ROCM_ARCH"
    echo "  HIP_PATH=$HIP_PATH"
    echo "  HIP_PLATFORM=$HIP_PLATFORM"
    echo "  HSA_PATH=$HSA_PATH"
    echo "  DEVICE_LIB_PATH=$DEVICE_LIB_PATH"
    echo "  GPU_BACKEND=$GPU_BACKEND"
    echo ""
    echo "ROCm include paths:"
    echo "  ROCM_INCLUDE_PATH=$ROCM_INCLUDE_PATH"
    echo ""
    echo "ROCM_CFLAGS:"
    echo "  $ROCM_CFLAGS"
    echo ""
    echo "ROCM_LDLIBS:"
    echo "  $ROCM_LDLIBS"
    echo ""
    echo "To build: make strix-halo"
    echo "To build: make cuda-spark"
  '';
}
