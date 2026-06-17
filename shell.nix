let
  pkgs = import <nixpkgs> {
    config = {
      cudaSupport = true;
      allowUnfree = true;
    };
  };

in pkgs.mkShell rec {
  name = "ds4-cuda-dev-env";

  packages = with pkgs; [
    gcc
    gnumake
    coreutils
    which
    bash
    git
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
    cudaPackages.libcublas
  ];

  shellHook = ''
    echo "=== DS4 CUDA (DGX Spark) Development Environment ==="
    echo "CUDA target: sm_121a (GB10)"
    echo "To build: make ds4"
  '';
}
