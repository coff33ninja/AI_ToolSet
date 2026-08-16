"""Verify the GPU is detected by TensorFlow using the ai-toolset helpers."""

from ai_toolset import cuda

gpu = cuda.detect_gpu()
if gpu is None:
    print("No NVIDIA GPU detected. Run scripts/get_cuda_runtime.ps1 after installing drivers.")
    raise SystemExit(1)

entry = cuda.matrix_entry("2.10")
print("GPU:    {}".format(gpu["name"]))
print("Driver: {} (TF 2.10 needs >= {})".format(gpu["driver"], entry["driver_min"]))

cuda.configure_memory_growth()
print("TensorFlow GPU devices:", cuda.verify_tf_gpu())
