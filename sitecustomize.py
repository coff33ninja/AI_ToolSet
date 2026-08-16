import os


def _find_cuda_runtime(start):
    d = os.path.abspath(start)
    while True:
        candidate = os.path.join(d, "cuda_runtime", "bin")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _nvidia_lib_dirs(start):
    # The official NVIDIA wheels (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12)
    # drop their DLLs in site-packages/nvidia/<pkg>/bin (newer wheels) or
    # .../lib (older ones). faster-whisper's ctranslate2 loads
    # cublas64_12.dll / cudart64_12.dll through the normal Windows DLL search
    # path, so those dirs must be on PATH before it imports.
    d = os.path.abspath(start)
    while True:
        nvidia = os.path.join(d, "nvidia")
        if os.path.isdir(nvidia):
            for pkg in sorted(os.listdir(nvidia)):
                for sub in ("lib", "bin"):
                    lib = os.path.join(nvidia, pkg, sub)
                    if os.path.isdir(lib):
                        yield lib
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent


def _setup_path():
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = []
    dll_dir = os.environ.get("AI_TOOLSET_CUDA_RUNTIME")
    if not dll_dir:
        dll_dir = _find_cuda_runtime(base)
    if dll_dir and os.path.isdir(dll_dir):
        dirs.append(dll_dir)
    dirs.extend(_nvidia_lib_dirs(base))
    path = os.environ.get("PATH", "")
    for d in dirs:
        if d not in path:
            path = d + os.pathsep + path
    os.environ["PATH"] = path


_setup_path()
