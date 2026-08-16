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


def _setup_path():
    dll_dir = os.environ.get("AI_TOOLSET_CUDA_RUNTIME")
    if not dll_dir:
        dll_dir = _find_cuda_runtime(os.path.dirname(os.path.abspath(__file__)))
    if dll_dir and os.path.isdir(dll_dir) and dll_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


_setup_path()
