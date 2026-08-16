"""ai-toolset: reusable helpers for GPU-accelerated machine learning projects.

Modules:
    cuda    - CUDA/GPU detection and TensorFlow GPU setup
    screen  - mss-based screen capture utilities
    video   - frame extraction and video stitching
    images  - image padding, splitting, and label conversion
    dataset - synthetic object-detection dataset generation
    audio   - voice-cloning / TTS dataset prep (resample, silence-split, RVC)

Run the CLI with:  uv run python -m ai_toolset --help
"""

__version__ = "0.1.0"
