"""RVC (Retrieval-based Voice Conversion) via rvc-python.

Converts one speaker's voice into another using a trained RVC v2 model. The
model .pth (and optional .index) files are user-provided - download an RVC v2
model from your trainer of choice (e.g. the RVC webui or a shared model hub).

Requires the `rvc` extra:  uv sync --extra rvc
rvc-python pins numpy<=1.23.5, matching the TF-pinned numpy already in this
project. It also pulls fairseq/hydra for the HuBERT content encoder.
"""

from ai_toolset.cuda import set_visible_gpus


def convert_voice(input_path, output_path, model_path, index_path=None,
                  device=None, f0_method="rmvpe", index_rate=0.75,
                  protect=0.33, pitch=0, gpus=None):
    """Run RVC conversion. Returns output_path.

    device defaults to cuda:0 when a GPU is visible, else cpu. index_path
    enables feature-index retrieval (improves timbre for target speakers).
    pitch = semitone shift (e.g. 12 = one octave up, -12 = down).
    """
    set_visible_gpus(gpus)
    from rvc_python.infer import RVCInference

    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rvc = RVCInference(device=device)
    rvc.load_model(model_path, index_path=index_path)
    rvc.infer_file(input_path, output_path, f0_method=f0_method,
                   index_rate=index_rate, protect=protect, pitch=pitch)
    return output_path
