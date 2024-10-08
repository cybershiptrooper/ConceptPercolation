import torch


def get_default_device(use_mps_if_available=False):
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and use_mps_if_available:
        return "mps"
    return "cpu"
