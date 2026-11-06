import numpy as np
import torch


def set_seed(seed):
    """Seed the global NumPy and PyTorch random generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
