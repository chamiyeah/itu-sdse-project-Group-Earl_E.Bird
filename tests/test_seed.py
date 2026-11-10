import numpy as np
import torch

from vial_classifier.seed import set_seed


def draw_numbers(seed):
    set_seed(seed)
    return np.random.rand(5), torch.rand(5)


def test_same_seed_gives_same_numbers():
    numpy_first, torch_first = draw_numbers(42)
    numpy_second, torch_second = draw_numbers(42)

    assert np.array_equal(numpy_first, numpy_second)
    assert torch.equal(torch_first, torch_second)


def test_different_seeds_give_different_numbers():
    numpy_first, torch_first = draw_numbers(42)
    numpy_second, torch_second = draw_numbers(43)

    assert not np.array_equal(numpy_first, numpy_second)
    assert not torch.equal(torch_first, torch_second)
