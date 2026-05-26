import os
import random
import torch
import numpy as np
from src.utils import set_seed, ensure_dir, num_threads


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = random.random()
        b = torch.rand(1).item()
        c = np.random.rand()
        set_seed(42)
        assert random.random() == a
        assert torch.rand(1).item() == b
        assert np.random.rand() == c


class TestEnsureDir:
    def test_creates_dir(self, tmp_path):
        d = tmp_path / "a" / "b"
        ensure_dir(str(d))
        assert d.exists()

    def test_idempotent(self, tmp_path):
        d = tmp_path / "exists"
        d.mkdir()
        ensure_dir(str(d))  # should not raise
        assert d.exists()


class TestNumThreads:
    def test_positive(self):
        assert num_threads() >= 1
