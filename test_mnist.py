import pytest
import numpy as np
import torch
from benchmarks.reference.pytorch_mnist import torch_mlp, torch_primal
from benchmarks.mlir_bindings import mnist_mlp_primal


def test_mnist_primal():
    parameters = [param.detach().numpy() for param in torch_mlp.parameters()]
    X = np.random.randn(784, 64).astype(np.float32)
    y = np.random.randint(low=0, high=10, size=(64,)).astype(np.int32)
    ref_primal = torch_primal(torch.from_numpy(X).mT, torch.from_numpy(y).long())
    mlir_params = [X, y] + parameters
    act_primal = mnist_mlp_primal(*mlir_params)
    assert act_primal == pytest.approx(ref_primal.item())
