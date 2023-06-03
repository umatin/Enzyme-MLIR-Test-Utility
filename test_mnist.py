import pytest
import numpy as np
import torch
from benchmarks.reference.pytorch_mnist import torch_mlp, torch_primal
from benchmarks.mlir_bindings import mnist_mlp_primal, enzyme_mnist_mlp


def test_mnist_primal():
    parameters = [param.detach().numpy() for param in torch_mlp.parameters()]
    X = np.random.randn(784, 64).astype(np.float32)
    y = np.random.randint(low=0, high=10, size=(64,)).astype(np.int32)
    ref_primal = torch_primal(torch.from_numpy(X).mT, torch.from_numpy(y).long())
    mlir_params = [X, y] + parameters
    act_primal = mnist_mlp_primal(*mlir_params)
    assert act_primal == pytest.approx(ref_primal.item())


@pytest.mark.xfail
def test_mnist_adjoint():
    parameters = [param.detach().numpy() for param in torch_mlp.parameters()]
    X = np.random.randn(784, 64).astype(np.float32)
    y = np.random.randint(low=0, high=10, size=(64,)).astype(np.int32)
    ref_primal = torch_primal(torch.from_numpy(X).mT, torch.from_numpy(y).long())
    ref_primal.backward()

    mlir_params = [X, y] + parameters
    dweight0, dbias0, dweight1, dbias1, dweight2, dbias2 = enzyme_mnist_mlp(
        *mlir_params
    )
    ref_grads = [p.grad for p in torch_mlp.parameters()]

    # pytest.approx tests appear to be too slow for these bigger comparisons.
    tol = 3e-8
    assert np.abs(dweight0 - ref_grads[0].numpy()).max() < tol
    assert np.abs(dbias0 - ref_grads[1].numpy()).max() < tol
    assert np.abs(dweight1 - ref_grads[2].numpy()).max() < tol
    assert np.abs(dbias1 - ref_grads[3].numpy()).max() < tol
    assert np.abs(dweight2 - ref_grads[4].numpy()).max() < tol
    assert dbias2 == pytest.approx(ref_grads[-1], rel=5e-5)
