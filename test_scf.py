from toolchain import jit_file
from stdout_parser import parse_results
import pytest
import numpy as np


def execute(filename: str):
    return parse_results(jit_file(filename))


def test_pow_scf():
    assert pytest.approx(execute("src/pow.mlir")) == [10 * 1.3 ** 9]


def test_pow_active_iter():
    assert execute("src/pow_iter_arg.mlir") == [1025]


def test_pow_multi_iter():
    assert pytest.approx(execute("src/for_multi_iter.mlir")) == [116.045]


@pytest.mark.skip(reason="Known issue with memrefs in iter args")
def test_memref_iter_arg():
    assert pytest.approx(np.array(execute("src/scf_memref_iterarg.mlir"))) == np.array(
        [[58.4688, 51.1566]]  # Verified against Enzyme and JAX
    )


@pytest.mark.skip(reason="Known issue with nested loops")
def test_pow_nested():
    assert pytest.approx(execute("src/pow_nested.mlir")) == [9 * 1.3 ** 8]
