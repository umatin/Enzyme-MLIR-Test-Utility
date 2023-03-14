from toolchain import jit_file
from stdout_parser import parse_results
import pytest


def execute(filename: str):
    return parse_results(jit_file(filename))


@pytest.mark.skip(reason="scf.for with free vars currently not working")
def test_pow_scf():
    assert pytest.approx(execute("src/pow.mlir")) == [10 * 1.3 ** 9]


@pytest.mark.skip(reason="scf.for currently not working")
def test_pow_active_iter():
    assert execute("src/pow_iter_arg.mlir") == [1024]


@pytest.mark.skip(reason="scf.for with multiple results currently not working")
def test_pow_multi_iter():
    assert pytest.approx(execute("src/for_multi_iter.mlir")) == [116.045]
