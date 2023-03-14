from toolchain import jit_file
from stdout_parser import parse_results
import pytest


def execute(filename: str):
    return parse_results(jit_file(filename))


def test_multi_operands():
    assert execute("src/arith1.mlir") == [2]


def test_custom_seed():
    assert execute("src/arith2.mlir") == [23.01]


def test_3():
    # TODO: Not sure what this is testing
    assert execute("src/arith3.mlir") == [1]


def test_cf_loop():
    assert execute("src/arith4.mlir") == [10]


def test_pow_cf():
    assert pytest.approx(execute("src/pow_cf.mlir")) == [10 * 1.3 ** 9]


def test_constant_arg():
    assert execute("src/const_arg.mlir") == [17]


def test_multi_use():
    assert execute("src/arith5.mlir") == [2]


def test_memref():
    assert execute("src/memref1.mlir") == [1, [1]]


def test_memref_partially_used():
    assert execute("src/memref2.mlir") == [0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
