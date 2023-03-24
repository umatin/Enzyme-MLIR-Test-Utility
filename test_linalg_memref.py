from toolchain import jit_file
from stdout_parser import parse_results
import pytest


def execute(filename: str):
    return parse_results(jit_file(f"src/linalg_memref/{filename}"))


@pytest.mark.skip(reason="linalg interfaces not yet implemented")
def test_dot_product():
    assert execute("dot.mlir") == [[1.0, 4.0, 6.0, 8.0]]


@pytest.mark.skip(reason="linalg interfaces not yet implemented")
def test_matvec():
    assert execute("matvec.mlir") == [
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        [15.0, 18.0, 21.0, 24.0],
    ]
