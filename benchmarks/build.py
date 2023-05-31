import pathlib

from ronin.cli import cli
from ronin.projects import Project
from ronin.contexts import new_context
from ronin.utils.paths import glob
from toolchain import compile_mlir_enzyme, clang_dynamiclib

with new_context() as ctx:
    project = Project("Enzyme MLIR Benchmarks")
    tensor_sources = glob("*.mlir", pathlib.Path(__file__).parent / "tensor")

    obj = compile_mlir_enzyme(project, tensor_sources)
    clang_dynamiclib(project, [obj], "enzyme_mlir_benchmarks")

    cli(project)
