import platform
import pathlib
import subprocess

from ronin.gcc import GccExecutor, GccLink
from ronin.executors import ExecutorWithArguments
from ronin.phases import Phase
from ronin.projects import Project
from ronin.utils.platform import which

##
# Configuration
##

with open(pathlib.Path(__file__).parent / "paths.txt", "r") as f:
    mlir_path, enzyme_path = [pathlib.Path(line.strip()) for line in f.readlines()]

LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"
MLIR_BIN = mlir_path / "bin"
MLIR_LIB = mlir_path / "lib"
ENZYMEMLIR_OPT = enzyme_path / "enzyme/build/Enzyme/MLIR/enzymemlir-opt"
ENZYME_DYLIB = enzyme_path / f"enzyme/build/Enzyme/LLVMEnzyme-16.{LIB_EXT}"
OPT = MLIR_BIN / "opt"
LLI = MLIR_BIN / "lli"
CLANG = MLIR_BIN / "clang"
MLIR_OPT = MLIR_BIN / "mlir-opt"
MLIR_TRANSLATE = MLIR_BIN / "mlir-translate"
MLIR_CPU_RUNNER = MLIR_BIN / "mlir-cpu-runner"
RUNNER_UTILS = MLIR_LIB / f"libmlir_runner_utils.{LIB_EXT}"
C_RUNNER_UTILS = MLIR_LIB / f"libmlir_c_runner_utils.{LIB_EXT}"

##
# Lowering flags
##


class MLIRLoweringFlags:
    enzyme_to_core = [
        "-enzyme",
        "-symbol-dce",
        "-convert-enzyme-shadowed-gradient-to-cache",
        "-convert-enzyme-to-memref",
        "-canonicalize",
        "-reconcile-unrealized-casts",
    ]

    core_to_llvm_dialect = [
        "-one-shot-bufferize=bufferize-function-boundaries",
        "-convert-linalg-to-loops",
        "-convert-scf-to-cf",
        "-convert-cf-to-llvm",
        "-convert-func-to-llvm",
        "-convert-memref-to-llvm",
        "-convert-arith-to-llvm",
        "-convert-math-to-llvm",
        "-reconcile-unrealized-casts",
    ]


##
# JIT Execution (for unit tests)
##


def run_safe(args, stdin: bytes = None, suppress_stderr=False) -> bytes:
    try:
        p = subprocess.run(args, input=stdin, check=True, capture_output=True)
        if p.stderr and not suppress_stderr:
            print(p.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode("utf-8"))
        raise Exception(e.stderr.decode("utf-8"))
    return p.stdout


def differentiate(source: bytes) -> bytes:
    return run_safe([ENZYMEMLIR_OPT, "-enzyme", "-symbol-dce"], stdin=source)


def lower_to_llvm_dialect(source: bytes) -> bytes:
    memref_dialect = run_safe(
        [ENZYMEMLIR_OPT] + MLIRLoweringFlags.enzyme_to_core, stdin=source
    )
    return run_safe(
        [MLIR_OPT] + MLIRLoweringFlags.core_to_llvm_dialect, stdin=memref_dialect
    )


def jit_file(filename: str):
    with open(filename, "rb") as f:
        return jit_mlir(f.read())


def jit_mlir(source: bytes):
    jit_args = [
        "-entry-point-result=void",
        f"-shared-libs={RUNNER_UTILS},{C_RUNNER_UTILS}",
    ]

    lowered = lower_to_llvm_dialect(differentiate(source))
    stdout = run_safe([MLIR_CPU_RUNNER] + jit_args, stdin=lowered)
    return stdout.decode("utf-8")


#
# Ronin phases (for larger integration tests and benchmarks)
#


class EnzymeMLIROptExecutor(ExecutorWithArguments):
    def __init__(self, default_args=True):
        super(EnzymeMLIROptExecutor, self).__init__()
        self.command = ENZYMEMLIR_OPT
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = "mlir"
        if default_args:
            self.add_argument(*MLIRLoweringFlags.enzyme_to_core)


class MLIROptExecutor(ExecutorWithArguments):
    def __init__(self, default_args=True):
        super(MLIROptExecutor, self).__init__()
        self.command = MLIR_OPT
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = ".core.mlir"
        if default_args:
            self.add_argument(*MLIRLoweringFlags.core_to_llvm_dialect)


class MLIRTranslateExecutor(ExecutorWithArguments):
    def __init__(self):
        super(MLIRTranslateExecutor, self).__init__()
        self.command = MLIR_TRANSLATE
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = "ll"
        self.add_argument("-mlir-to-llvmir")


class ClangCompileLLVM(GccExecutor):
    def __init__(self, command: str = None, ccache=True, platform=None):
        super(ClangCompileLLVM, self).__init__(command, ccache, platform)
        self.command = lambda ctx: which(
            ctx.fallback(command, "gcc.gcc_command", CLANG)
        )
        self.add_argument_unfiltered("$in")

        self.command_types = ["clang_compile"]
        self.output_type = "object"
        self.output_extension = "o"
        self.compile_only()

    def ignore_override_module(self):
        self.add_argument("-Wno-override-module")
        return self


def compile_mlir_enzyme(project: Project, inputs: list[str]) -> Phase:
    emlir_opt = EnzymeMLIROptExecutor()

    core_mlir = Phase(
        project=project,
        name="Enzyme MLIR Lower to Core MLIR Dialects",
        executor=emlir_opt,
        inputs=inputs,
    )

    llvm_dialect = Phase(
        project=project,
        name="Core MLIR Dialects to LLVM Dialect",
        executor=MLIROptExecutor(),
        inputs_from=[core_mlir],
    )

    llvm_ir = Phase(
        project=project,
        name="Enzyme MLIR Lower to LLVM IR",
        inputs_from=[llvm_dialect],
        executor=MLIRTranslateExecutor(),
    )

    clang = ClangCompileLLVM()
    clang.optimize(3)
    clang.ignore_override_module()
    return Phase(
        project=project,
        name="LLVM IR to object file",
        inputs_from=[llvm_ir],
        executor=clang,
    )


def clang_dynamiclib(project: Project, inputs_from: list[Phase], output: str):
    build = GccLink()
    if platform.system() == "Darwin":
        build.output_type = "library"
        build.output_extension = "dylib"
        build.add_argument("-dynamiclib")
    else:
        build.create_shared_library()

    Phase(
        project=project,
        name="Build dynamic lib",
        inputs_from=inputs_from,
        inputs=[str(C_RUNNER_UTILS)],
        executor=build,
        output=output,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mlir_file")
    args = parser.parse_args()
    print(jit_file(args.mlir_file))
