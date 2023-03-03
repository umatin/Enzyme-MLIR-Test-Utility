import platform
import pathlib
import subprocess

with open("paths.txt", "r") as f:
    mlir_path, enzyme_path = [pathlib.Path(line.strip()) for line in f.readlines()]

LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"
MLIR_BIN = mlir_path / "bin"
MLIR_LIB = mlir_path / "lib"
ENZYMEMLIR_OPT = enzyme_path / "enzyme/build/Enzyme/MLIR/enzymemlir-opt"
ENZYME_DYLIB = enzyme_path / f"enzyme/build/Enzyme/LLVMEnzyme-16.{LIB_EXT}"
OPT = MLIR_BIN / "opt"
LLI = MLIR_BIN / "lli"
MLIR_OPT = MLIR_BIN / "mlir-opt"
MLIR_TRANSLATE = MLIR_BIN / "mlir-translate"
MLIR_CPU_RUNNER = MLIR_BIN / "mlir-cpu-runner"
RUNNER_UTILS = MLIR_LIB / f"libmlir_runner_utils.{LIB_EXT}"
C_RUNNER_UTILS = MLIR_LIB / f"libmlir_c_runner_utils.{LIB_EXT}"


def run_safe(args, stdin: bytes = None) -> bytes:
    try:
        p = subprocess.run(args, input=stdin, check=True, capture_output=True)
        if p.stderr:
            print(p.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode("utf-8"))
        raise Exception(e.stderr.decode("utf-8"))
    return p.stdout


def differentiate(source: bytes) -> bytes:
    return run_safe([ENZYMEMLIR_OPT, "-enzyme", "-symbol-dce"], stdin=source)


def lower_to_llvm_dialect(source: bytes) -> bytes:
    lower_enzyme_args = [
        "-convert-enzyme-shadowed-gradient-to-cache",
        "-convert-enzyme-to-memref",
        "-canonicalize",
        "-reconcile-unrealized-casts",
    ]
    memref_dialect = run_safe([ENZYMEMLIR_OPT] + lower_enzyme_args, stdin=source)

    lower_to_llvm_args = [
        "-convert-scf-to-cf",
        "-convert-cf-to-llvm",
        "-convert-func-to-llvm",
        "-convert-memref-to-llvm",
        "-convert-arith-to-llvm",
        "-reconcile-unrealized-casts",
    ]
    return run_safe([MLIR_OPT] + lower_to_llvm_args, stdin=memref_dialect)


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
