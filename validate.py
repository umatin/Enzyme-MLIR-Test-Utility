import os
import subprocess
import re
import argparse

f  =  open("paths.txt", "r")
mlir_path = f.readline().rstrip()
enzyme_path = f.readline().rstrip()

ENZYMEMLIR_OPT = enzyme_path + "/enzyme/build/Enzyme/MLIR/enzymemlir-opt"
ENZYME_DYLIB = enzyme_path + "/enzyme/build/Enzyme/LLVMEnzyme-16.so"
OPT = mlir_path + "/bin/opt"
LLI = mlir_path + "/bin/lli"
MLIR_OPT = mlir_path + "/bin/mlir-opt"
MLIR_TRANSLATE = mlir_path + "/bin/mlir-translate"
RUNNER_UTILS = mlir_path + "/lib/libmlir_runner_utils.so"
C_RUNNER_UTILS = mlir_path + "/lib/libmlir_c_runner_utils.so"

def lower_base_enzyme(filename):
    fname = "out/" + filename.split("/")[-1][:-5] + "_base.mlir"
    LOWER = ENZYMEMLIR_OPT + " " + filename + " -lower-to-llvm-enzyme -reconcile-unrealized-casts -o " + fname
    MEMREF_TO_LLVM = MLIR_OPT + " " + fname + " -convert-memref-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts"
    TRANSLATE = MLIR_TRANSLATE + " -mlir-to-llvmir"
    RUN_ENZYME = OPT + " -load " + ENZYME_DYLIB + " -enable-new-pm=0 -enzyme -S"
    JIT = LLI + " -load=" + RUNNER_UTILS + " -load=" + C_RUNNER_UTILS

    return LOWER + " && " + MEMREF_TO_LLVM + " | " + TRANSLATE + " | " + RUN_ENZYME + " | " + JIT

def lower_mlir_enzyme(filename):
    fname = "out/" + filename.split("/")[-1][:-5] + "_mlir.mlir"
    fname2 = "out/" + filename.split("/")[-1][:-5] + "_mlir2.mlir"
    fname_llvm = "out/" + filename.split("/")[-1][:-5] + "_llvm.ll"
    
    #LOWER = ENZYMEMLIR_OPT + " " + filename + " -enzyme -simplify-memref-cache -add-to-op-to-index-and-load -symbol-dce -o " + fname
    LOWER = ENZYMEMLIR_OPT + " " + filename + " -enzyme -simplify-memref-cache -add-to-op-to-split -symbol-dce -o " + fname
    
    LOWER2 = ENZYMEMLIR_OPT + " " + fname + " -convert-enzyme-shadowed-gradient-to-cache --convert-enzyme-to-memref -reconcile-unrealized-casts -o " + fname2
    #MEMREF_TO_LLVM = MLIR_OPT + " " + fname2 + " -convert-linalg-to-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -reconcile-unrealized-casts"
    MEMREF_TO_LLVM = MLIR_OPT + " " + fname2 + " --test-lower-to-llvm"
    
    TRANSLATE = MLIR_TRANSLATE + " -mlir-to-llvmir -o " + fname_llvm
    JIT = LLI + " -load=" + RUNNER_UTILS + " -load=" + C_RUNNER_UTILS + " " + fname_llvm

    return LOWER + " && " + LOWER2 + " && " + MEMREF_TO_LLVM + " | " + TRANSLATE+ " && " + JIT

def run_command(command):
    ret = subprocess.run(command, shell=True, text = True, capture_output = True)
    return ret.stdout, ret.stderr
    
def extract_floats(s):
    out = re.findall(r"[-+]?(?:\d*\.*\d+)", s)
    ret = []
    for x in out:
        try:
            ret.append(float(x))
        except ValueError:
            pass
    return ret

def all_close(a,b, eps = 0.00001):
    if len(a) != len(b):
        return False
    else:
        x = [abs(a_-b_) for a_, b_ in zip(a,b)]
        if max(x) > eps:
            return False
    return True

def compare_output(base, mlir):
    for l1, l2 in zip(base.splitlines(),mlir.splitlines()):
        if l1 != l2:
            if "Memref" not in l1:
                values1 = extract_floats(l1)
                values2 = extract_floats(l2)
                if not all_close(values1, values2):
                    print("Found values which are not close:")
                    print(values1)
                    print(values2)
                    return False
            else:
                if "Memref" not in l2:
                    print("Completely different Lines")
                    print(l1)
                    print(l2)
    return True


parser = argparse.ArgumentParser(
                    prog = 'Test Enzyme MLIR',
                    description = 'Checks if Enzyme MLIR returns approximately the same values as Enzyme(llvm)',
                    epilog = '')

parser.add_argument('filename')
parser.add_argument('-m', '--mlir-help', action='store_true')
parser.add_argument('-e', '--enzyme-help', action='store_true')
parser.add_argument('-s', '--show-output', action='store_true')

args = parser.parse_args()

if args.mlir_help:
    os.system(MLIR_OPT + " --help")
if args.enzyme_help:
    os.system(ENZYMEMLIR_OPT + " --help")

file = args.filename

result_base, err_base = run_command(lower_base_enzyme(file))
result_mlir, err_mlir = run_command(lower_mlir_enzyme(file))

if args.show_output:
    print("LLVM")
    print(result_base)
if len(err_base) > 0:
    print("LLVM err")
    print(err_base)

if args.show_output:
    print("MLIR")
    print(result_mlir)
if len(err_mlir) > 0:
    print("MLIR err")
    print(err_mlir)

compare_output(result_base, result_mlir)