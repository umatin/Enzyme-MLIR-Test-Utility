import os
import subprocess
import re
import argparse

f  =  open("paths.txt", "r")
mlir_path = "/local/temp/JAX-Devving/mlir_enzyme/update/mlir-build"
f.readline()
enzyme_path = f.readline().rstrip()
CLANG_FORMAT = mlir_path + "/bin/clang-format"

def run_command(command):
    ret = subprocess.run(command, shell=True, text = True, capture_output = True)
    return ret.stdout, ret.stderr
    
def format(path):
    print(run_command(CLANG_FORMAT + " -i " + path + "*.cpp " + path + "*.h"))

MLIR = enzyme_path + "/enzyme/Enzyme/MLIR"
format(MLIR)
format(MLIR + "/Dialect/")
format(MLIR + "/Implementations/")
format(MLIR + "/Interfaces/")
format(MLIR + "/Passes/")