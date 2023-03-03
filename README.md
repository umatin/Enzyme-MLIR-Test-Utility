# Test Utils for Enzyme Mlir Reverse Mode

Checks for approximate equality between Enzyme(Mlir) and Enzyme(LLVM) and other nice utilities. Requires Python 3.6+.

## Setup
Write the absolute path to the mlir build directory and enzyme directory into paths.txt each in a new line.

Install dependencies (ideally in a virtual env):
```sh
pip install -r requirements.txt
```

Run tests:
```sh
pytest
```

## File Format
For printing exclusively use:
```
func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()
```
After each printed value add a newline.
