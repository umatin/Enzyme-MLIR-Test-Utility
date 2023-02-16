# Test Utils for Enzyme Mlir Reverse Mode

Checks for approximate equality between Enzyme(Mlir) and Enzyme(LLVM) and other nice utilities

## Setup
Write the absolute path to the mlir build directory and enzyme directory into paths.txt each in a new line.

## File Format
For printing exclusively use:
```
func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()
```
After each printed value add a newline.
