func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()

func.func @ppow(%x: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  cf.br ^bb1(%c0, %cst : index, f64)
^bb1(%0: index, %r: f64):  // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %0, %c10 : index
  cf.cond_br %2, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %3 = arith.mulf %r, %x : f64
  %4 = arith.addi %0, %c1 : index
  cf.br ^bb1(%4, %3 : index, f64)
^bb3:  // pred: ^bb1
  return %r : f64
}

func.func @main() {
  %x = arith.constant 1.3 : f64
  %dx = arith.constant 0.0 : f64
  %g = arith.constant 1.0 : f64
  %c10 = arith.constant 10 : index
  %res = enzyme.autodiff @ppow(%x, %g) {activity=[#enzyme<activity enzyme_out>]} : (f64, f64) -> f64
  call @printF64(%res) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
