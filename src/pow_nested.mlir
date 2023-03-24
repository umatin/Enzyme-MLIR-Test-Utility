func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()

func.func @ppow(%x: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 3 : index
  %res = scf.for %iv = %c0 to %n step %c1 iter_args(%r_it = %cst) -> f64 {
    %res_inner = scf.for %jv = %c0 to %n step %c1 iter_args(%r_inner = %r_it) -> f64 {
      %r_next = arith.mulf %r_inner, %x : f64
      scf.yield %r_next : f64
    }
    scf.yield %res_inner : f64
  }
  return %res : f64
}

func.func @main() {
  %x = arith.constant 1.3 : f64
  %g = arith.constant 1.0 : f64
  %c10 = arith.constant 10 : index
  %res = enzyme.autodiff @ppow(%x, %g) {activity=[#enzyme<activity enzyme_out>]} : (f64, f64) -> f64
  call @printF64(%res) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
