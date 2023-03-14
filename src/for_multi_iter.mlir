func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()

func.func @multi_iter(%arg0: f64) -> f64 {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 1.000000e+00 : f64
  %0:2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst_0, %arg3 = %cst) -> (f64, f64) {
    %2 = arith.mulf %arg2, %arg0 : f64
    %3 = arith.addf %arg3, %arg0 : f64
    scf.yield %2, %3 : f64, f64
  }
  %1 = arith.addf %0#0, %0#1 : f64
  return %1 : f64
}

func.func @main() {
  %x = arith.constant 1.3 : f64
  %g = arith.constant 1.0 : f64
  %c10 = arith.constant 10 : index
  %res = enzyme.autodiff @multi_iter(%x, %g) {activity=[#enzyme<activity enzyme_out>]} : (f64, f64) -> f64
  call @printF64(%res) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
