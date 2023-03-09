func.func private @printF64(f64) -> ()
func.func @ppow(%arg0: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %3 = arith.addf %cst, %arg0 : f64
  %4 = arith.addf %3, %arg0 : f64
  return %4 : f64
}
func.func @main() {
  %cst = arith.constant 2.00000e+00 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  %0 = enzyme.autodiff @ppow(%cst, %cst_1) {activity = [#enzyme<activity enzyme_out>]} : (f64, f64) -> f64
  call @printF64(%0) : (f64) -> ()
  return
}
