func.func private @printF64(f64) -> ()
func.func @ppow(%arg0: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  cf.br ^bb1(%c0, %cst : i32, f64)
^bb1(%0: i32, %1: f64):  // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %0, %c10 : i32
  cf.cond_br %2, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %3 = arith.addf %1, %arg0 : f64
  %4 = arith.addi %0, %c1 : i32
  cf.br ^bb1(%4, %3 : i32, f64)
^bb3:  // pred: ^bb1
  return %1 : f64
}
func.func @main() {
  %cst = arith.constant 2.00000e+00 : f64
  %cst_0 = arith.constant 0.000000e+00 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  %0 = enzyme.autodiff @ppow(%cst, %cst_1) {activity = [#enzyme<activity enzyme_out>]} : (f64, f64) -> f64
  //%0 = call @ppow(%cst) : (f64) -> (f64)
  call @printF64(%0) : (f64) -> ()
  return
}
