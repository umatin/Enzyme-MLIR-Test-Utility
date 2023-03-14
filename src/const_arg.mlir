func.func private @printF64(f64) -> ()

func.func @f(%x: f64, %n: i64) -> f64 {
  %casted = arith.sitofp %n : i64 to f64
  %r = arith.mulf %x, %casted : f64
  return %r : f64
}

func.func @main() -> () {
  %cst0 = arith.constant 0.00 : f64
  %cst1 = arith.constant 1.00 : f64
  %n = arith.constant 17 : i64
  %f = enzyme.autodiff @f(%cst0, %n, %cst1) { activity=[#enzyme<activity enzyme_out>, #enzyme<activity enzyme_const>] } : (f64, i64, f64) -> (f64)
  call @printF64(%f) : (f64) -> ()
  return
}
