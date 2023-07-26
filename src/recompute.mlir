func.func private @printF64(f64) -> ()

func.func @f(%x : f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    %w = arith.mulf %r, %r : f64
    return %w : f64
}

func.func @main() -> () {
    %cst0 = arith.constant 0.00 : f64
    %cst1 = arith.constant 1.00 : f64
    %f = enzyme.autodiff @f(%cst0, %cst1) { activity=[#enzyme<activity enzyme_out>] } : (f64, f64) -> (f64)
    call @printF64(%f) : (f64) -> ()
    return
}
