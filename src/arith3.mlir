func.func private @printF64(f64) -> ()

func.func @f(%x : f64) -> f64 {
    %max = arith.constant 10.0 : f64
    cf.br ^bb1(%x : f64)
^bb1(%x_ : f64):
    %x_out = arith.addf %x, %x : f64
    %y = arith.subf %x_out, %max : f64
    %1 = arith.cmpf ogt, %y, %max : f64
    cf.cond_br %1, ^bb1(%x_out : f64), ^bb2
^bb2:
    return %x : f64
}

func.func @df(%x : f64, %gradient : f64) -> (f64) {
    %r = enzyme.autodiff @f(%x, %gradient) { activity=[#enzyme<activity enzyme_out>] } : (f64, f64) -> (f64)
    return %r: f64
}

func.func @main() -> () {
    %cst0 = arith.constant 0.00 : f64
    %cst1 = arith.constant 1.0 : f64
    %f = call @df(%cst1, %cst1) : (f64, f64) -> (f64)
    call @printF64(%f) : (f64) -> ()
    return
}