func.func private @printF64(f64) -> ()

func.func @f(%x : f64) -> f64 {
        cf.br ^bb1(%x : f64)
        
    ^bb1(%0 : f64):
        %cst1 = arith.constant 1.000000e+00 : f64
        %2 = arith.addf %0, %cst1 : f64
        cf.br ^bb2(%2 : f64)

    ^bb2(%1 : f64):
        %cst = arith.constant 0.000000e+00 : f64
        %flag = arith.cmpf ult, %1, %cst : f64
        cf.cond_br %flag, ^bb2(%1 : f64), ^bb3(%1 : f64)

    ^bb3(%ret : f64):
        return %ret : f64
}

func.func @df(%x : f64, %dx : f64, %gradient : f64) -> (f64) {
    %r = enzyme.diff @f(%x, %dx, %gradient) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64, f64) -> (f64)
    return %r: f64
}

func.func @main() -> () {
    %cst0 = arith.constant 0.00 : f64
    %cst1 = arith.constant 23.01 : f64
    %f = call @df(%cst0, %cst0, %cst1) : (f64, f64, f64) -> (f64)
    call @printF64(%f) : (f64) -> ()
    return
}
