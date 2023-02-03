func.func private @printF64(f64) -> ()

func.func @f(%x : f64) -> f64 {
    %r = arith.addf %x, %x : f64
    return %r : f64
}
func.func @df(%x : f64, %dx : f64, %gradient : f64) -> (f64) {
    %r = enzyme.diff @f(%x, %dx, %gradient) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64, f64) -> (f64)
    return %r : f64
}
func.func @main() -> () {
    %cst0 = arith.constant 0.00 : f64
    %cst1 = arith.constant 1.00 : f64
    %f = call @df(%cst0, %cst0, %cst1) : (f64, f64, f64) -> (f64)
    call @printF64(%f) : (f64) -> ()
    return
}