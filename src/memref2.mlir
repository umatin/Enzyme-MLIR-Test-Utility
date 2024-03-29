func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()

func.func @f(%arg0: memref<?xf64>, %arg1 : f64) -> f64 {
    %c0 = arith.constant 0 : index
    %1 = memref.load %arg0[%c0] : memref<?xf64>

    return %1 : f64
}

func.func @df(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: f64, %arg4: f64) -> f64 {
    %r = enzyme.autodiff @f (%arg0, %arg1, %arg2, %arg4)  { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_out>] } : (memref<?xf64>, memref<?xf64>, f64, f64) -> (f64)
    return %r: f64
}

func.func @main() -> () {
    %cst0 = arith.constant 0.00 : f64
    %cst1 = arith.constant 1.00 : f64
    %csti1 = arith.constant 10 : index
    %c0 = arith.constant 0 : index

    %mem = memref.alloc (%csti1) : memref<?xf64>
    memref.store %cst1, %mem[%c0] : memref<?xf64>
    %mem_shadow = memref.alloc (%csti1) : memref<?xf64>
    linalg.fill ins(%cst0 : f64) outs(%mem_shadow : memref<?xf64>)
    %f = call @df(%mem, %mem_shadow, %cst1, %cst1) : (memref<?xf64>, memref<?xf64>, f64, f64) -> (f64)

    call @printF64(%f) : (f64) -> ()
    call @printNewline() : () -> ()

    %mstar = memref.cast %mem_shadow : memref<?xf64> to memref<*xf64>
    call @printMemrefF64(%mstar) : (memref<*xf64>) -> ()
    return
}