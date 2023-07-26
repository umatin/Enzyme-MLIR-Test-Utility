func.func private @printF64(f64) -> ()
func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()

func.func @f(%arg0: memref<f64>, %arg1 : index) -> f64 {
    // Constants
    %cst1 = arith.constant 1 : index
    %cst0 = arith.constant 0 : index

    %x = memref.load %arg0[] : memref<f64>
    %i_0 = arith.constant 0 : index

    cf.br ^bb1(%i_0, %x : index, f64)
^bb1(%i : index, %r : f64):
    // Increase i
    %i_p1 = arith.addi %i, %cst1 : index

    // Alloc something
    %mem = memref.alloc (%i_p1) : memref<?xf64>
    
    //Imagine some weird scheme to fill %here which is invertible etc.
    
    // Reduce
    %res = scf.for %iv = %cst0 to %i_p1 step %cst1 iter_args(%r_it = %r) -> f64 {
        %loaded = memref.load %mem[%iv] : memref<?xf64>
        %out = arith.mulf %r_it, %x : f64
        scf.yield %out : f64
    }
    
    %r_new = arith.addf %res, %r : f64

    %condition = arith.cmpi ugt, %i_p1, %arg1 : index
    cf.cond_br %condition, ^bb0, ^bb1(%i_p1, %r_new : index, f64)
^bb0:
    return %x : f64
}

func.func @main() -> () {
    %ind = arith.constant 5 : index
    
    %dout = arith.constant 36.0 : f64
    %mem = memref.alloc () : memref<f64>
    %dmem = memref.alloc () : memref<f64>

    // ad
    enzyme.autodiff @f(%mem, %dmem, %ind, %dout) {activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>]} : (memref<f64>, memref<f64>, index, f64) -> ()

    // print after
    %out = memref.load %dmem[] : memref<f64>
    call @printNewline() : () -> ()
    call @printF64(%out) : (f64) -> ()

    return
}
