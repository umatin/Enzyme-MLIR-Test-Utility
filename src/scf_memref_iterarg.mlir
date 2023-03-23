func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

func.func @memref_iterarg(%x: memref<2xf64>) -> memref<2xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 3 : index
  %res = scf.for %iv = %c0 to %n step %c1 iter_args(%r_it = %x) -> memref<2xf64> {
    %0 = memref.load %r_it[%c0] : memref<2xf64>
    %1 = memref.load %r_it[%c1] : memref<2xf64>
    %2 = arith.mulf %0, %1 : f64
    %3 = arith.addf %0, %1 : f64
    memref.store %2, %r_it[%c0] : memref<2xf64>
    memref.store %3, %r_it[%c1] : memref<2xf64>
    scf.yield %r_it : memref<2xf64>
  }

  return %res : memref<2xf64>
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %a = arith.constant 1.3 : f64
  %b = arith.constant 1.6 : f64
  %x = memref.alloca() : memref<2xf64>
  %dx = memref.alloca() : memref<2xf64>
  %g = memref.alloca() : memref<2xf64>
  memref.store %a, %x[%c0] : memref<2xf64>
  memref.store %b, %x[%c1] : memref<2xf64>
  linalg.fill ins(%zero : f64) outs(%dx : memref<2xf64>)
  linalg.fill ins(%one : f64) outs(%g : memref<2xf64>)
  enzyme.autodiff @memref_iterarg(%x, %dx, %g) {activity=[#enzyme<activity enzyme_dup>]} : (memref<2xf64>, memref<2xf64>, memref<2xf64>) -> ()

  %casted = memref.cast %dx : memref<2xf64> to memref<*xf64>
  call @printMemrefF64(%casted) : (memref<*xf64>) -> ()
  return
}
