func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @mm(%x: memref<2x2xf64>) -> f64 {
  %space = memref.alloc() : memref<2x2xf64>
  //%zero = arith.constant 0.0 : f64
  //linalg.fill ins(%zero : f64) outs(%space : memref<2x2xf64>)

  linalg.matmul ins(%x, %x: memref<2x2xf64>, memref<2x2xf64>)
             outs(%space: memref<2x2xf64>)
  %0 = arith.constant 0 : index
  %result = memref.load %space[%0, %0] : memref<2x2xf64>
  return %result : f64
}

memref.global @x : memref<2x2xf64> = dense<[[1., 2.], [3., 4.]]>

func.func @main() {
  %zero = arith.constant 0.0 : f64
  %x = memref.get_global @x : memref<2x2xf64>
  %dx = memref.alloca() : memref<2x2xf64>

  %g = arith.constant 1.0 : f64
  linalg.fill ins(%zero : f64) outs(%dx : memref<2x2xf64>)

  enzyme.autodiff @mm(%x, %dx, %g) {activity=[#enzyme<activity enzyme_dup>]} : (memref<2x2xf64>, memref<2x2xf64>, f64) -> ()

  %casted = memref.cast %dx : memref<2x2xf64> to memref<*xf64>
  call @printMemrefF64(%casted) : (memref<*xf64>) -> ()
  return
}
