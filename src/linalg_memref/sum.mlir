func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @dot_prod(%x: memref<4xf64>) -> f64 {
  %space = memref.alloc() : memref<f64>
  %zero = arith.constant 0.0 : f64
  memref.store %zero, %space[] : memref<f64>
  linalg.generic
    {
      indexing_maps = [#map, #map1],
      iterator_types = ["reduction"]
    }
    ins(%x : memref<4xf64>)
    outs(%space : memref<f64>) {
      ^bb0(%in0: f64, %out: f64):
        %1 = arith.addf %in0, %out : f64
        linalg.yield %1 : f64
    }
  %result = memref.load %space[] : memref<f64>
  return %result : f64
}

memref.global @x : memref<4xf64> = dense<[1., 2., 3., 4.]>

func.func @main() {
  %zero = arith.constant 0.0 : f64
  %x = memref.get_global @x : memref<4xf64>
  %dx = memref.alloca() : memref<4xf64>
  %g = arith.constant 1.0 : f64
  linalg.fill ins(%zero : f64) outs(%dx : memref<4xf64>)
  enzyme.autodiff @dot_prod(%x, %dx, %g) {activity=[#enzyme<activity enzyme_dup>]} : (memref<4xf64>, memref<4xf64>, f64) -> ()

  %casted = memref.cast %dx : memref<4xf64> to memref<*xf64>
  call @printMemrefF64(%casted) : (memref<*xf64>) -> ()
  return
}
