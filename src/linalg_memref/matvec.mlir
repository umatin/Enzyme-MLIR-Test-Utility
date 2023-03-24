func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>

func.func @matvec(%M: memref<3x4xf64>, %x: memref<4xf64>, %out: memref<3xf64>) {
  linalg.generic
    {
      indexing_maps = [#map, #map2, #map1],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%M, %x : memref<3x4xf64>, memref<4xf64>)
    outs(%out : memref<3xf64>) {
      ^bb0(%in0: f64, %in1: f64, %out0: f64):
        %0 = arith.mulf %in0, %in1 : f64
        %1 = arith.addf %0, %out0 : f64
        linalg.yield %1 : f64
    }
  return
}

memref.global @M : memref<3x4xf64> = dense<[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]>
memref.global @x : memref<4xf64> = dense<[1., 2., 3., 4.]>

func.func @main() {
  %zero = arith.constant 0.0 : f64
  %M = memref.get_global @M : memref<3x4xf64>
  %x = memref.get_global @x : memref<4xf64>
  %out = memref.alloca() : memref<3xf64>

  %dM = memref.alloca() : memref<3x4xf64>
  %dx = memref.alloca() : memref<4xf64>
  %dout = memref.alloca() : memref<3xf64>

  %one = arith.constant 1.0 : f64
  linalg.fill ins(%zero : f64) outs(%out : memref<3xf64>)
  linalg.fill ins(%zero : f64) outs(%dM : memref<3x4xf64>)
  linalg.fill ins(%zero : f64) outs(%dx : memref<4xf64>)
  linalg.fill ins(%one : f64) outs(%dout : memref<3xf64>)
  enzyme.autodiff @matvec(%M, %dM, %x, %dx, %out, %dout) {
    activity=[
      #enzyme<activity enzyme_dup>,
      #enzyme<activity enzyme_dup>,
      #enzyme<activity enzyme_dup>
    ]} : (
      memref<3x4xf64>, memref<3x4xf64>,
      memref<4xf64>, memref<4xf64>,
      memref<3xf64>, memref<3xf64>
    ) -> ()

  %cast_dM = memref.cast %dM : memref<3x4xf64> to memref<*xf64>
  call @printMemrefF64(%cast_dM) : (memref<*xf64>) -> ()
  %cast_dx = memref.cast %dx : memref<4xf64> to memref<*xf64>
  call @printMemrefF64(%cast_dM) : (memref<*xf64>) -> ()
  return
}
