// TODO(jacob): Optimize memory allocations and copies
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
module  {
  memref.global "private" constant @__constant_512xf32 : memref<512xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_512x64xf32 : memref<512x64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10xf32 : memref<10xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10x64xf32 : memref<10x64xf32> = dense<0.000000e+00>
  func.func @ebatched_cross_entropy(%arg0: memref<10x64xf32>, %arg1: memref<64xi32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %c64 = arith.constant 64 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.subview %arg0[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32>
    %1 = memref.alloca() : memref<64xf32>
    linalg.copy ins(%0 : memref<64xf32>) outs(%1 : memref<64xf32>)
    scf.for %arg2 = %c1 to %c10 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %7 = memref.load %arg0[%arg2, %arg3] : memref<10x64xf32>
        %8 = memref.load %1[%arg3] : memref<64xf32>
        %9 = arith.cmpf ogt, %7, %8 : f32
        %10 = arith.select %9, %7, %8 : f32
        memref.store %10, %1[%arg3] : memref<64xf32>
      }
    }
    %2 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %3 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : memref<10x64xf32>, memref<64xf32>) outs(%3 : memref<10x64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %7 = arith.subf %arg2, %arg3 : f32
      %8 = math.exp %7 : f32
      linalg.yield %8 : f32
    }
    %4 = memref.alloca() : memref<64xf32>
    linalg.copy ins(%2 : memref<64xf32>) outs(%4 : memref<64xf32>)
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "parallel"]} ins(%3 : memref<10x64xf32>) outs(%4 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %7 = arith.addf %arg2, %arg3 : f32
      linalg.yield %7 : f32
    }
    %5 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %7 = memref.load %arg1[%arg2] : memref<64xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %3[%8, %arg2] : memref<10x64xf32>
      %10 = memref.load %4[%arg2] : memref<64xf32>
      %11 = arith.divf %9, %10 : f32
      %12 = math.log %11 : f32
      %13 = arith.negf %12 : f32
      %14 = arith.addf %arg3, %13 : f32
      scf.yield %14 : f32
    }
    memref.dealloc %3 : memref<10x64xf32>
    %6 = arith.divf %5, %cst_0 : f32
    return %6 : f32
  }
  func.func @mnist_mlp(%arg0: memref<784x64xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %1 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %2 = memref.get_global @__constant_10x64xf32 : memref<10x64xf32>
    %3 = memref.alloc() : memref<512x64xf32>
    linalg.copy ins(%0 : memref<512x64xf32>) outs(%3 : memref<512x64xf32>)
    linalg.matmul ins(%arg2, %arg0 : memref<512x784xf32>, memref<784x64xf32>) outs(%3 : memref<512x64xf32>)
    %4 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %arg3 : memref<512x64xf32>, memref<512xf32>) outs(%4 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %12 = arith.addf %arg8, %arg9 : f32
      linalg.yield %12 : f32
    }
    memref.dealloc %3 : memref<512x64xf32>
    %5 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<512x64xf32>) outs(%5 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %12 = arith.cmpf ogt, %arg8, %cst : f32
      %13 = arith.select %12, %arg8, %cst : f32
      linalg.yield %13 : f32
    }
    memref.dealloc %4 : memref<512x64xf32>
    %6 = memref.alloc() : memref<512x64xf32>
    linalg.copy ins(%1 : memref<512x64xf32>) outs(%6 : memref<512x64xf32>)
    linalg.matmul ins(%arg4, %5 : memref<512x512xf32>, memref<512x64xf32>) outs(%6 : memref<512x64xf32>)
    memref.dealloc %5 : memref<512x64xf32>
    %7 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %arg5 : memref<512x64xf32>, memref<512xf32>) outs(%7 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %12 = arith.addf %arg8, %arg9 : f32
      linalg.yield %12 : f32
    }
    memref.dealloc %6 : memref<512x64xf32>
    %8 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<512x64xf32>) outs(%8 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %12 = arith.cmpf ogt, %arg8, %cst : f32
      %13 = arith.select %12, %arg8, %cst : f32
      linalg.yield %13 : f32
    }
    memref.dealloc %7 : memref<512x64xf32>
    %9 = memref.alloc() : memref<10x64xf32>
    linalg.copy ins(%2 : memref<10x64xf32>) outs(%9 : memref<10x64xf32>)
    linalg.matmul ins(%arg6, %8 : memref<10x512xf32>, memref<512x64xf32>) outs(%9 : memref<10x64xf32>)
    memref.dealloc %8 : memref<512x64xf32>
    %10 = memref.alloc() : memref<10x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %arg7 : memref<10x64xf32>, memref<10xf32>) outs(%10 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %12 = arith.addf %arg8, %arg9 : f32
      linalg.yield %12 : f32
    }
    memref.dealloc %9 : memref<10x64xf32>
    %11 = call @ebatched_cross_entropy(%10, %arg1) : (memref<10x64xf32>, memref<64xi32>) -> f32
    memref.dealloc %10 : memref<10x64xf32>
    return %11 : f32
  }

  func.func @enzyme_mnist_mlp(
    %arg0: memref<784x64xf32>,
    %arg1: memref<64xi32>,
    %arg2: memref<512x784xf32>,
    %arg3: memref<512xf32>,
    %arg4: memref<512x512xf32>,
    %arg5: memref<512xf32>,
    %arg6: memref<10x512xf32>,
    %arg7: memref<10xf32>
  ) -> (
    memref<512x784xf32>,
    memref<512xf32>,
    memref<512x512xf32>,
    memref<512xf32>,
    memref<10x512xf32>,
    memref<10xf32>
  ) {
    %dweight0 = memref.alloc() : memref<512x784xf32>
    %dbias0   = memref.alloc() : memref<512xf32>
    %dweight1 = memref.alloc() : memref<512x512xf32>
    %dbias1   = memref.alloc() : memref<512xf32>
    %dweight2 = memref.alloc() : memref<10x512xf32>
    %dbias2   = memref.alloc() : memref<10xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%dweight0 : memref<512x784xf32>)
    linalg.fill ins(%zero : f32) outs(%dbias0 : memref<512xf32>)
    linalg.fill ins(%zero : f32) outs(%dweight1 : memref<512x512xf32>)
    linalg.fill ins(%zero : f32) outs(%dbias1 : memref<512xf32>)
    linalg.fill ins(%zero : f32) outs(%dweight2 : memref<10x512xf32>)
    linalg.fill ins(%zero : f32) outs(%dbias2 : memref<10xf32>)

    enzyme.autodiff @mnist_mlp(
      %arg0, %arg1,
      %arg2, %dweight0,
      %arg3, %dbias0,
      %arg4, %dweight1,
      %arg5, %dbias1,
      %arg6, %dweight2,
      %arg7, %dbias2
    ) {
      activity = [
        #enzyme<activity enzyme_const>,
        #enzyme<activity enzyme_const>,
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_dup>
      ]
    } : (
      memref<784x64xf32>,
      memref<64xi32>,
      memref<512x784xf32>,
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10x512xf32>,
      memref<10xf32>,
      memref<10xf32>
    ) -> f32

    return %dweight0, %dbias0, %dweight1, %dbias1, %dweight2, %dbias2 :
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10xf32>
  }
}
