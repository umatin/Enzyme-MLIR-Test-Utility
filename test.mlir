
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }
func.func private @printNewline() -> ()

memref.global "private" @gv0 : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>
func.func @test1DMemref() {
  %0 = memref.get_global @gv0 : memref<4xf32>
  %U = memref.cast %0 : memref<4xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() -> () {
  call @test1DMemref() : () -> ()
  return
}


