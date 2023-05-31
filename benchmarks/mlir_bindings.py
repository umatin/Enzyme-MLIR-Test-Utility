import ctypes
import platform
import pathlib
import numpy as np
from numpy.typing import NDArray


with open(pathlib.Path(__file__).parents[1] / "paths.txt", "r") as f:
    mlir_path, enzyme_path = [pathlib.Path(line.strip()) for line in f.readlines()]

LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"
MLIR_LIB = mlir_path / "lib"
RUNNER_UTILS = MLIR_LIB / f"libmlir_runner_utils.{LIB_EXT}"
C_RUNNER_UTILS = MLIR_LIB / f"libmlir_c_runner_utils.{LIB_EXT}"

double_ptr = ctypes.POINTER(ctypes.c_double)
float_ptr = ctypes.POINTER(ctypes.c_float)
cstdlib = ctypes.cdll.LoadLibrary("libSystem.dylib")
cstdlib.free.argtypes = [ctypes.c_void_p]
cstdlib.free.restype = ctypes.c_void_p


class MemRefDescriptor(ctypes.Structure):
    freed = False
    nparr: NDArray = None

    def to_numpy(self):
        assert not self.freed, "Memory was freed"
        if not self.nparr:
            self.nparr = np.ctypeslib.as_array(self.aligned, self.shape).copy()
        self.free()
        return self.nparr

    def free(self):
        assert not self.freed, "Memory was already freed"
        cstdlib.free(self.allocated)
        self.freed = True


class F32Descriptor1D(MemRefDescriptor):
    _fields_ = [
        ("allocated", float_ptr),
        ("aligned", float_ptr),
        ("offset", ctypes.c_longlong),
        ("size", ctypes.c_longlong),
        ("stride", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size]


class F32Descriptor2D(MemRefDescriptor):
    _fields_ = [
        ("allocated", float_ptr),
        ("aligned", float_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1]


class F64Descriptor1D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size", ctypes.c_longlong),
        ("stride", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size]


class F64Descriptor2D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1]


class F64Descriptor3D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("size_2", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
        ("stride_2", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1, self.size_2]


class F64Descriptor4D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("size_2", ctypes.c_longlong),
        ("size_3", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
        ("stride_2", ctypes.c_longlong),
        ("stride_3", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1, self.size_2, self.size_3]


memref_1d_f32 = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_2d_f32 = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 5
memref_1d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_1d_int = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_1d_index = [
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_2d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 5
memref_2d_int = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 5
memref_3d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 7
memref_4d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=4, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 9


def ndto_args(arr):
    if not isinstance(arr, np.ndarray):
        assert isinstance(arr, np.float64) or isinstance(
            arr, np.int64
        ), f"Unexpected argument type: '{type(arr)}'"
        return (arr,)
    return (
        (arr, arr, 0)
        + arr.shape
        + tuple(stride // arr.itemsize for stride in arr.strides)
    )


TMP_DIR = pathlib.Path(__file__).parent / "build" / "osx64"


class MNISTGrad(ctypes.Structure):
    _fields_ = [
        ("dweight0", F32Descriptor2D),
        ("dbias0", F32Descriptor1D),
        ("dweight1", F32Descriptor2D),
        ("dbias1", F32Descriptor1D),
        ("dweight2", F32Descriptor2D),
        ("dbias2", F32Descriptor1D),
    ]


def struct_to_tuple(s):
    if isinstance(s, float):
        return s
    elif isinstance(s, MemRefDescriptor):
        return s.to_numpy()
    descriptors = (getattr(s, field[0]) for field in s._fields_)
    return (
        (desc if isinstance(desc, float) else desc.to_numpy()) for desc in descriptors
    )


ctypes.CDLL(RUNNER_UTILS)
ctypes.CDLL(C_RUNNER_UTILS)
mlirlib = ctypes.CDLL(TMP_DIR / "enzyme_mlir_benchmarks.dylib")


mlp_args = (
    memref_2d_f32
    + memref_1d_int
    + memref_2d_f32
    + memref_1d_f32
    + memref_2d_f32
    + memref_1d_f32
    + memref_2d_f32
    + memref_1d_f32
)
mlirlib.mnist_mlp.argtypes = mlp_args
mlirlib.mnist_mlp.restype = ctypes.c_float
# mlirlib.enzyme_mnist_mlp.argtypes = mlp_args
# mlirlib.enzyme_mnist_mlp.restype = MNISTGrad


def wrap(mlir_func):
    def wrapped(*args):
        args = tuple(arg for ndarr in args for arg in ndto_args(ndarr))
        return struct_to_tuple(mlir_func(*args))

    return wrapped


mnist_mlp_primal = wrap(mlirlib.mnist_mlp)
# enzyme_mnist_mlp = wrap(mlirlib.enzyme_mnist_mlp)
