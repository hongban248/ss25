import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# 创建一个简单的CUDA内核
kernel_code = """
__global__ void hello_from_gpu() {
    printf("Hello from GPU! ThreadIdx: %d\\n", threadIdx.x);
}
"""

# 编译内核
mod = SourceModule(kernel_code)
hello_from_gpu = mod.get_function("hello_from_gpu")

# 启动内核
while True:

    hello_from_gpu(block=(100, 1, 1))
