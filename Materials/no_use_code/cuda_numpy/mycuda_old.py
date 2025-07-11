import numpy as np
import ctypes

class arr:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape
        self.size = len(data)
        # 假设底层库是通过 ctypes 加载的
        self.lib = ctypes.CDLL("./mycuda_int.so")
        self.lib.multiple.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), 
                                      ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.multiple.restype = None

    def __repr__(self):
        return f"arr(data={self.data}, shape={self.shape})"

    def multiple(self, obj2):
        if type(obj2) != arr:
            # 标量乘法
            return arr([self.data[i] * obj2 for i in range(self.size)], self.shape)
        
        if self.shape[1] != obj2.shape[0]:
            raise ValueError("can not matrix multiple!!")

        # 将 Python 列表转换为 ctypes 指针
        data1 = (ctypes.c_int * len(self.data))(*self.data)
        data2 = (ctypes.c_int * len(obj2.data))(*obj2.data)
        result = (ctypes.c_int * (self.shape[0] * obj2.shape[1]))()

        # 调用底层库的 multiple 函数
        self.lib.multiple(data1, data2, result, self.shape[0], self.shape[1], obj2.shape[1])

        # 将结果转换为 Python 列表
        result_data = list(result)
        return arr(result_data, (self.shape[0], obj2.shape[1]))

# 测试代码
if __name__ == "__main__":
    # 测试标量乘法
    mat1 = arr([1, 2, 3, 4], (2, 2))
    scalar = 2
    result_scalar = mat1.multiple(scalar)
    print("Scalar multiplication:", result_scalar)

    # 测试矩阵乘法
    mat2 = arr([1, 2, 3, 4], (2, 2))
    mat3 = arr([5, 6, 7, 8], (2, 2))
    result_matrix = mat2.multiple(mat3)
    print("Matrix multiplication:", result_matrix)