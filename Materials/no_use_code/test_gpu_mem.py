import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time

# 计算需要分配的字节数
bytes_to_allocate = int(1.8 * 1024 * 1024 * 1024)  # 1.5 GB

# 分配显存
print("Allocating 1.5 GB of GPU memory...")
gpu_array = drv.mem_alloc(bytes_to_allocate)
print("Memory allocation complete.")

# 保持10秒
print("Holding memory for 10 seconds...")
time.sleep(10)

# 释放显存
print("Releasing GPU memory...")
del gpu_array
drv.Context.synchronize()  # 确保释放操作完成
print("Memory released. Exiting program.")
