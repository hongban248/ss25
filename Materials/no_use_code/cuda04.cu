#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h> // 用于 sleep 函数

int main() {
    // 定义占用的显存大小（1.6GB）
    size_t size = 1.6 * 1024 * 1024 * 1024; // 1.6GB
    char *d_data;

    // 在 GPU 上分配显存
    cudaMalloc((void**)&d_data, size);
    if (d_data == NULL) {
        printf("显存分配失败！\n");
        return -1;
    }
    printf("成功占用 1.6GB 显存。\n");

    // 等待 10 秒
    sleep(10);

    // 释放显存
    cudaFree(d_data);
    printf("显存已释放。\n");

    return 0;
}