#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 原始的 CPU 实现
void get_sub_mat(int *x, int s, int *sub_mat, int lin, int col) {
    int i, j;
    int index = 0;
    for (i = 0; i < s; i++) {
        if (i == lin) continue;
        for (j = 0; j < s; j++) {
            if (j == col) continue;
            sub_mat[index++] = x[i * s + j];
        }
    }
}

int det_mat_cpu(int *x, int s) {
    if (s == 1) {
        return x[0];
    }
    if (s == 2) {
        return x[0] * x[3] - x[1] * x[2];
    }

    int outcome = 0;
    int *sub_mat = (int *)malloc((s - 1) * (s - 1) * sizeof(int));
    if (sub_mat == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    for (int i = 0; i < s; i++) {
        get_sub_mat(x, s, sub_mat, 0, i);
        outcome += x[i] * det_mat_cpu(sub_mat, s - 1) * ((i % 2 == 0) ? 1 : -1);
    }

    free(sub_mat);
    return outcome;
}

// 修改后的设备函数：传递子矩阵缓冲区和动态偏移
__device__ int det_mat_device(int *x, int s, int *sub_mat_buffer, int &buffer_offset) {
    if (s == 1) {
        return x[0];
    }
    if (s == 2) {
        return x[0] * x[3] - x[1] * x[2];
    }

    int outcome = 0;
    int current_offset = atomicAdd(&buffer_offset, s * (s-1)*(s-1)); // 原子操作分配空间
    int *sub_mat = sub_mat_buffer + current_offset;

    for (int i = 0; i < s; i++) {
        // 生成子矩阵到sub_mat的相应位置
        int index = 0;
        for (int row = 0; row < s; row++) {
            if (row == 0) continue;
            for (int col = 0; col < s; col++) {
                if (col == i) continue;
                sub_mat[i*(s-1)*(s-1) + index++] = x[row * s + col];
            }
        }
        // 递归计算子行列式
        outcome += x[i] * det_mat_device(sub_mat + i*(s-1)*(s-1), s-1, sub_mat_buffer, buffer_offset) * ((i % 2 == 0) ? 1 : -1);
    }

    return outcome;
}

__global__ void det_mat_kernel(int *x, int s, int *result, int *sub_mat_buffer, int *d_offset) {
    *d_offset = 0; // 初始化偏移量
    *result = det_mat_device(x, s, sub_mat_buffer, *d_offset);
}

int det_mat_cuda(int *x, int s) {
    int *d_x, *d_result, *d_sub_mat, *d_offset;
    int size = s * s * (s - 1) * (s - 1); // 更精确的缓冲区大小估计
    cudaMalloc((void **)&d_x, sizeof(int) * s * s);
    cudaMalloc((void **)&d_result, sizeof(int));
    cudaMalloc((void **)&d_sub_mat, sizeof(int) * size);
    cudaMalloc((void **)&d_offset, sizeof(int));

    cudaMemcpy(d_x, x, sizeof(int) * s * s, cudaMemcpyHostToDevice);
    cudaMemset(d_offset, 0, sizeof(int));

    det_mat_kernel<<<1, 1>>>(d_x, s, d_result, d_sub_mat, d_offset);
    cudaDeviceSynchronize();

    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_result);
    cudaFree(d_sub_mat);
    cudaFree(d_offset);
    return result;
}

int main() {
    int s;
    printf("请输入矩阵的大小（方阵）：");
    scanf("%d", &s);

    int *matrix = (int *)malloc(s * s * sizeof(int));
    if (matrix == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    printf("请输入矩阵的元素（按行输入）：\n");
    for (int i = 0; i < s * s; i++) {
        scanf("%d", &matrix[i]);
    }

    // 计算行列式（CPU 实现）
    int determinant_cpu = det_mat_cpu(matrix, s);
    printf("矩阵的行列式（CPU 实现）：%d\n", determinant_cpu);

    // 计算行列式（CUDA 实现）
    int determinant_cuda = det_mat_cuda(matrix, s);
    printf("矩阵的行列式（CUDA 实现）：%d\n", determinant_cuda);

    free(matrix);
    return 0;
}