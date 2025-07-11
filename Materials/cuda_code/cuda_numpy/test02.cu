#include"mycuda.h"

// CPU 上的矩阵乘法（int 类型）
int *matrix_multiply_cpu_int(int *x, int *y, int m, int s, int n) {
    int *result = (int *)malloc(m * n * sizeof(int));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = 0;
            for (int k = 0; k < s; k++) {
                result[i * n + j] += x[i * s + k] * y[k * n + j];
            }
        }
    }
    return result;
}

// CPU 上的矩阵乘法（double 类型）
double *matrix_multiply_cpu_double(double *x, double *y, int m, int s, int n) {
    double *result = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = 0.0;
            for (int k = 0; k < s; k++) {
                result[i * n + j] += x[i * s + k] * y[k * n + j];
            }
        }
    }
    return result;
}

// 比较两个矩阵是否相等（int 类型）
int compare_matrices_int(int *mat1, int *mat2, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        if (mat1[i] != mat2[i]) {
            return 0; // 不相等
        }
    }
    return 1; // 相等
}

// 比较两个矩阵是否相等（double 类型）
int compare_matrices_double(double *mat1, double *mat2, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        if (fabs(mat1[i] - mat2[i]) > 1e-5) {
            return 0; // 不相等
        }
    }
    return 1; // 相等
}

int main() {
    srand(time(NULL));

    // 测试矩阵大小
    int m = 3, s = 4, n = 5;

    // 生成随机矩阵（int 类型）
    int *x_int = (int *)malloc(m * s * sizeof(int));
    int *y_int = (int *)malloc(s * n * sizeof(int));
    for (int i = 0; i < m * s; i++) x_int[i] = rand() % 10;
    for (int i = 0; i < s * n; i++) y_int[i] = rand() % 10;

    // 生成随机矩阵（double 类型）
    double *x_double = (double *)malloc(m * s * sizeof(double));
    double *y_double = (double *)malloc(s * n * sizeof(double));
    for (int i = 0; i < m * s; i++) x_double[i] = (double)(rand() % 10);
    for (int i = 0; i < s * n; i++) y_double[i] = (double)(rand() % 10);

    // 调用 GPU 矩阵乘法（int 类型）
    int *result_gpu_int = multiple(x_int, y_int, m, s, n);
    // 调用 CPU 矩阵乘法（int 类型）
    int *result_cpu_int = matrix_multiply_cpu_int(x_int, y_int, m, s, n);

    // 调用 GPU 矩阵乘法（double 类型）
    
    double *result_gpu_double = multiple(x_double, y_double, m, s, n);
    
    // 调用 CPU 矩阵乘法（double 类型）
    
    
    double *result_cpu_double = matrix_multiply_cpu_double(x_double, y_double, m, s, n);

    // 比较结果（int 类型）
    if (compare_matrices_int(result_gpu_int, result_cpu_int, m, n)) {
        printf("Matrix multiplication (int) is correct.\n");
    } else {
        printf("Matrix multiplication (int) failed.\n");
    }

    // 比较结果（double 类型）
    if (compare_matrices_double(result_gpu_double, result_cpu_double, m, n)) {
        printf("Matrix multiplication (double) is correct.\n");
    } else {
        printf("Matrix multiplication (double) failed.\n");
    }

    // 释放内存
    // free(x_int);
    // free(y_int);
    // free(result_gpu_int);
    // free(result_cpu_int);
    // free(x_double);
    // free(y_double);
    // free(result_gpu_double);
    // free(result_cpu_double);

    return 0;
}