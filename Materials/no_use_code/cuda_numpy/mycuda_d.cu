#include<cuda.h>
#include<stdio.h>
#include<sys/time.h>
#include<time.h>
#include<math.h>

// mycuda_d.cu
#ifndef USE_GCC
#ifdef __cplusplus
extern "C" {
#endif



extern "C" void init_arr(double *x,int lin,int col);
extern "C" void init_arr_zero(double *x,int lin,int col);
extern "C" void init_arr_one(double *x,int lin,int col);
extern "C" double sum_arr(double *x, int lin, int col);
extern "C" double aver_arr(double *x,int lin,int col);
extern "C" __host__ double det_mat(double *x,int s);
extern "C" double * multiple(double *x,double *y, int m,int s,int n);
double * inverse(double *x,int lin,int col);

#ifdef __cplusplus
}
#endif

#endif

__device__ double atomicAdd_d(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

//基本操作
void init_mat(double *x, int s) {
    for (int i = 0; i < s * s; i++) {
        x[i] = i; // 
    }
}
void show_mat(double *x, int s) {
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            printf("%.2f ", x[i * s + j]);
        }
        printf("\n");
    }
}
void init_mat_zero(double *x, int s) {
    for (int i = 0; i < s * s; i++) {
        x[i] = 0.0;
    }
}
double *creat_mat_d(int s) {
    int nbyte=s*s*sizeof(double);
	double *x=NULL;
	cudaMallocHost((void **)&x,nbyte);
	if(x==NULL){
		printf("RAM allocate failed!\n");
		return NULL;
	}
	return x;
}
void init_mat_one(double *x, int s) {
    for (int i = 0; i < s * s; i++) {
        x[i] = 1.0;
    }
}

void init_vec(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = i; 
    }
}
void show_vec(double *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.2f ", x[i]);
    }
    printf("\n");
}
void init_vec_zero(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }
}
void init_vec_one(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
    }
}
double *creat_vec_d(int n) {
    int nbyte=n*sizeof(double);
	double *x=NULL;
	cudaMallocHost((void **)&x,nbyte);
	if(x==NULL){
		printf("RAM allocated fail!!\n");
		return NULL;
	}
	return x;
}

void init_arr(double *x, int lin, int col) {
    for (int i = 0; i < lin * col; i++) {
        x[i] = i;
    }
}
void show_arr(double *x, int lin, int col) {
    for (int i = 0; i < lin; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.2f ", x[i * col + j]);
        }
        printf("\n");
    }
}
void init_arr_zero(double *x, int lin, int col) {
    for (int i = 0; i < lin * col; i++) {
        x[i] = 0.0;
    }
}
void init_arr_one(double *x, int lin, int col) {
    for (int i = 0; i < lin * col; i++) {
        x[i] = 1.0;
    }
}
double *creat_arr_d(int lin, int col) {
    int nbyte = lin * col * sizeof(double); // 计算数组占用的字节数
    double *x = NULL; // 定义一个指向整数的指针

    // 分配页锁定内存
    cudaError_t err = cudaMallocHost((void **)&x, nbyte);
    if (err != cudaSuccess) {
        printf("RAM allocate failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return x; // 返回分配的内存指针
}

//求和
__global__ void sum_mat_knl(double *x, int n, double *outcome) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double sdata_d[];
    if (tid < n) {
        sdata_d[threadIdx.x] = x[tid];
    } else {
        sdata_d[threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata_d[threadIdx.x] += sdata_d[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd_d(outcome, sdata_d[0]);
      //  printf("this contirbute %lf and result is %lf\n",sdata_d[0],*outcome);
    }
}
double sum_mat(double *x, int s) {
    int bs = 256;
    int nbyte = s * s * sizeof(double);
    int numElements = s * s;
    int blocks = (numElements + bs - 1) / bs;

    dim3 grid(blocks, 1);

    double *dx = NULL;
    cudaMalloc((void **)&dx, nbyte);

    double *outcome = NULL;
    double *doutcome = NULL;
    cudaMalloc((void **)&doutcome, sizeof(double));
    cudaMallocHost((void **)&outcome, sizeof(double));
    if (dx == NULL || doutcome == NULL || outcome == NULL) {
        printf("cuda allocate failed!\n");
        return -1;
    }
    *outcome = 0.0;

    cudaMemcpy(dx, x, nbyte, cudaMemcpyHostToDevice);
    cudaMemcpy(doutcome, outcome, sizeof(double), cudaMemcpyHostToDevice);

    sum_mat_knl<<<grid, bs, bs * sizeof(double)>>>(dx, s * s, doutcome);
    cudaDeviceSynchronize();

    cudaMemcpy(outcome, doutcome, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(doutcome);
    return *outcome;
}
double sum_vec(double *x, int n) {
    int bs = 256;
    int nbyte = n * sizeof(double);
    int numElements = n;
    int blocks = (numElements + bs - 1) / bs;

    dim3 grid(blocks, 1);

    double *dx = NULL;
    cudaMalloc((void **)&dx, nbyte);

    double *outcome = NULL;
    double *doutcome = NULL;
    cudaMalloc((void **)&doutcome, sizeof(double));
    cudaMallocHost((void **)&outcome, sizeof(double));
    if (dx == NULL || doutcome == NULL || outcome == NULL) {
        printf("cuda allocate failed!\n");
        return -1;
    }
    *outcome = 0.0;

    cudaMemcpy(dx, x, nbyte, cudaMemcpyHostToDevice);
    cudaMemcpy(doutcome, outcome, sizeof(double), cudaMemcpyHostToDevice);

    sum_mat_knl<<<grid, bs, bs * sizeof(double)>>>(dx, n, doutcome);
    cudaDeviceSynchronize();

    cudaMemcpy(outcome, doutcome, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(doutcome);
    return *outcome;
}
double sum_arr(double *x, int lin, int col) {
    int bs = 256;
    int nbyte = lin * col * sizeof(double);
    int numElements = lin * col;
    int blocks = (numElements + bs - 1) / bs;

    dim3 grid(blocks, 1);

    double *dx = NULL;
    cudaMalloc((void **)&dx, nbyte);

    double *outcome = NULL;
    double *doutcome = NULL;
    cudaMalloc((void **)&doutcome, sizeof(double));
    cudaMallocHost((void **)&outcome, sizeof(double));
    if (dx == NULL || doutcome == NULL || outcome == NULL) {
        printf("cuda allocate failed!\n");
        return -1;
    }
    *outcome = 0.0;

    cudaMemcpy(dx, x, nbyte, cudaMemcpyHostToDevice);
    cudaMemcpy(doutcome, outcome, sizeof(double), cudaMemcpyHostToDevice);

    sum_mat_knl<<<grid, bs, bs * sizeof(double)>>>(dx, lin * col, doutcome);
    cudaDeviceSynchronize();

    cudaMemcpy(outcome, doutcome, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(doutcome);
    return *outcome;
}

//求平均值
double aver_arr(double *x,int lin,int col){
	return sum_arr(x,lin,col)/(lin*col);
}
double aver_mat(double *x,int s){
	return aver_arr(x,s,s);
}
double aver_vec(double *x ,int n){
	return aver_arr(x,n,1);
}

//求det
extern __host__ void get_sub_mat(double *x,int s,double *sub_mat,int lin,int col){
	int i,j;
	int index=0;
	for(i=0;i<s;i++){
		if(i==lin) continue;
		for(j=0;j<s;j++){
			if(j==col) continue;
			sub_mat[index]=x[i*s+j];
			index++;
		}
	}
}
extern __host__ double det_mat(double *x,int s){
	if(s==1){
		return x[0];
	}
	if(s==2){
		return x[0]*x[3]-x[1]*x[2];
	}
	double outcome=0;
	double *sub_mat=NULL;
	cudaMallocHost((void **)&sub_mat,sizeof(double)*(s-1)*(s-1));
	if(sub_mat==NULL){
		printf("mem allocated failed!\n");
		return -1;
	}
	int i;
	for(i=0;i<s;i++){
		get_sub_mat(x,s,sub_mat,0,i);
		outcome+= x[i]*det_mat(sub_mat,s-1)* ((i % 2 == 0) ? 1 : -1);
	}
	cudaFreeHost(sub_mat);
	return outcome;
}

//求矩阵乘法
__global__ void multiple_knl(double *x,double *y,double *o,int m,int s,int n){
	int size=m*n;
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<size){
		o[tid]=0.0;
		int lin=tid/n;
		int col=tid%n;

		int i;
		for(i=0;i<s;i++){
			o[tid]=o[tid]+x[lin*s+i]*y[i*n+col];
		}
	}

}
double * multiple(double *x,double *y ,int m,int s,int n){
	int bs=256;
	int nbyte_x=m*s*sizeof(double);
	int nbyte_y=s*n*sizeof(double);
	int nbyte_o=m*n*sizeof(double);

//	int size_x=m*s;
//	int size_y=s*n;
	int size_o=m*n;

	int grid=(size_o+bs-1)/bs;

	double *x_d=NULL;
	double *y_d=NULL;
	double *o_d=NULL;
	double *o_h=NULL;
	cudaMalloc((void **)&x_d,nbyte_x);
	cudaMalloc((void **)&y_d,nbyte_y);
	cudaMalloc((void **)&o_d,nbyte_o);
	cudaMallocHost((void **)&o_h,nbyte_o);

	if (!x_d || !y_d || !o_d || !o_h) {
        printf("Memory allocation failed!\n");
        if (x_d) cudaFree(x_d);
        if (y_d) cudaFree(y_d);
        if (o_d) cudaFree(o_d);
        if (o_h) cudaFreeHost(o_h);
        return NULL;
    }
	
	cudaMemcpy(x_d,x,nbyte_x,cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,y,nbyte_y,cudaMemcpyHostToDevice);

	multiple_knl<<<grid,bs>>>(x_d,y_d,o_d,m,s,n);
	cudaDeviceSynchronize();

	cudaMemcpy(o_h,o_d,nbyte_o,cudaMemcpyDeviceToHost);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(o_d);
	return o_h;
}

//求逆
__global__ void inverse_knl(double *x,int lin,int col){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<lin*col){
		int x1=tid/col;
		int y=tid%col;
		double temp=x[tid];
		__syncthreads();
		x[y*lin+x1]=temp;
	}
}
double * inverse(double *x,int lin,int col){
	int bs=256;
	int nbyte=lin*col*sizeof(double);
	int size=lin*col;
	int grid=(size+bs-1)/bs;

	double *dx=NULL;
	cudaMalloc((void **)&dx,nbyte);
	if(dx==NULL){
		printf("cuda allocate failed!\n");
		return NULL;
	}

	cudaMemcpy(dx,x,nbyte,cudaMemcpyHostToDevice);

	inverse_knl<<<grid,bs>>>(dx,lin,col);
	cudaDeviceSynchronize();

	cudaMemcpy(x,dx,nbyte,cudaMemcpyDeviceToHost);

	cudaFree(dx);
	return x;

}












