#include<cuda.h>
#include<stdio.h>
#include<sys/time.h>
#include<time.h>
#include<math.h>

// mycuda.cu
#ifndef USE_GCC
#ifdef __cplusplus
extern "C" {
#endif

extern "C" void init_arr(int *x, int lin,int col);
extern "C" void init_arr_zero(int *x, int lin,int col);
extern "C" void init_arr_one(int *x, int lin,int col);
extern "C" int sum_arr(int *x, int lin,int col);
extern "C" double aver_arr(int *x,int lin,int col);
extern "C" __host__ int det_mat(int *x,int s);
extern "C" int * multiple(int *x,int *y ,int m,int s,int n);
extern "C" int * inverse(int *x,int lin,int col);

#ifdef __cplusplus
}
#endif
#endif

double get_time(void)
{
	struct timeval tv;
	double t;

	gettimeofday(&tv,(struct timezone *)0);
	t=tv.tv_sec + (double)tv.tv_usec*1e-6;

	return t;
}

__global__ void warmup_knl(int n)
{
	int i,j;
	i=1;
	j=2;
	i=i+j;
	if(threadIdx.x==1 && n==0){
		printf("成功warm up!!\n");
	}
}
void warmup()
{
	int i;
	for (i=0;i<8;i++){
		warmup_knl<<<1,256>>>(i);
	}
	cudaDeviceSynchronize();
}


//基本操作
void init_mat_zero(int *x,int s){
	int i;
	for(i=0;i<s*s;i++){
		x[i]=0;
	}
}
void init_mat_one(int *x,int s){
	int i;
	for(i=0;i<s*s;i++){
		x[i]=1;
	}
}
void init_mat(int *x,int s){
	int i;
	for(i=0;i<s*s;i++){
		x[i]=i;
	}
}
void show_mat(int *x,int s){
	int i,j;
	for(i=0;i<s;i++){
		for(j=0;j<s;j++){
			printf("%d ",x[i*s+j]);
		}
		printf("\n");
	}
}
int * creat_mat(int s){
	int nbyte=s*s*sizeof(int);
	int *x=NULL;
	cudaMallocHost((void **)&x,nbyte);
	if(x==NULL){
		printf("RAM allocate failed!\n");
		return NULL;
	}
	return x;
}


void init_vec(int *x,int n){
	int i;
	for(i=0;i<n;i++){
		x[i]=i;
	}
}
void init_vec_zero(int *x,int n){
	int i;
	for(i=0;i<n;i++){
		x[i]=0;
	}
}
void init_vec_one(int *x,int n){
	int i;
	for(i=0;i<n;i++){
		x[i]=1;
	}
}
void show_vec(int *x,int n){
	int i;
	for(i=0;i<n;i++){
		printf("%d ",x[i]);
	}
	printf("\n");
}
int * creat_vec(int s){
	int nbyte=s*sizeof(int);
	int *x=NULL;
	cudaMallocHost((void **)&x,nbyte);
	if(x==NULL){
		printf("RAM allocated fail!!\n");
		return NULL;
	}
	return x;
}

void init_arr(int *x, int lin, int col) {
    for (int i = 0; i < lin; i++) {
        for (int j = 0; j < col; j++) {
            x[i * col + j] = i * col + j ; // 按行优先顺序初始化
        }
    }
}
void show_arr(int *x, int lin, int col) {
    for (int i = 0; i < lin; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", x[i * col + j]);
        }
        printf("\n");
    }
}
void init_arr_zero(int *x, int lin, int col) {
    for (int i = 0; i < lin * col; i++) {
        x[i] = 0;
    }
}
void init_arr_one(int *x, int lin, int col) {
    for (int i = 0; i < lin * col; i++) {
        x[i] = 1;
    }
}
int *creat_arr(int lin, int col) {
    int nbyte = lin * col * sizeof(int); // 计算数组占用的字节数
    int *x = NULL; // 定义一个指向整数的指针

    // 分配页锁定内存
    cudaError_t err = cudaMallocHost((void **)&x, nbyte);
    if (err != cudaSuccess) {
        printf("RAM allocate failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return x; // 返回分配的内存指针
}

//求和
__global__ void sum_mat_knl(int *x,int n,int *outcome){
	

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("temp here is %d and %d\n",*outcome,x[tid]);
	extern __shared__ int sdata[];
	if(tid<n){
		sdata[threadIdx.x]=x[tid];
	}
	else{
		sdata[threadIdx.x]=0;
	}
	__syncthreads();

	int i;
	for(i=blockDim.x/2;i>0;i=i>>1){
		if (threadIdx.x < i){
		sdata[threadIdx.x]=sdata[threadIdx.x]+sdata[threadIdx.x+i];
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		atomicAdd(outcome, sdata[0]);
		//printf("temp here is %d and %d\n",*outcome,sdata[0]);
	}
	
	// if(tid==0){
	// 	printf("now this grid comtribute and the temp outcome is %d\n",*outcome);
	// }
}
int sum_mat(int *x,int s){
	int bs=256;
	int nbyte=s*s*sizeof(int);
	int numElements = s * s;
        int blocks = (numElements + bs - 1) / bs;
	
        dim3 grid(blocks, 1); // 正确的一维网格

	int *dx=NULL;
	cudaMalloc((void **)&dx,nbyte);

	int *outcome=NULL;
	int *doutcome=NULL;
	cudaMalloc((void **)&doutcome,sizeof(int));
	cudaMallocHost((void **)&outcome,sizeof(int));
	if (dx==NULL || doutcome==NULL || outcome==NULL){
		printf("cuda allocate failed!\n");
		return -1;
	}
	*outcome=0;

	cudaMemcpy(dx,x,nbyte,cudaMemcpyHostToDevice);
	cudaMemcpy(doutcome,outcome,sizeof(int),cudaMemcpyHostToDevice);

	sum_mat_knl<<<grid,bs,bs*sizeof(int)>>>(dx,s*s,doutcome);
	cudaDeviceSynchronize();

	cudaMemcpy(outcome,doutcome,sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dx);
	cudaFree(doutcome);
	
	return *outcome;
//	return 114;
}
int sum_vec(int *x,int n){
	int bs=256;
	int nbyte=n*sizeof(int);
	int numElements = n;
        int blocks = (numElements + bs - 1) / bs;
	
        dim3 grid(blocks, 1); // 正确的一维网格

	int *dx=NULL;
	cudaMalloc((void **)&dx,nbyte);

	int *outcome=NULL;
	int *doutcome=NULL;
	cudaMalloc((void **)&doutcome,sizeof(int));
	cudaMallocHost((void **)&outcome,sizeof(int));
	if (dx==NULL || doutcome==NULL || outcome==NULL){
		printf("cuda allocate failed!\n");
		return -1;
	}
	*outcome=0;

	cudaMemcpy(dx,x,nbyte,cudaMemcpyHostToDevice);
	cudaMemcpy(doutcome,outcome,sizeof(int),cudaMemcpyHostToDevice);

	sum_mat_knl<<<grid,bs,bs*sizeof(int)>>>(dx,n,doutcome);
	cudaDeviceSynchronize();

	cudaMemcpy(outcome,doutcome,sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dx);
	cudaFree(doutcome);
	return *outcome;
}
int sum_arr(int *x,int lin,int col){
	int bs=256;
	int nbyte=lin*col*sizeof(int);
	int numElements = lin * col;
        int blocks = (numElements + bs - 1) / bs;
	
        dim3 grid(blocks, 1); // 正确的一维网格

	int *dx=NULL;
	cudaMalloc((void **)&dx,nbyte);

	int *outcome=NULL;
	int *doutcome=NULL;
	cudaMalloc((void **)&doutcome,sizeof(int));
	cudaMallocHost((void **)&outcome,sizeof(int));
	if (dx==NULL || doutcome==NULL || outcome==NULL){
		printf("cuda allocate failed!\n");
		return -1;
	}
	*outcome=0;

	cudaMemcpy(dx,x,nbyte,cudaMemcpyHostToDevice);
	cudaMemcpy(doutcome,outcome,sizeof(int),cudaMemcpyHostToDevice);

	sum_mat_knl<<<grid,bs,bs*sizeof(int)>>>(dx,lin*col,doutcome);
	cudaDeviceSynchronize();

	cudaMemcpy(outcome,doutcome,sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dx);
	cudaFree(doutcome);
	return *outcome;
}

//求平均值
double aver_arr(int *x,int lin,int col){
	return (double)sum_arr(x,lin,col)/(lin*col);
}
double aver_mat(int *x,int s){
	return aver_arr(x,s,s);
}
double aver_vec(int *x ,int n){
	return aver_arr(x,n,1);
}

//方阵求det
extern __host__ void get_sub_mat(int *x,int s,int *sub_mat,int lin,int col){
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
extern __host__ int det_mat(int *x,int s){
	if(s==1){
		return x[0];
	}
	if(s==2){
		return x[0]*x[3]-x[1]*x[2];
	}
	int outcome=0;
	int *sub_mat=NULL;
	cudaMallocHost((void **)&sub_mat,sizeof(int)*(s-1)*(s-1));
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

//求乘
__global__ void multiple_knl(int *x,int *y,int *o,int m,int s,int n){
	int size=m*n;
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<size){
	//	printf("now tid is %d\n",tid);
		o[tid]=0;
		int lin=tid/n;
		int col=tid%n;

		int i;
		for(i=0;i<s;i++){
		//	printf("here %d and %d\n",lin*s+i,i*n+col);
			o[tid]=o[tid]+x[lin*s+i]*y[i*n+col];
		}
	//	printf("tid is %d and result here is %d\n",tid,o[tid]);
	}
	//printf("tid is %d \n",tid);

}
int * multiple(int *x,int *y ,int m,int s,int n){
	// printf("start!!and %d,%d,%d\n",m,s,n);
	// show_arr(y,m,s);
	int bs=256;
	int nbyte_x=m*s*sizeof(int);
	int nbyte_y=s*n*sizeof(int);
	int nbyte_o=m*n*sizeof(int);

//	int size_x=m*s;
//	int size_y=s*n;
	int size_o=m*n;

	int grid=(size_o+bs-1)/bs;

	int *x_d=NULL;
	int *y_d=NULL;
	int *o_d=NULL;
	int *o_h=NULL;
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
//	printf("success malloc!\n");
	
	cudaMemcpy(x_d,x,nbyte_x,cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,y,nbyte_y,cudaMemcpyHostToDevice);


	multiple_knl<<<grid,bs>>>(x_d,y_d,o_d,m,s,n);
	
	

	cudaDeviceSynchronize();

	
	
	cudaMemcpy(o_h,o_d,nbyte_o,cudaMemcpyDeviceToHost);
	//printf("1111\n");
	//show_arr(o_d,m,n);
	//printf("2222\n");

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(o_d);
	return o_h;
}

//转置
__global__ void inverse_knl(int *x,int lin,int col){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<lin*col){
		int x1=tid/col;
		int y=tid%col;
		int temp=x[tid];
		__syncthreads();
		x[y*lin+x1]=temp;
	}
}
int * inverse(int *x,int lin,int col){
	int bs=256;
	int nbyte=lin*col*sizeof(int);
	int size=lin*col;
	int grid=(size+bs-1)/bs;

	int *dx=NULL;
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

//逆矩阵
