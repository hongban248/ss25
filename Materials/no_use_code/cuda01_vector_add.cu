#include<stdio.h>
#include<cuda.h>

typedef float FLOAT;
#define USE_UNIX 1

#define get_tid() (blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x)
#define get_bid() (blockIdx.x+blockIdx.y*gridDim.x)

void warmup();

double get_time(void);

void vec_add_host(FLOAT *x,FLOAT *y, FLOAT *z,int n);

__global__ void vec_add(FLOAT *x,FLOAT *y ,FLOAT*z ,int n){

	int idx=get_tid();
	if (idx<n) z[idx]=x[idx]+y[idx]+z[idx];
}

void vec_add_host(FLOAT *x,FLOAT *y,FLOAT *z,int n){
	int i;
	for (i=0;i<n;i++) z[i]=x[i]+y[i]+z[i];

}

#if USE_UNIX
#include<sys/time.h>
#include<time.h>

double get_time(void)
{
	struct timeval tv;
	double t;

	gettimeofday(&tv,(struct timezone *)0);
	t=tv.tv_sec + (double)tv.tv_usec*1e-6;

	return t;
}
#else
#include<windows.h>
double get_time(void) {
    LARGE_INTEGER frequency;  // 用于存储计时器的频率
    LARGE_INTEGER start, end; // 用于存储计时器的开始和结束值

    // 获取计时器的频率
    QueryPerformanceFrequency(&frequency);

    // 获取开始时间
    QueryPerformanceCounter(&start);

    // 在这里执行需要计时的代码
    // 例如：Sleep(1000); // 模拟耗时操作

    // 获取结束时间
    QueryPerformanceCounter(&end);

    // 计算经过的时间（单位：秒）
    double elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    return elapsed_time;
}

#endif
__global__ void warmup_knl()
{
	int i,j;
	i=1;
	j=2;
	i=i+j;
}
void warmup()
{
	int i;
	for (i=0;i<8;i++){
		warmup_knl<<<1,256>>>();
	}
}




int main()
{
	printf("hello_from cpu\n");
	int N=20000000;
	int nbytes=N*sizeof(FLOAT);
	int bs=256;
	
	int s=ceil(sqrt((N+bs-1.) /bs ));
	dim3 grid=dim3(s,s);

	FLOAT *dx=NULL,*hx=NULL;
	FLOAT *dy=NULL,*hy=NULL;
	FLOAT *dz=NULL,*hz=NULL;

	int itr=30;
	int i;
	double th,td;

	warmup();

	cudaMalloc((void **)&dx,nbytes);
	cudaMalloc((void **)&dy,nbytes);
        cudaMalloc((void **)&dz,nbytes);
	
	if(dx==NULL || dy==NULL || dz==NULL){
		printf("can't cuda beging!!!\n");
		return -1;
	}
	printf("Cuda Consinder beautiful\n");
	printf("allocated %.2f MB on GPU\n",nbytes/(1024.f *1024.f));
	hx=(FLOAT *)malloc(nbytes);
	hy=(FLOAT *)malloc(nbytes);
	hz=(FLOAT *)malloc(nbytes);
	
	if (hx==NULL || hy==NULL || hz==NULL)
	{
		printf("CPU Could Broken!!\n");
		return -2;
	}

	printf("allocated %.2f MB on CPU\n",nbytes/(1024.f *1024.f));

	for (i=0;i<N;i++){
		hx[i]=1;
		hy[i]=1;
		hz[i]=1;
	}

	cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
	cudaMemcpy(dy,hy,nbytes,cudaMemcpyHostToDevice);
	cudaMemcpy(dz,hz,nbytes,cudaMemcpyHostToDevice);
	
	warmup();

	cudaDeviceSynchronize();
	td=get_time();

	for (i=0;i<itr;i++) vec_add<<<grid,bs>>>(dx,dy,dz,N);
	cudaDeviceSynchronize();
	
	td=get_time()-td;
	printf("compute cost time:%e\n",td);

	th=get_time();
	for (i=0;i<itr;i++) vec_add_host(hx,hy,hz,N);
	th=get_time()-th;
	printf("CPU time is %e and speendup %g\n",th,th/td);

	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);

	free(hx);
	free(hy);
	free(hz);

	return 0;
}
