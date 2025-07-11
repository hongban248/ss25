#include<stdio.h>
#include<cuda.h>

void init_vec(int *x,int n){
	int i;
	for(i=0;i<n;i++){
		x[i]=i;
	}
}

void show_vec(int *x,int n){
	int i;
	for(i=0;i<n;i++){
		printf("%d ",x[i]);
	}
	printf("\n");
}

__global__ void add_vec(int *x,int *y,int n){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i<n){
		x[i]=x[i]+y[i];
	}
}

__global__ void sum_vec(int *x,int *result,int n){

	int tid=threadIdx.x;
	int i=tid+blockDim.x*blockIdx.x;

	extern __shared__ int sdata[];
	
	if(i<n){
		sdata[tid]=x[i];
	}
	else{
		sdata[tid]=0;
	}
	__syncthreads();

	int s=0;
	for (s=blockDim.x/2;s>0;s=s>>1){
	    if(tid<s){
		sdata[tid]=sdata[tid]+sdata[tid+s];
		__syncthreads();
		}}
	if (tid==0)
	{
		atomicAdd(result, sdata[0]);
		printf("this block is %d,and now reslut is %d\n",sdata[0],*result);
	}



}
int main(){
	long long int n=1000;
	
	int nbyte=n*sizeof(int);

	int *dx=NULL;
	int *dy=NULL;
	int *hx=NULL;
	int *hy=NULL;

	cudaMalloc((void **)&dx,nbyte);
	cudaMalloc((void **)&dy,nbyte);
	cudaMallocHost((void **)&hx,nbyte);
	cudaMallocHost((void **)&hy,nbyte);
	
	if (dx==NULL || dy==NULL || hx==NULL || hy==NULL){
	       printf("Cuda or ram alllocate failed!\n");
		return -1;
	}
	printf("ram and cuda allocated success!\n");

	int bs=256;
	int grid= n/256 +1;
	
	init_vec(hx,n);
	init_vec(hy,n);
//	show_vec(hx,n);

	cudaMemcpy(dx,hx,nbyte,cudaMemcpyHostToDevice);
	cudaMemcpy(dy,hy,nbyte,cudaMemcpyHostToDevice);
	add_vec<<<grid,bs>>>(dx,dy,n);
	cudaDeviceSynchronize();

	cudaMemcpy(hx,dx,nbyte,cudaMemcpyDeviceToHost);
//	show_vec(hx,n);

	int *dresult=NULL;
	int *hresult=NULL;

	cudaMalloc((void **)&dresult, sizeof(int));
    cudaMallocHost((void **)&hresult, sizeof(int));

    if (dresult == NULL || hresult == NULL) {
        printf("outcome malloc failed!\n");
        return -1;
    }
    printf("outcome malloc success!\n");

    // 初始化全局结果变量为0
    int init_value = 0;
    cudaMemcpy(dresult, &init_value, sizeof(int), cudaMemcpyHostToDevice);

    sum_vec<<<grid, bs, bs * sizeof(int)>>>(dx, dresult, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hresult, dresult, sizeof(int), cudaMemcpyDeviceToHost);
    printf("outcome is %d\n", *hresult);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dresult);
    cudaFreeHost(hx);
    cudaFreeHost(hy);
    cudaFreeHost(hresult);

	return 0;
}	









