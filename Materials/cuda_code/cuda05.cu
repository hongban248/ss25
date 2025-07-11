#include<stdio.h>
#include<cuda.h>
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
__global__ void func(float *x,int n){
	int pid=threadIdx.x+blockDim.x*blockIdx.x;
	if(pid<n){
	x[pid]=113/3;
	if (pid%2==1){
		x[pid]++;
	}
	}
}

void init_vec(float *x,int n){
	int i;
	for(i=0;i<n;i++){
		x[i]=1;
	}
}

void show_vec(float *x,int n){
	int i;
	for(i=0;i<n;i++){
		printf("%f ",x[i]);
	}
	printf("\n");
}

int main(){
	printf("size of float is %d\n",(int)sizeof(float));
	int n=262144000;
	int nbyte=n*sizeof(float);

	float *dx=NULL;
	float *hx=NULL;

	cudaMalloc((void **)&dx,nbyte);
	cudaMallocHost((void **)&hx,nbyte);

	if(dx==NULL || hx ==NULL){
		printf("mem allocated failed!i\n");
		return -1;
	}
	printf("mem allocated success!!\n");

	init_vec(hx,n);
	cudaMemcpy(dx,hx,nbyte,cudaMemcpyHostToDevice);
	int bs=256;
	int grid=(n+bs-1)/bs+1;
	
	while(1){	
	double use_time=get_time();
	func<<<grid,bs>>>(dx,n);
	cudaDeviceSynchronize();
	
	cudaMemcpy(hx,dx,nbyte,cudaMemcpyDeviceToHost);
	printf("经过了一个循环,用时%lf\n",get_time()-use_time);
	}
	show_vec(hx,n);
		
	cudaFree(dx);
	cudaFreeHost(hx);
	
	return 0;
}












