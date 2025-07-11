#include<stdio.h>
#include<cuda.h>
#include"mycuda.h"


void reverse_vec(int *x,int n){
	int s=n/2;
	int i,temp;
	for(i=0;i<s;i++){
		temp=x[i];
		x[i]=x[n-i-1];
		x[n-i-1]=temp;
	}
}

__global__ void reverse_vec_kernel(int *x,int n){
	int tid=get_tid();
	int s=n/2;
	if(tid<s){
		int temp=x[tid];
		x[tid]=x[n-1-tid];
		x[n-1-tid]=temp;
	}
}
double reverse_vec_gpu(int *x,int n){
	int bs=256;
	int nbyte=n*sizeof(int);
	
	int s=ceil(sqrt((n+bs-1)/bs));
	dim3 grid=dim3(s,s);

	int *dx=NULL;
	cudaMalloc((void **)&dx,nbyte);
	if(dx==NULL){
		printf("cuda malloc failed!\n");
		return -1;
	}
	cudaMemcpy(dx,x,nbyte,cudaMemcpyHostToDevice);
	
	double dt=get_time();
	reverse_vec_kernel<<<grid,bs>>>(dx,n);
	cudaDeviceSynchronize();
	dt=get_time()-dt;

	cudaMemcpy(x,dx,nbyte,cudaMemcpyDeviceToHost);

	cudaFree(dx);
	return dt;
}

int main(){
	
	int n=16919*16919;
	int nbyte=n*sizeof(int);

	int *hx=NULL;
	cudaMallocHost((void **)&hx,nbyte);
	if (hx==NULL){
		printf("mem allocate failed!");
		return -1;
	}

	init_vec(hx,n);
	printf("the first is %d \n",hx[0]);
	
	
	double ht=get_time();
	reverse_vec(hx,n);
	ht=get_time()-ht;
	printf("after cpu reverse,the first is %d,and time is %lf \n",hx[0],ht);
	
	warmup();	
	double dt=reverse_vec_gpu(hx,n);
	
	printf("after gpu reverse,the first is %d,and time is %lf and speed up %lf\n",hx[0],dt,ht/dt);

	cudaFreeHost(hx);
	return 0;
}





