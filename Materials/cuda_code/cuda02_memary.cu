#include<stdio.h>
#include<cuda.h>

__global__ void add_array(float *a)
{
	int i=threadIdx.x;
	a[i]=a[i]+1;
}

int main()
{
	int N=24;
	int i;

	float *dx=NULL;
	float *hx=NULL;

	int nbyte=N*sizeof(float);

	cudaMalloc((void **)&dx,nbyte);
	
	if (dx==NULL){
		printf("cuda malloc fail!!");
		return -1;
	}
	printf("cuda melloc success!\n");
	
//	hx=(float *)malloc(nbyte);
	cudaMallocHost((void **)&hx,nbyte);
	if (hx==NULL){
		printf("ram melloc fail!\n");
		return -2;
	}
	printf("ram melloc success!!\n");
	
	for(i=0;i<N;i++)
	{
		hx[i]=i;
		printf("%lf ",hx[i]);
	}
	printf("\n");
	
	cudaMemcpy(dx,hx,nbyte,cudaMemcpyHostToDevice);
	add_array<<<i,N>>>(dx);
	cudaDeviceSynchronize();

	cudaMemcpy(hx,dx,nbyte,cudaMemcpyDeviceToHost);
	
	printf("N===%d\n",N);
	for (i=0;i<N;i++)
	{
		printf("%lf ",hx[i]);
		printf(" aaa ");
	}
	printf("\n");
	cudaFree(dx);
//	free(hx);
	cudaFree(hx);


	return 0;
}


