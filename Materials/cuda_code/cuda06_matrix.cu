#include<stdio.h>
#include<cuda.h>
#include<math.h>

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

void tran_mat_cpu(int *x,int s){
	int j,i;
	int tem;
	for(i=0;i<s;i++){
		for(j=0;j<i;j++){
			tem=x[i*s+j];
			x[i*s+j]=x[j*s+i];
			x[j*s+i]=tem;
		}
	}
}

__global__ void trans_kernel(int *x,int s){
	int tid=threadIdx.x+blockDim.x*blockIdx.x;
	if (tid<s*s){
		int lin=tid/s;
		int col=tid%s;
		if(col<lin){
			int tem;
			tem=x[lin*s+col];
			x[lin*s+col]=x[col*s+lin];
			x[col*s+lin]=tem;
		}	
	}
}
void trans_mat_gpu(int *x,int s){
	int nbyte=s*s*sizeof(int);

	int *dx=NULL;
	cudaMalloc((void **)&dx,nbyte);
	if (dx==NULL){
		printf("cuda malloc failed!!\n");
		return;
	}
	int bs=256;
	int grid=(s*s+bs)/bs+1;
	cudaMemcpy(dx,x,nbyte,cudaMemcpyHostToDevice);
	trans_kernel<<<grid,bs>>>(dx,s);
	cudaDeviceSynchronize();
	cudaMemcpy(x,dx,nbyte,cudaMemcpyDeviceToHost);
}
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
int main(){
	printf("start !!\n");
	int order=16191;
	int nbyte=order*order*sizeof(int);
	
	int *hx=NULL;
	int *dx=NULL;

	cudaMalloc((void **)&dx,nbyte);
	cudaMallocHost((void **)&hx,nbyte);

	if (hx==NULL || dx==NULL){
		printf("mem allocate failed!\n");
		return -1;}
	printf("mem allocate success!!\n");
	
	printf("row matrix:\n");	
	init_mat(hx,order);
//	show_mat(hx,order);

	//cpu trans
	double ht=get_time();
	tran_mat_cpu(hx,order);
	ht=get_time()-ht;

	printf("transpose 1 time:\n");
//	show_mat(hx,order);
	
	warmup();

	//gpu trans
	double dt=get_time();
	trans_mat_gpu(hx,order);
	dt=get_time()-dt;

	printf("transpose 2 time:\n");
//	show_mat(hx,order);
	printf("cpu use time is:%lf,and gpu cost:%lf,and uptime is %lf \n",ht,dt,ht/dt);

	cudaFree(dx);
	cudaFreeHost(hx);
	

	return 0;
}

