#include<cuda.h>
#include<stdio.h>
#include<sys/time.h>
#include<time.h>
#include<math.h>

double get_time(void)
{
	struct timeval tv;
	double t;

	gettimeofday(&tv,(struct timezone *)0);
	t=tv.tv_sec + (double)tv.tv_usec*1e-6;

	return t;
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
















