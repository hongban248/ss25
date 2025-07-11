/*
这个文件是mycuda的头文件库。
*/

#ifndef MYCUDA
#define MYCUDA 1

#define USE_UNIX 1

#define get_tid() (blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x) //仅仅对于grid2维，thread1维。
#define get_bid() (blockIdx.x+blockIdx.y*gridDim.x)

#define USE_GCC
#include"mycuda.cu"
#define USE_GCC2
#include"mycuda_d.cu"

double get_time(void); //获取一个时间戳，精度较高
void warmup();  //gpu暖机

//初始化一个s*s的方针matrix。展示方阵。
void init_mat(int *x,int s);
void show_mat(int *x,int s);
void init_mat_zero(int *x,int s);
void init_mat_one(int *x,int s);
int * creat_mat(int s);
//重载
void init_mat(double *x,int s);
void show_mat(double *x,int s);
void init_mat_zero(double *x,int s);
void init_mat_one(double *x,int s);
double * creat_mat_d(int s);


//初始化一个n的一维向量vectory。展示向量。
void init_vec(int *x,int n);
void show_vec(int *x,int n);
void init_vec_zero(int *x,int n);
void init_vec_one(int *x,int n);
int * creat_vec(int s);
//重载
void init_vec(double *x, int n);
void show_vec(double *x, int n);
void init_vec_zero(double *x, int n);
void init_vec_one(double *x, int n);
double *creat_vec_d(int n);

//对于一个一般的向量array：
void init_arr(int *x,int lin,int col);
void show_arr(int *x,int lin,int col);
void init_arr_zero(int *x,int lin,int col);
void init_arr_one(int *x,int lin,int col);
int * creat_arr(int lin,int col);
//重载
void init_arr(double *x, int lin, int col);
void show_arr(double *x, int lin, int col);
void init_arr_zero(double *x, int lin, int col);
void init_arr_one(double *x, int lin, int col);
double *creat_arr_d(int lin, int col);

//我需要实现什么功能？？
//对于单个向量，可以实现求和(实现了），求平均值(实现了）。对于方阵，有一个det(实现了，但是是串行)和求逆。当然还有一个转置。
//对于2个向量，可以实现向量点乘，叉乘。如果要相加的话，考虑一下广播机制？

//简单求个和
int sum_mat(int *x,int s);
int sum_vec(int *x,int n);
int sum_arr(int *x,int lin,int col);
//重载
__global__ void sum_mat_knl(double *x, int n, double *outcome);
double sum_mat(double *x, int s);
double sum_vec(double *x, int n);
double sum_arr(double *x, int lin, int col);

//求平均值aver：
double aver_arr(int *x,int lin,int col);
double aver_mat(int *x,int s);
double aver_vec(int *x,int n);
//重载：
double aver_arr(double *x,int lin,int col);
double aver_mat(double *x,int s);
double aver_vec(double *x,int n);

//方阵求det
extern __host__ void get_sub_mat(int *x,int s,int *sub_mat,int lin,int col);
extern __host__ void get_sub_mat(double *x,int s,double *sub_mat,int lin,int col);
extern __host__ int det_mat(int *x,int s);
extern __host__ double det_mat(double *x,int s);

//方阵转置
int * inverse(int *x,int lin,int col);
double * inverse(double *x,int lin,int col);

//向量和常数相加：
//向量和向量矩阵相乘：
int * multiple(int *x,int *y ,int m,int s,int n);
double * multiple(double *x,double *y, int m,int s,int n);

//逆矩阵
double *reverse(int *x, int s);
double *reverse(double *x, int s);

#endif