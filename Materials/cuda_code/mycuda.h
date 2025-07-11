/*
这个文件是mycuda的头文件库。
*/

#ifndef MYCUDA
#define MYCUDA 1

#define USE_UNIX 1

#define get_tid() (blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x) //仅仅对于grid2维，thread1维。
#define get_bid() (blockIdx.x+blockIdx.y*gridDim.x)

#include"mycuda.cu"
double get_time(void); //获取一个时间戳，精度较高
void warmup();  //gpu暖机

//初始化一个s*s的方针matrix。展示方阵。
void init_mat(int *x,int s);
void show_mat(int *x,int s);
void init_mat_zero(int *x,int s);
void init_mat_one(int *x,int s);
int * creat_mat(int s);

//初始化一个n的一维向量vectory。展示向量。
void init_vec(int *x,int n);
void show_vec(int *x,int n);
void init_vec_zero(int *x,int n);
void init_vec_one(int *x,int n);
int * creat_vec(int s);

//对于一个一般的方针array：
void init_arr(int *x,int lin,int col);
void show_arr(int *x,int lin,int col);
void init_arr_zero(int *x,int lin,int col);
void init_arr_one(int *x,int lin,int col);
int * creat_arr(int lin,int col);

//我需要实现什么功能？？

#endif