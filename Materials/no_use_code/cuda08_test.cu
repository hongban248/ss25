#include"mycuda.h"

int main(){
	printf("strat process!!\n");
	printf("time: %lf\n",get_time());
	warmup();
	
	int *hx=creat_mat(3);
	init_mat_one(hx,3);
	show_mat(hx,3);
	
	int *hy=creat_vec(4);
	init_vec_one(hy,4);
	show_vec(hy,4);
	
	int *hz=creat_arr(2,3);
	init_arr_one(hz,2,3);
	show_arr(hz,2,3);

	cudaFreeHost(hx);
	cudaFreeHost(hy);
	cudaFreeHost(hy);
	return 0;
}

