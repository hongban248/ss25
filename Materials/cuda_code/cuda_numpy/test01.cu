#include"mycuda.h"


int main(){
	printf("begin test!\n");
	
	
	int *arr=creat_arr(2,2);
	init_arr(arr,2,2);
	show_arr(arr,2,2);
	double *r= reverse(arr,2);
	show_arr(r,2,2);
	

	



    return 0;
}
