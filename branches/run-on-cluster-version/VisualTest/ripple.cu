/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation. 
* Any use, reproduction, disclosure, or distribution of this software 
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA) 
* associated with this source code for terms and conditions that govern 
* your use of this NVIDIA software.
* 
*/


#include "common/book.h"
#include "common/gpu_anim.h"

#include "cutil_inline.h"
#include "curand_kernel.h"
#include "curand.h"

#define DIM 256
#define R	2

#define VEGE	1
#define FIRE	2
#define	BURNT	3
#define	NULL	0

#define PROB_B	0.1
#define PROB_D	0.8
#define PROB_I	0.8

__global__ void kernel( uchar4 *ptr, char* new_state, const char* old_state ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	char live_count = 0;
	if(x!=0)
		live_count+=old_state[DIM*y+x-1]; //west
	if(x!=DIM-1)
		live_count+=old_state[DIM*y+x+1]; //east
	if(y!=0)
		live_count+=old_state[DIM*(y-1)+x]; //north
	if(y!=DIM-1)
		live_count+=old_state[DIM*(y+1)+x]; //south
	if(x!=0&&y!=0)
		live_count+=old_state[DIM*(y-1)+x-1]; //northwest
	if(x!=0&&y!=DIM-1)
		live_count+=old_state[DIM*(y+1)+x-1]; //southwest
	if(x!=DIM-1&&y!=0)
		live_count+=old_state[DIM*(y-1)+x+1]; //northeast
	if(x!=DIM-1&&y!=DIM-1)
		live_count+=old_state[DIM*(y+1)+x+1]; //southeast

	if(offset<DIM*DIM){
		new_state[offset]=old_state[offset];
		if(live_count==3){
			new_state[offset]=1;
			ptr[offset].x = 255;
			ptr[offset].y = 255;
			ptr[offset].z = 255;
			ptr[offset].w = 255;
		}

		if(live_count!=2&&live_count!=3){
			new_state[offset]=0;
			ptr[offset].x = 0;
			ptr[offset].y = 0;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		}
	}
}

__global__ void kernel2( uchar4 *ptr, 
	char* new_state, const char* old_state ) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	char live_count = 0;
	if(x!=0 && old_state[DIM*y+x-1]==FIRE)
		live_count++; //west
	if(x!=DIM-1 && old_state[DIM*y+x+1]==FIRE)
		live_count++; //east
	if(y!=0 && old_state[DIM*(y-1)+x]==FIRE)
		live_count++; //north
	if(y!=DIM-1 && old_state[DIM*(y+1)+x]==FIRE)
		live_count++; //south
	if(x!=0 && y!=0 && old_state[DIM*(y-1)+x-1]==FIRE)
		live_count++; //northwest
	if(x!=0 && y!=DIM-1 && old_state[DIM*(y+1)+x-1]==FIRE)
		live_count++; //southwest
	if(x!=DIM-1 && y!=0 && old_state[DIM*(y-1)+x+1]==FIRE)
		live_count++; //northeast
	if(x!=DIM-1 && y!=DIM-1 && old_state[DIM*(y+1)+x+1]==FIRE)
		live_count++; //southeast
	
	curandState localState;
	curand_init(1234,offset,0,&localState);
	if(offset<DIM*DIM){
		double xf = curand_uniform(&localState);
		new_state[offset]=old_state[offset];	
		if(new_state[offset]==VEGE && live_count>0 && xf<PROB_I)
			new_state[offset] = FIRE;
		if(new_state[offset]==FIRE && xf<PROB_B)
			new_state[offset] = BURNT;
		if(new_state[offset]==VEGE){
			ptr[offset].x = 0;
			ptr[offset].y = 255;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		}
		else if(new_state[offset]==FIRE){
			new_state[offset]=FIRE;
			ptr[offset].x = 255;
			ptr[offset].y = 0;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		}
		else if(new_state[offset]==NULL){
			ptr[offset].x = 0;
			ptr[offset].y = 0;
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		} else if(new_state[offset]==BURNT){
			ptr[offset].x = 0;
			ptr[offset].y = 0;
			ptr[offset].z = 255;
			ptr[offset].w = 255;
		}
	}
}

void generate_frame( uchar4 *pixels, void*, char* new_state, char* old_state ) {
	dim3    grids(DIM/16,DIM/16);
	dim3    threads(16,16);
	kernel2<<<grids,threads>>>( pixels, new_state, old_state );
}

__global__ void init_state(char* state1, char* state2){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x+y*blockDim.x*gridDim.x;
	state1[offset]=NULL;
	state2[offset]=NULL;

	float xf;
	curandState localState;
	curand_init(2345,offset,0,&localState);
	int nnodes=DIM*DIM;
	if(offset<nnodes){
		xf = curand_uniform(&localState);
		if(xf<PROB_D){
			state1[offset]=VEGE;
			state2[offset]=VEGE;
		}
	}	
	if (x < DIM/2+R && x > DIM/2-R){
		if (y < DIM/2+R && y > DIM/2-R){
			state1[offset]=FIRE;
			state2[offset]=FIRE;
		}
	}
}

__global__ void init_state3(char* state1, char* state2){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x+y*blockDim.x*gridDim.x;

	float xf;
	curandState localState;
	curand_init(1234,offset,0,&localState);
	int nnodes=DIM*DIM;
	if(offset<nnodes){
		xf = curand_uniform(&localState);
		if(xf<0.66){
			state1[offset]=1;
			state2[offset]=1;
		}
	}
}

__global__ void init_state2(char* state1, char* state2){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x+y*blockDim.x*gridDim.x;
	state1[offset]=0;
	state2[offset]=0;
	if((x==1||x==2)&&(y==5||y==6)){
		state1[offset]=1;
		state2[offset]=1;
	}
	
	if(x==11&&(y==5||y==6||y==7)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==12&&(y==4||y==8)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==13&&(y==3||y==9)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==14&&(y==3||y==9)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==15&&y==6){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==16&&(y==4||y==8)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==17&&(y==5||y==6||y==7)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==18&&y==6){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==21&&(y==3||y==4||y==5)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==22&&(y==3||y==4||y==5)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==23&&(y==2||y==6)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if(x==25&&(y==1||y==2||y==6||y==7)){
		state1[offset]=1;
		state2[offset]=1;
	}
	if((x==35||x==36)&&(y==3||y==4)){
		state1[offset]=1;
		state2[offset]=1;
	}
}

int main( void ) {
	int n = 100;
	curandGenerator_t gen;
	float * devData , * hostData ;
	hostData = ( float *) calloc (n, sizeof ( float ));
	cudaMalloc (( void **)& devData , n* sizeof ( float ));
	curandCreateGenerator (&gen, CURAND_RNG_PSEUDO_DEFAULT );


	GPUAnimBitmap  bitmap( DIM, DIM, NULL );
	dim3 grid(DIM/16, DIM/16);
	dim3 block(16,16); 
	char *state1;
	char *state2;
	size_t size = DIM*DIM*sizeof(char);
	cudaMalloc((void**)&state1, size);
	cudaMalloc((void**)&state2, size);
	cudaMemset(state1,0,size);
	cudaMemset(state2,0,size);
	init_state<<<grid,block>>>(state1,state2);
	bitmap.state1=state1;
	bitmap.state2=state2;

	bitmap.anim_and_exit(
		(void (*)(uchar4*,void*,char*,char*))generate_frame, NULL );

	cudaFree(state1);
	cudaFree(state2);
	return 0;
}

int main2( void ){
	char c1=1;
	char c2=2;
	char c3=c1+c2;
	if(c3==3)
		printf("c3==3");
	return 0;
}

int main3( void ){
	char *res;
	char *state1;
	char *state2;
	uchar4 * pixels;
	dim3    grid_dim(DIM/16,DIM/16);
	dim3    block_dim(16,16);
	size_t size = DIM*DIM*sizeof(char);
	cudaMalloc((void**)&state1, size);
	cudaMalloc((void**)&state2, size);
	cudaMalloc((void**)&pixels, size);
	res = (char*)malloc(size);
	init_state2<<<grid_dim,block_dim>>>(state1,state2);
	cudaMemcpy(res,state1,size,cudaMemcpyDeviceToHost);
	/*
	for(int j=0; j<DIM*DIM; j++){
		printf("%i",res[j]);
		if(j%DIM==DIM-1)
			printf("\n");
	}
	printf("\n");
	
	cudaMemcpy(res,state2,size,cudaMemcpyDeviceToHost);
	for(int j=0; j<DIM*DIM; j++){
		printf("%i",res[j]);
		if(j%DIM==DIM-1)
			printf("\n");
	}
	printf("\n");
	*/
	for(int i=0; i<100; i++){
		if(i%2==0){
			kernel<<<grid_dim,block_dim>>>( pixels, state2, state1 );
			cudaMemcpy(res,state2,size,cudaMemcpyDeviceToHost);
		}
		else {
			kernel<<<grid_dim,block_dim>>>( pixels, state1, state2 );
			cudaMemcpy(res,state1,size,cudaMemcpyDeviceToHost);
		}
		for(int j=0; j<DIM*DIM; j++){
			printf("%i",res[j]);
			if(j%DIM==DIM-1)
				printf("\n");
		}
		printf("\n");
	}
	
	return 0;
}
