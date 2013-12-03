
#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"curand.h"
#include"curand_kernel.h"

#define HOSTGEN	1
#define DEVGEN	0
#define BLOCK_SIZE	1024
#define	CUDA_CALL(x)do{if((x)!=cudaSuccess){\
	printf("Errorat%s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}}while(0)

class RNG{
	curandState* states;
	__device__ float nextFloat(){
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		return curand_uniform(&states[idx]);
	}
	friend __global__ void initStates(RNG *rng);
};

__global__ void initStates(RNG *rng){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void setup_kernel(curandState*state){
	int id=threadIdx.x+blockIdx.x*64;
	/*Eachthreadgetssameseed,adifferentsequence
	number,nooffset*/
	curand_init(1234,id,0,&state[id]);
}

__global__ void generate_kernel(curandState*state,int*result){
	int id=threadIdx.x+blockIdx.x*64;
	int count=0;
	unsigned int x;
	/*Copystatetolocalmemoryforefficiency*/
	curandState localState=state[id];
	/*Generatepseudo-randomunsignedints*/
	//for(int n=0; n<100000; n++){
		x=curand(&localState);
		/*Checkiflowbitset*/
		if(x&1){
			count++;
		}
	//}
	/*Copystatebacktoglobalmemory*/
	state[id]=localState;
	/*Storeresults*/
	result[id]+=count;
}

__global__ void setZero(int* res){
	int idx = threadIdx.x + blockIdx.x * 64;
	res[idx] = 0;
}

__global__ void consume(int *res, unsigned int *pool){
	int idx = threadIdx.x + blockIdx.x * 64;
	int x = pool[idx];
	if (x&1)
		res[idx]+=1;
}

int test4_1()
{
	int i,total;
	curandState*devStates;
	int*devResults,*hostResults;
	/*Allocatespaceforresultsonhost*/
	hostResults=(int*)calloc(64*64,sizeof(int));
	/*Allocatespaceforresultsondevice*/
	CUDA_CALL(cudaMalloc((void**)&devResults,64*64*sizeof(int)));
	/*Setresultsto0*/
	CUDA_CALL(cudaMemset(devResults,0,64*64*sizeof(int)));
	/*Allocatespaceforprngstatesondevice*/
	CUDA_CALL(cudaMalloc((void**)&devStates,64*64*sizeof(curandState)));
	/*Setupprngstates*/
	setup_kernel<<<64,64>>>(devStates);
	/*Generateandusepseudo-random*/
	for(i=0;i<10000;i++){
		generate_kernel<<<64,64>>>(devStates,devResults);
	}
	/*Copydevicememorytohost*/
	CUDA_CALL(cudaMemcpy(hostResults,devResults,64*64*
		sizeof(int),cudaMemcpyDeviceToHost));
	/*Showresult*/
	total=0;
	for(i=0;i<64*64;i++){
		total+=hostResults[i];
	}
	printf("Fraction with low bit set was %10.13f\n",
		(float)total/(64.0f*64.0f*10000.0f));
	/*Cleanup*/
	CUDA_CALL(cudaFree(devStates));
	CUDA_CALL(cudaFree(devResults));
	free(hostResults);
	return EXIT_SUCCESS;
}

int test4_2(){
	unsigned int *pool;
	int *res;
	int *res_h;

	cudaMalloc(&pool,64*64*sizeof(int));
	cudaMalloc(&res, 64*64*sizeof(int));
	cudaMemset(res, 0, 64*64*sizeof(int));
	res_h = (int*)calloc(64*64,sizeof(int));

	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,time(0));

	for(int i=0;i<10000;i++){
		curandGenerate(gen, pool, 64*64);
		consume<<<64,64>>>(res, pool);
	}

	CUDA_CALL(cudaMemcpy(res_h,res,64*64*sizeof(int),cudaMemcpyDeviceToHost));

	int total=0;
	for(int i=0;i<64*64;i++){
		total+=res_h[i];
	}
	printf("Fraction with low bit set was %10.13f\n",
		(float)total/(64.0f*64.0f*10000.0f));
	/*Cleanup*/
	CUDA_CALL(cudaFree(pool));
	CUDA_CALL(cudaFree(res));
	free(res_h);
	return EXIT_SUCCESS;
}

int main(int argc, char** argv){
	int sel = atoi(argv[1]);
	if(sel == 1)
		test4_1();
	else
		test4_2();
	system("PAUSE");
	return EXIT_SUCCESS;
}