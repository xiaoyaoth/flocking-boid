#include "../FlockingBoid/header.cuh"

class Dummy{
public:
	int *data;

	__device__ void foo(){
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx<AGENT_NO_D){
			data[idx] = idx;
			printf("%d ", data[idx]);
		}
	}

	Dummy(){
		cudaMalloc((void**)&data, AGENT_NO*sizeof(int));
		getLastCudaError("Dummy cudaMalloc");
	}
};

class DummyPack{
public:
	Dummy *dummy;

	DummyPack(){
		Dummy *dummy_h = new Dummy();
		cudaMalloc((void**)&dummy, sizeof(Dummy));
		getLastCudaError("DummyPack cudaMalloc");
		cudaMemcpy(dummy, dummy_h, sizeof(Dummy), cudaMemcpyHostToDevice);
		getLastCudaError("DummyPack cudaMemcpy");
	}
};

__device__ Dummy *dummy;

__global__ void foo_kernel(Dummy *dummy){
	dummy->foo();
}

__global__ void test_dp_kernel(DummyPack *dp){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D)
		//printf("%d ", dp->dummy->data[idx]);
		dp->dummy->data[idx] = 1;
}

int test1(){
	Dummy *dummy_d;
	int *dummy_d_data;
	cudaMalloc((void**)&dummy_d, sizeof(Dummy));
	getLastCudaError("cudaMalloc dummy_d");
	cudaMalloc((void**)&dummy_d_data, AGENT_NO*sizeof(int));
	getLastCudaError("cudaMalloc dummy_d_data");
	//cudaMemcpy(&dummy, &dummy_d, sizeof(dummy_d), cudaMemcpyHostToDevice);
	//getLastCudaError("cudaMemcpyToSymbol dummy");
	cudaMemcpy(&dummy_d->data, &dummy_d_data, sizeof(dummy_d->data), cudaMemcpyHostToDevice);
	getLastCudaError("cudaMemcpyToSymbol dummy_data");

	foo_kernel<<<1, BLOCK_SIZE>>>(dummy_d);
	getLastCudaError("foo_kernel");

	
	Dummy *dummy_h = new Dummy();
	int *dummy_h_data = (int*)malloc(AGENT_NO*sizeof(int));
	cudaMemcpy(dummy_h, dummy_d, sizeof(Dummy), cudaMemcpyDeviceToHost);
	getLastCudaError("cudaMemcpy dummy_h");
	cudaMemcpy(dummy_h_data, dummy_h->data, AGENT_NO*sizeof(int), cudaMemcpyDeviceToHost);
	getLastCudaError("cudaMemcpy dummy_h_data");

	for (int i=0; i<AGENT_NO; i++)
		printf("%d ", dummy_h_data[i]);
	return 0;
}

int main2(){
	DummyPack *dp_h = new DummyPack();
	DummyPack *dp_d;
	cudaMalloc((void**)&dp_d, sizeof(DummyPack));
	cudaMemcpy(dp_d, dp_h, sizeof(DummyPack), cudaMemcpyHostToDevice);

	test_dp_kernel<<<1, AGENT_NO>>>(dp_d);
	dp_h = new DummyPack();
	cudaMemcpy(dp_h, dp_d, sizeof(DummyPack), cudaMemcpyDeviceToHost);
	Dummy *d_h = new Dummy();
	cudaMemcpy(d_h, dp_h->dummy, sizeof(Dummy), cudaMemcpyDeviceToHost);
	int *data_h = (int*)malloc(AGENT_NO*sizeof(int));
	cudaMemcpy(data_h, d_h->data, AGENT_NO*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<AGENT_NO; i++)
		printf("%d ", data_h[i]);
	
	printf("test");
	return 0;
}