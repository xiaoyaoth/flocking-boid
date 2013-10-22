#include "testheader.cuh"

class GModel;
class Continuous2D{
private:
	int *neighborIdx, *cellIdx;
public:
	void allocOnDevice(){
		size_t sizeAgArray = AGENT_NO*sizeof(int);
		size_t sizeCellArray = CELL_NO*sizeof(int);

		cudaMalloc((void**)&neighborIdx, sizeAgArray);
		getLastCudaError("Continuous2D():cudaMalloc:neighborIdx");
		cudaMalloc((void**)&cellIdx, sizeCellArray);
		getLastCudaError("Continuous2D():cudaMalloc:cellIdx");


		int *neighborIdx_h = (int*)malloc(sizeAgArray);
		int *cellIdx_h = (int*)malloc(sizeCellArray);
		for(int i=0; i<AGENT_NO; i++)
			neighborIdx_h[i] = 2;
		for(int i = 0; i<CELL_NO; i++)
			cellIdx_h[i] = 3;
		cudaMemcpy(this->neighborIdx, neighborIdx_h, sizeAgArray, cudaMemcpyHostToDevice);
		getLastCudaError("Continuous2D():cudaMemcpy:neighborIdx");
		cudaMemcpy(this->cellIdx, cellIdx_h, sizeCellArray, cudaMemcpyHostToDevice);
		getLastCudaError("Continuous2D():cudaMemcpy:cellIdx");
	}
	void allocOnHost(){
		size_t sizeAgArray = AGENT_NO*sizeof(int);
		size_t sizeCellArray = CELL_NO*sizeof(int);
		neighborIdx = (int*)malloc(sizeAgArray);
		cellIdx = (int*)malloc(sizeCellArray);
	}
	friend void seeThrough(GModel *model);
	friend void test1();
};

class GModel{
private:
	Continuous2D *world;
public:
	void allocOnDevice(){
		Continuous2D *world_d = new Continuous2D();
		world_d->allocOnDevice();
		cudaMalloc((void**)&world, sizeof(Continuous2D));
		getLastCudaError("GModel():cudaMalloc:world");
		cudaMemcpy(world, world_d, sizeof(Continuous2D), cudaMemcpyHostToDevice);
		getLastCudaError("GModel():cudaMemcpy:world");
	}
	void allocOnHost(){
		world = new Continuous2D();
		world->allocOnHost();
	}
	friend void seeThrough(GModel *model);
	friend void test1();
};

__global__ void seeThrough(GModel *model){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO){
		int a = model->world->neighborIdx[idx];
		int b = model->world->cellIdx[idx];
	}
}

void test1(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);

	GModel *model_h = new GModel();
	model_h->allocOnDevice();
	GModel *model_d;
	cudaMalloc((void**)&model_d, sizeof(GModel));
	cudaMemcpy(model_d, model_h, sizeof(GModel), cudaMemcpyHostToDevice);
	delete model_h;
	seeThrough<<<1, BLOCK_SIZE>>>(model_d);
	model_h = new GModel();
	cudaMemcpy(model_h, model_d, sizeof(GModel), cudaMemcpyDeviceToHost);
	Continuous2D *world_h = new Continuous2D();
	cudaMemcpy(world_h, model_h->world, sizeof(Continuous2D), cudaMemcpyDeviceToHost);
	int *data_h = (int*)malloc(sizeAgArray);
	int *data2_h = (int*)malloc(sizeCellArray);
	cudaMemcpy(data_h, world_h->neighborIdx, sizeAgArray, cudaMemcpyDeviceToHost);
	cudaMemcpy(data2_h, world_h->cellIdx, sizeCellArray, cudaMemcpyDeviceToHost);
	for (int i=0; i<AGENT_NO; i++)
		printf("%d ", data_h[i]);
	for (int i=0; i<CELL_NO; i++)
		printf("%d ", data2_h[i]);

	printf("\nDONE");

}

int main3(){
	test1();
	return 0;
}

