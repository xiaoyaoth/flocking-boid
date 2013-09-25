#ifndef GSIMCORE_H
#define GSIMCORE_H

#include "header.cuh"
#include <thrust\sort.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <thrust\functional.h>
#include <thrust\transform.h>
#include <curand_kernel.h>

//class delaration
class GSteppalbe;
class GAgent;
class GIterativeAgent;
class Continuous2D;
class GScheduler;
class GModel;
class GRandomGen;

typedef struct iter_info_per_thread
{
	int2d_t cell_cur;
	int2d_t cell_ul;
	int2d_t cell_dr;

	int ptr;
	int boarder;
	int count;
	const GAgent *agent;

	float range;

	//__device__ __host__ void print(){
	//	printf("======iter info======\n");
	//	printf("agent_id: %d", ag_id);
	//	printf("cell_cur: ");
	//	cell_cur.print();
	//	printf("\ncell_ul: ");
	//	cell_ul.print();
	//	printf("\ncell_dr: ");
	//	cell_dr.print();
	//	printf("\nptr: %d\ncellIdx_border: %d\n", ptr, boarder);
	//	printf("=====================\n");
	//}
} iterInfo;
enum NextNeighborControl{CONTINUE, STOP, FOUND};

namespace c2dUtil{
	void gen_hash_kernel(int *hash, Continuous2D *c2d);
	void sort_hash_kernel(int *hash, int *neighborIdx);
	void gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	void queryNeighbor(Continuous2D *c2d);
	void genNeighbor(GModel *model);
};
namespace schUtil{
	void sortWithKey(GModel *model);
	__global__ void scheduleRepeatingAllAgents(GModel *gm);
	__global__ void step(GModel *gm);
}
namespace rgenUtil{
	__global__ void initStates(GRandomGen *rgen, int seed);
}

class GSteppable{
public:
	float time;
	int rank;
	virtual __device__ void step(GModel *model) = 0;
};
class GAgent : public GSteppable{
private:
	int ag_id;
public:
	float2d_t loc;
	__device__ GAgent(){
		this->ag_id = threadIdx.x + blockIdx.x * blockDim.x;
	}
	__device__ void allocOnDevice();
	__device__ int getAgId();
	__device__ virtual void step(GModel *model) = 0;
};
class GIterativeAgent : public GAgent{
private:
	GAgent *realAg;
	float interval;
public:
	__device__ GIterativeAgent(float time, int rank, 
		GAgent *ag, float interval){
			this->realAg = ag;
			this->time = time;
			this->rank = rank;
			this->interval = interval;
	}
	__device__ void step(GModel *model);
};
class Continuous2D{
private:
	GAgent **allAgents;
	int *neighborIdx, *cellIdx;
	__device__ NextNeighborControl nextNeighborPrimitive(iterInfo &info);
public:
	void allocOnDevice();
	void allocOnHost();
	__device__ float tdx(float ax, float bx);
	__device__ float tdy(float ay, float by);
	__device__ float tds(float2d_t aloc, float2d_t bloc);
	__device__ bool add(GAgent *ag);
	__device__ bool remove(GAgent *ag);
	__device__ GAgent* obtainAgentPerThread();
	__device__ NextNeighborControl nextNeighborInit(const GAgent* ag, const float range, iterInfo &info);
	__device__ NextNeighborControl nextNeighbor(iterInfo &info);

	friend void c2dUtil::gen_hash_kernel(int *hash, Continuous2D *c2d);
	friend void c2dUtil::gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	friend void c2dUtil::genNeighbor(GModel *model);
	friend void c2dUtil::queryNeighbor(Continuous2D *c2d);
	friend class GModel;
};
class GScheduler{
private:
	GAgent **allAgents;
	float time;
	int steps;
public:
	__device__ bool ScheduleOnce(const float time, const int rank,
		GAgent *ag);
	__device__ bool scheduleRepeating(const float time, const int rank, 
		GAgent *ag, const float interval);
	__device__ GAgent* obtainAgentPerThread();
	void allocOnHost();
	void allocOnDevice();
	friend void schUtil::sortWithKey(GModel *model);
	friend class GModel;
};
class GRandomGen{
public:
	curandState *states;
public:
	void allocOnDevice();
	__device__ float nextFloat();
	__device__ float nextGaussian();
	__device__ float nextFloat(curandState *state);
	friend void rgenUtil::initStates(GRandomGen *rgen, int seed);
};
class GModel{
private:
	Continuous2D *world, *worldH;
	GScheduler *scheduler, *schedulerH;
public:
	GRandomGen *rgen, *rgenH;
	void allocOnHost();
	void allocOnDevice();
	__device__ Continuous2D* getWorld();
	__device__ GScheduler* getScheduler();
	//__device__ GRandomGen* getGRandomGen();
	__device__ void addToWorld(GAgent *ag, int idx);
	__device__ void addToScheduler(GAgent *ag, int idx);
	__device__ void foo();
	friend void schUtil::sortWithKey(GModel *model);
	friend void c2dUtil::genNeighbor(GModel *model);
};

//Continuous2D
void Continuous2D::allocOnDevice(){

	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);

	cudaMalloc((void**)&this->allAgents, AGENT_NO*sizeof(GAgent*));
	cudaCheckErrors("Continuous2D():cudaMalloc:allAgents");
	cudaMalloc((void**)&neighborIdx, sizeAgArray);
	cudaCheckErrors("Continuous2D():cudaMalloc:neighborIdx");
	cudaMalloc((void**)&cellIdx, sizeCellArray);
	cudaCheckErrors("Continuous2D():cudaMalloc:cellIdx");

#if C2DDEBUG == 1
	int *neighborIdx_h = (int*)malloc(sizeAgArray);
	for(int i=0; i<AGENT_NO; i++)
		neighborIdx_h[i] = 2;
	cudaMemcpy(this->neighborIdx, neighborIdx_h, sizeAgArray, cudaMemcpyHostToDevice);
	cudaCheckErrors("Continuous2D():cudaMemcpy:neighborIdx");
	cudaMemcpy(this->cellIdx, neighborIdx_h, sizeCellArray, cudaMemcpyHostToDevice);
	cudaCheckErrors("Continuous2D():cudaMemcpy:cellIdx");
#endif
}
void Continuous2D::allocOnHost(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);
	neighborIdx = (int*)malloc(sizeAgArray);
	cellIdx = (int*)malloc(sizeCellArray);
}
__device__ float Continuous2D::tdx(float ax, float bx){return 0;}
__device__ float Continuous2D::tdy(float ay, float by){return 0;}
__device__ float Continuous2D::tds(float2d_t loc1, float2d_t loc2){
	float dx = loc1.x - loc2.x;
	float dy = loc2.x - loc2.y;
	return sqrt(dx*dx + dy*dy);
}
__device__ bool Continuous2D::add(GAgent *ag){return true;}
__device__ bool Continuous2D::remove(GAgent *ag){return true;}
__device__ GAgent* Continuous2D::obtainAgentPerThread(){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D)
		return this->allAgents[idx];
	else
		return NULL;
}
__device__ NextNeighborControl Continuous2D::nextNeighborInit(const GAgent* ag, 
		const float range, iterInfo &info){
	float2d_t pos = ag->loc;
	info.agent = ag;
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.range = range;
	info.cell_ul.x = (pos.x-range)>BOARDER_L ? 
		(int)(pos.x-range)/CELL_RESO : (int)BOARDER_L/CELL_RESO;
	info.cell_dr.x = (pos.x+range)<BOARDER_R ?
		(int)(pos.x+range)/CELL_RESO : (int)BOARDER_R/CELL_RESO-1;
	info.cell_ul.y = (pos.y-range)>BOARDER_U ? 
		(int)(pos.y-range)/CELL_RESO : (int)BOARDER_U/CELL_RESO;
	info.cell_dr.y = (pos.y+range)<BOARDER_D ? 
		(int)(pos.y+range)/CELL_RESO : (int)BOARDER_D/CELL_RESO-1;
	info.cell_cur.x = info.cell_ul.x;
	info.cell_cur.y = info.cell_ul.y;
	info.ptr = cellIdx[info.cell_cur.cell_id()];
	if (info.cell_cur.cell_id() == CELL_NO_D-1)
		info.boarder = AGENT_NO_D-1;
	else
		info.boarder = cellIdx[info.cell_cur.cell_id()+1];

	GAgent *other = this->allAgents[this->neighborIdx[info.ptr]];
	if (tds(ag->loc, other->loc) < range){
		info.count++;
		return FOUND;
	} else
		return this->nextNeighbor(info);
}
__device__ NextNeighborControl Continuous2D::nextNeighborPrimitive(iterInfo &info){
	info.ptr++;
	info.count++;
	if (info.ptr > info.boarder) {
		info.cell_cur.x++;
		if (info.cell_cur.x <= info.cell_dr.x)
			info.ptr = cellIdx[info.cell_cur.cell_id()];
		else {
			info.cell_cur.x = info.cell_ul.x;
			info.cell_cur.y++;
			if(info.cell_cur.y <= info.cell_dr.y)
				info.ptr = cellIdx[info.cell_cur.cell_id()];
			else
				return STOP;
		}
		if (info.cell_cur.cell_id() == CELL_NO_D-1)
			info.boarder = AGENT_NO_D-1;
		else
			info.boarder = cellIdx[info.cell_cur.cell_id()+1];
	}
	//info.print();
	return CONTINUE;
}
__device__ NextNeighborControl Continuous2D::nextNeighbor(iterInfo &info){
	NextNeighborControl nnc = this->nextNeighborPrimitive(info);
	GAgent *other;
	while (nnc == CONTINUE){
		other = this->allAgents[this->neighborIdx[info.ptr]];
		if (tds(info.agent->loc, other->loc) < info.range){
			return FOUND;
		}
		nnc = this->nextNeighborPrimitive(info);
	}
	return nnc;
}
//GAgent
__device__ int	GAgent::getAgId(){
	return this->ag_id;
}
__device__ void GAgent::allocOnDevice(){
	ag_id = threadIdx.x + blockIdx.x * blockDim.x;
}
//GIterativeAgent
__device__ void GIterativeAgent::step(GModel *model){
	this->time += this->interval;
	this->realAg->step(model);
	model->getScheduler()->ScheduleOnce(this->time, this->rank, this);
}

//GScheduler
__device__ bool GScheduler::ScheduleOnce(const float time, 	const int rank, GAgent *ag){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		ag->time = time;
		ag->rank = rank;
		allAgents[idx] = ag;
	}
	return true;
}
__device__ bool GScheduler::scheduleRepeating(const float time, const int rank, GAgent *ag, const float interval){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		GIterativeAgent *iterAg = new GIterativeAgent(time, rank, ag, interval);
		allAgents[idx] = iterAg;
	}
	return true;
}
__device__ GAgent* GScheduler::obtainAgentPerThread(){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D)
		return this->allAgents[idx];
	else
		return NULL;
}
void GScheduler::allocOnHost(){
}
void GScheduler::allocOnDevice(){
	cudaMalloc((void**)&allAgents, AGENT_NO*sizeof(GAgent*));
	cudaMalloc((void**)&time, sizeof(int));
	cudaMalloc((void**)&steps, sizeof(int));
	cudaCheckErrors("Scheduler::allocOnDevice:cudaMalloc");
}

//GModel
void GModel::allocOnDevice(){
	worldH = new Continuous2D();
	worldH->allocOnDevice();
	cudaMalloc((void**)&world, sizeof(Continuous2D));
	cudaMemcpy(world, worldH, sizeof(Continuous2D), cudaMemcpyHostToDevice);

	schedulerH = new GScheduler();
	schedulerH->allocOnDevice();
	cudaMalloc((void**)&scheduler, sizeof(GScheduler));
	cudaMemcpy(scheduler, schedulerH, sizeof(GScheduler), cudaMemcpyHostToDevice);

	rgenH = new GRandomGen();
	rgenH->allocOnDevice();
	cudaMalloc((void**)&rgen, sizeof(GRandomGen));
	cudaMemcpy(rgen, rgenH, sizeof(GRandomGen), cudaMemcpyHostToDevice);
	int gSize = GRID_SIZE;
	//rgenUtil::initStates<<<gSize, BLOCK_SIZE>>>(rgen, 1234);

	cudaCheckErrors("GModel()");
}
void GModel::allocOnHost(){
	world = new Continuous2D();
	world->allocOnHost();
	this->scheduler = new GScheduler();
	this->scheduler->allocOnHost();
}
__device__ Continuous2D* GModel::getWorld(){
	return this->world;
}
__device__ GScheduler* GModel::getScheduler(){
	return this->scheduler;
}
__device__ void GModel::addToWorld(GAgent *ag, int idx){
	this->world->allAgents[idx] = ag;
}
__device__ void GModel::addToScheduler(GAgent *ag, int idx){
	this->scheduler->allAgents[idx] = ag;
}

//GRandomGen
void GRandomGen::allocOnDevice(){
	size_t genRandStatesSize = GRID_SIZE*BLOCK_SIZE*sizeof(curandState);
	printf("curandStateSize: %d\n", genRandStatesSize);
	cudaMalloc((void**)&states,genRandStatesSize);
	cudaCheckErrors("GRandomGen::allocOnDevice");
}
__device__ float GRandomGen::nextFloat(){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	return 0;
	//return curand_uniform(&states[idx]);
}
__device__ float GRandomGen::nextFloat(curandState *state){
	return curand_uniform(state);
}
__device__ float GRandomGen::nextGaussian(){return 0;}

//namespace continuous2D Utility
int count = 0;
__global__ void c2dUtil::gen_hash_kernel(int *hash, Continuous2D *c2d)
{
	GAgent *ag = c2d->obtainAgentPerThread();
	if(ag != NULL) {
		int idx = ag->getAgId();
		hash[idx] = (int)(ag->loc.x/100) + 10 * (int)(ag->loc.y/100);
		c2d->neighborIdx[idx] = ag->getAgId();
	}
}
__global__ void c2dUtil::gen_cellIdx_kernel(int *hash, Continuous2D *c2d)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D && idx > 0) {
		if (hash[idx] != hash[idx-1])
			c2d->cellIdx[hash[idx]] = idx;
	}
	if (idx == 0)
		c2d->cellIdx[idx] = idx;
}
__global__ void c2dUtil::queryNeighbor(Continuous2D *c2d){
	iterInfo info;
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < AGENT_NO_D){
		GAgent *ag = c2d->allAgents[idx];
		int ptr = c2d->nextNeighborInit(ag, 200, info);
		while(ptr!=-1)
			ptr = c2d->nextNeighbor(info);
	}
}
void c2dUtil::sort_hash_kernel(int *hash, int *neighborIdx)
{
	thrust::device_ptr<int> id_ptr(neighborIdx);
	thrust::device_ptr<int> hash_ptr(hash);
	typedef thrust::device_vector<int>::iterator Iter;
	Iter key_begin(hash_ptr);
	Iter key_end(hash_ptr + AGENT_NO);
	Iter val_begin(id_ptr);
	thrust::sort_by_key(key_begin, key_end, val_begin);
}
void c2dUtil::genNeighbor(GModel *model)
{
	int bSize = BLOCK_SIZE;
	int gSize = GRID_SIZE;
	if (AGENT_NO%bSize != 0)
		gSize++;

	GModel *model_h = new GModel();
	cudaMemcpy(model_h, model, sizeof(GModel), cudaMemcpyDeviceToHost);
	printf("sizeof(GModel): %d\n", sizeof(GModel));
	cudaCheckErrors("genNeighbor:cudaMemcpy:model_h");
	Continuous2D *c2d_h = new Continuous2D();
	cudaMemcpy(c2d_h, model_h->world, sizeof(Continuous2D), cudaMemcpyDeviceToHost);
	cudaCheckErrors("genNeighbor:cudaMemcpy:c2d_h");

	int *hash;
	cudaMalloc((void**)&hash, AGENT_NO*sizeof(int));
	cudaCheckErrors("genNeighbor:cudaMalloc:hash");
	gen_hash_kernel<<<gSize, bSize>>>(hash, model_h->world);
	sort_hash_kernel(hash, c2d_h->neighborIdx);
	gen_cellIdx_kernel<<<gSize, bSize>>>(hash, model_h->world);

	//debug
	//int *id_h, *hash_h, *cidx_h;
	//id_h = new int[AGENT_NO];
	//hash_h = new int[AGENT_NO];
	//cidx_h = new int[CELL_NO];
	//cudaMemcpy(id_h, c2d_h->neighborIdx, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaCheckErrors("genNeighbor:cudaMemcpy(id_h");
	//cudaMemcpy(hash_h, hash, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaCheckErrors("genNeighbor:cudaMemcpy(hash_h");
	//cudaMemcpy(cidx_h, c2d_h->cellIdx, CELL_NO * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaCheckErrors("genNeighbor:cudaMemcpy(cidx_h");
	//std::fstream fout;
	//char *outfname = new char[10];
	//sprintf(outfname, "out%d.txt", count++);
	//fout.open(outfname, std::ios::out);
	//for (int i = 0; i < AGENT_NO; i++){
	//	fout << id_h[i] << " " << hash_h[i] <<std::endl;
	//	fout.flush();
	//}
	//for (int i = 0; i < CELL_NO; i++){
	//	fout << cidx_h[i] <<std::endl;
	//	fout.flush();
	//}
	//fout.close();
	//~debug

	cudaFree(hash);
	cudaCheckErrors("genNeighbor:cudaFree:hash");
}

//namespace Scheduler Utility
struct SchdulerComp : public thrust::binary_function<GAgent, GAgent, bool> {
	__host__ __device__
		bool operator () (const GAgent *a, const GAgent *b) const
	{
		if (a->time != b->time)
			return a->time < b->time;
		else
			return a->rank < b->rank;
	}
};
void schUtil::sortWithKey(GModel *model){
	//it is the Scheduler in GModel that we are interested
	GModel *model_h = new GModel();
	GScheduler *sch_h = new GScheduler();
	cudaMemcpy(model_h, model, sizeof(GModel), cudaMemcpyDeviceToHost);
	cudaMemcpy(sch_h, model_h->scheduler, sizeof(GScheduler), cudaMemcpyDeviceToHost);
	thrust::device_ptr<GAgent*> ptr(sch_h->allAgents);
	thrust::sort(ptr, ptr+AGENT_NO, SchdulerComp());
}
__global__ void schUtil::scheduleRepeatingAllAgents(GModel *gm){
	GScheduler *sch = gm->getScheduler();
	GAgent *ag = sch->obtainAgentPerThread();
	if (ag != NULL){
		sch->scheduleRepeating(0,0,ag,1);
	}
}
__global__ void schUtil::step(GModel *gm){
	GScheduler *sch = gm->getScheduler();
	GAgent *ag = sch->obtainAgentPerThread();
	if (ag != NULL) {
		ag->step(gm);
	}
}

//namespace GRandomGen Utility
__global__ void rgenUtil::initStates(GRandomGen *rgen, int seed){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &rgen->states[idx]);
}

namespace gsim{

};

#endif