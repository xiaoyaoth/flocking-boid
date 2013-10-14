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
	int2d_t cellCur;
	int2d_t cellUL;
	int2d_t cellDR;

	int ptr;
	int boarder;
	int count;
	const GAgent *agent;

	float range;

	//__device__ __host__ void print(){
	//	printf("======iter info======\n");
	//	printf("agent_id: %d", ag_id);
	//	printf("cellCur: ");
	//	cellCur.print();
	//	printf("\ncellUL: ");
	//	cellUL.print();
	//	printf("\ncellDR: ");
	//	cellDR.print();
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
	void genNeighbor(Continuous2D *world, Continuous2D *world_h);
	__global__ void swapAgentsInWorld(Continuous2D *world);
};
namespace schUtil{
	void sortWithKey(GModel *model);
	__global__ void scheduleRepeatingAllAgents(GModel *gm);
	__global__ void swapAgentsInScheduler(GModel *gm);
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
protected:
	GAgent *dummy;
	int ag_id;
public:
	float2d_t loc;
	__device__ GAgent(){
		this->ag_id = threadIdx.x + blockIdx.x * blockDim.x;
	}
	__device__ GAgent(float2d_t loc){
		this->ag_id = threadIdx.x + blockIdx.x * blockDim.x;
		this->loc = loc;
	}
	__device__ GAgent(float x, float y){
		this->ag_id = threadIdx.x + blockIdx.x * blockDim.x;
		this->loc.x = x;
		this->loc.y = y;
	}
	__device__ GAgent(GAgent *ag){ //prepared for creating the dummy;
		this->ag_id = ag->ag_id;
		this->loc = ag->loc;
		this->dummy = ag;
	}
	__device__ void allocOnDevice();
	__device__ int getAgId() const;
	__device__ void setDummy(GAgent *dummy);
	__device__ GAgent* getDummy() const;
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
	float width;
	float height;
	float discretization;
	GAgent **allAgents;
	int *neighborIdx, *cellIdx;
	__device__ NextNeighborControl nextNeighborPrimitive(iterInfo &info) const;
public:
	Continuous2D(float w, float h, float disc){
		this->width = w;
		this->height = h;
		this->discretization = disc;
	}
	void allocOnDevice();
	void allocOnHost();
	//agent list manipulation
	__device__ bool add(GAgent *ag, int idx);
	__device__ bool remove(GAgent *ag);
	__device__ void swap();
	__device__ GAgent* obtainAgentPerThread() const;
	__device__ GAgent* obtainAgent(const int agIdx) const;
	__device__ GAgent* obtainAgentFromNeighborIdx(const int ptr) const;
	//distance utility
	__device__ float stx(float x) const;
	__device__ float sty(float y) const;
	__device__ float tdx(float ax, float bx) const;
	__device__ float tdy(float ay, float by) const;
	__device__ float tds(float2d_t aloc, float2d_t bloc) const;
	//Neighbors related
	__device__ NextNeighborControl nextNeighborInit(const GAgent* ag, const float range, iterInfo &info) const;
	__device__ NextNeighborControl nextNeighbor(iterInfo &info) const;
	//__global__ functions
	friend void c2dUtil::gen_hash_kernel(int *hash, Continuous2D *c2d);
	friend void c2dUtil::gen_cellIdx_kernel(int *hash, Continuous2D *c2d);
	friend void c2dUtil::genNeighbor(Continuous2D *world, Continuous2D *world_h);
	friend void c2dUtil::queryNeighbor(Continuous2D *c2d);
	//friend class GModel;
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
	__device__ GAgent* obtainAgentPerThread() const;
	__device__ bool add(GAgent* ag, int idx);
	__device__ bool swap();
	void allocOnHost();
	void allocOnDevice();
	friend void schUtil::sortWithKey(GModel *model);
	//friend class GModel;
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
protected:
	GScheduler *scheduler, *schedulerH;
public:
	void allocOnHost();
	void allocOnDevice();
	__device__ GScheduler* getScheduler() const;
	__device__ void addToScheduler(GAgent *ag, int idx);
	__device__ void foo();
	friend void schUtil::sortWithKey(GModel *model);
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
}
void Continuous2D::allocOnHost(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);
	neighborIdx = (int*)malloc(sizeAgArray);
	cellIdx = (int*)malloc(sizeCellArray);
}
__device__ bool Continuous2D::add(GAgent *ag, int idx) {
	if(idx>=AGENT_NO_D)
		return false;
	this->allAgents[idx]=ag;
	return true;
}
__device__ bool Continuous2D::remove(GAgent *ag){return true;}
__device__ void Continuous2D::swap(){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx<AGENT_NO_D){
		GAgent *ag = this->allAgents[idx];
		this->allAgents[idx] = ag->getDummy();
	}
}
__device__ GAgent* Continuous2D::obtainAgentPerThread() const {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D)
		return this->allAgents[idx];
	else
		return NULL;
}
__device__ GAgent* Continuous2D::obtainAgent(int agIdx) const {
	if (agIdx<AGENT_NO_D && agIdx>=0)
		return this->allAgents[agIdx];
	return NULL;
}
__device__ GAgent* Continuous2D::obtainAgentFromNeighborIdx(const int ptr) const{
	if (ptr<AGENT_NO_D && ptr>=0){
		const int agIdx = this->neighborIdx[ptr];
		return this->allAgents[agIdx];
	}
	return NULL;
}
__device__ float Continuous2D::stx(const float x) const{
	if (x >= 0){
		if (x < this->width)
			return x;
		return x - this->width;
	}
	return x + this->width;
}
__device__ float Continuous2D::sty(const float y) const {
	if (y >= 0) {
		if (y < this->height)
			return y;
		return y - height;
	}
	return y + height;
}
__device__ float Continuous2D::tdx(float ax, float bx) const {
	float dx = abs(ax-bx);
	if (dx < BOARDER_R_D/2)
		return dx;
	else
		return BOARDER_R_D-dx;
}
__device__ float Continuous2D::tdy(float ay, float by) const {
	float dy = abs(ay-by);
	if (dy < BOARDER_D_D/2)
		return dy;
	else
		return BOARDER_D_D-dy;
}
__device__ float Continuous2D::tds(const float2d_t loc1, const float2d_t loc2) const {
	float dx = loc1.x - loc2.x;
	float dy = loc1.y - loc2.y;
	return sqrt(dx*dx + dy*dy);
}
__device__ NextNeighborControl Continuous2D::nextNeighborInit(const GAgent* ag, 
		const float range, iterInfo &info) const {
	float2d_t pos = ag->loc;
	info.agent = ag;
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.range = range;
	info.cellUL.x = (pos.x-range)>BOARDER_L_D ? 	(int)(pos.x-range)/CELL_RESO : (int)BOARDER_L_D/CELL_RESO;
	info.cellDR.x = (pos.x+range)<BOARDER_R_D ?	(int)(pos.x+range)/CELL_RESO : (int)BOARDER_R_D/CELL_RESO-1;
	info.cellUL.y = (pos.y-range)>BOARDER_U_D ? 	(int)(pos.y-range)/CELL_RESO : (int)BOARDER_U_D/CELL_RESO;
	info.cellDR.y = (pos.y+range)<BOARDER_D_D ? 	(int)(pos.y+range)/CELL_RESO : (int)BOARDER_D_D/CELL_RESO-1;
	info.cellCur.x = info.cellUL.x;
	info.cellCur.y = info.cellUL.y;
	info.ptr = cellIdx[info.cellCur.cell_id()];
	while (info.ptr == -1) {
		info.cellCur.x++;
		if (info.cellCur.x > info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if (info.cellCur.y > info.cellDR.y)
				return STOP;
		}
		info.ptr = cellIdx[info.cellCur.cell_id()];
	}
	if (info.cellCur.cell_id() == CELL_NO_D-1)
		info.boarder = AGENT_NO_D-1;
	else
		info.boarder = cellIdx[info.cellCur.cell_id()+1];

	GAgent *other = this->allAgents[this->neighborIdx[info.ptr]];
	if (tds(ag->loc, other->loc) < range){
		info.count++;
		return FOUND;
	} else
		return this->nextNeighbor(info);
}
__device__ NextNeighborControl Continuous2D::nextNeighborPrimitive(iterInfo &info) const {
	info.ptr++;

	if (info.ptr >= info.boarder) {
		info.cellCur.x++;
		if (info.cellCur.x > info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y <= info.cellDR.y)
				info.ptr = cellIdx[info.cellCur.cell_id()];
			else{
				return STOP;
			}
		}
		if (info.cellCur.cell_id() == CELL_NO_D-1)
			info.boarder = AGENT_NO_D-1;
		else
			info.boarder = cellIdx[info.cellCur.cell_id()+1];
	}
	//info.print();
	return CONTINUE;
}
__device__ NextNeighborControl Continuous2D::nextNeighbor(iterInfo &info) const {
	NextNeighborControl nnc = this->nextNeighborPrimitive(info);
	GAgent *other;
	float ds;
	while (nnc == CONTINUE){
		other = this->allAgents[this->neighborIdx[info.ptr]];
		ds = tds(info.agent->loc, other->loc);
		if (ds < info.range){
			info.count++;
			return FOUND;
		}
		nnc = this->nextNeighborPrimitive(info);
	}
	return nnc;
}

//GAgent
__device__ int	GAgent::getAgId() const {
	return this->ag_id;
}
__device__ void GAgent::allocOnDevice(){
	ag_id = threadIdx.x + blockIdx.x * blockDim.x;
}
__device__ void GAgent::setDummy(GAgent* dummy){
	this->dummy = dummy;
}
__device__ GAgent* GAgent::getDummy() const {
	return this->dummy;
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
__device__ GAgent* GScheduler::obtainAgentPerThread() const {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D)
		return this->allAgents[idx];
	else
		return NULL;
}
__device__ bool GScheduler::add(GAgent *ag, int idx){
	if(idx>=AGENT_NO_D)
		return false;
	this->allAgents[idx] = ag;
	return true;
}
__device__ bool GScheduler::swap(){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx<AGENT_NO_D){
		GAgent *ag = this->allAgents[idx];
		this->allAgents[idx] = ag->getDummy();
	}
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
	schedulerH = new GScheduler();
	schedulerH->allocOnDevice();
	cudaMalloc((void**)&scheduler, sizeof(GScheduler));
	cudaMemcpy(scheduler, schedulerH, sizeof(GScheduler), cudaMemcpyHostToDevice);
	cudaCheckErrors("GModel()");
}
void GModel::allocOnHost(){

}
__device__ GScheduler* GModel::getScheduler() const {
	return this->scheduler;
}
__device__ void GModel::addToScheduler(GAgent *ag, int idx){
	this->scheduler->add(ag, idx);
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
__global__ void c2dUtil::gen_hash_kernel(int *hash, Continuous2D *c2d)
{
	GAgent *ag = c2d->obtainAgentPerThread();
	if(ag != NULL) {
		int idx = ag->getAgId();
		hash[idx] = (int)(ag->loc.x/CELL_RESO) + XLENGTH * (int)(ag->loc.y/CELL_RESO);
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
void c2dUtil::genNeighbor(Continuous2D *world, Continuous2D *world_h)
{
	static int iterCount = 0;
	int bSize = BLOCK_SIZE;
	int gSize = GRID_SIZE;
	if (AGENT_NO%bSize != 0)
		gSize++;

	int *hash;
	cudaMalloc((void**)&hash, AGENT_NO*sizeof(int));
	cudaMemset(world_h->cellIdx, 0xff, CELL_NO*sizeof(int));
	cudaCheckErrors("genNeighbor:cudaMalloc:hash");
	gen_hash_kernel<<<gSize, bSize>>>(hash, world);
	sort_hash_kernel(hash, world_h->neighborIdx);
	gen_cellIdx_kernel<<<gSize, bSize>>>(hash, world);

	//debug
	if (iterCount == SELECTION){
		int *id_h, *hash_h, *cidx_h;
		id_h = new int[AGENT_NO];
		hash_h = new int[AGENT_NO];
		cidx_h = new int[CELL_NO];
		cudaMemcpy(id_h, world_h->neighborIdx, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
		cudaCheckErrors("genNeighbor:cudaMemcpy(id_h");
		cudaMemcpy(hash_h, hash, AGENT_NO * sizeof(int), cudaMemcpyDeviceToHost);
		cudaCheckErrors("genNeighbor:cudaMemcpy(hash_h");
		cudaMemcpy(cidx_h, world_h->cellIdx, CELL_NO * sizeof(int), cudaMemcpyDeviceToHost);
		cudaCheckErrors("genNeighbor:cudaMemcpy(cidx_h");
		std::fstream fout;
		char *outfname = new char[10];
		sprintf(outfname, "out%d.txt", iterCount);
		fout.open(outfname, std::ios::out);
		for (int i = 0; i < AGENT_NO; i++){
			fout << id_h[i] << " " << hash_h[i] <<std::endl;
			fout.flush();
		}
		for (int i = 0; i < CELL_NO; i++){
			fout << cidx_h[i] <<std::endl;
			fout.flush();
		}
		fout.close();
	}
	//~debug

	iterCount++;
	cudaFree(hash);
	cudaCheckErrors("genNeighbor:cudaFree:hash");
}
__global__ void c2dUtil::swapAgentsInWorld(Continuous2D *world){
	world->swap();
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
__global__ void schUtil::swapAgentsInScheduler(GModel *model) {
	model->getScheduler()->swap();
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