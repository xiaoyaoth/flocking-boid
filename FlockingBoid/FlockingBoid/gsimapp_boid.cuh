#ifndef GSIMAPP_BOID_H
#define GSIMAPP_BOID_H

#include "gsimcore.cuh"
#include "header.cuh"

enum BoidType {PREY_BOID, PREDATOR_BOID, FOOD_BOID};
enum BoidState{HUNGRY, NORMAL, FLEEING, STARVING, SEEKING_MATE};
__device__ float *randDebug;
#define STRIP 2

class BoidModel : public GModel{
public:
	Continuous2D *world, *worldH;
	GRandomGen *rgen, *rgenH;
	float cohesion;
	float avoidance;
	float randomness;
	float consistency;
	float momentum;
	float deadFlockerProbability;
	float neighborhood;
	float jump;
	BoidModel(){
		cohesion = 1.0;
		avoidance = 1.0;
		randomness = 1.0;
		consistency = 1.0;
		momentum = 1.0;
		deadFlockerProbability = 0.1;
		neighborhood = 150;
		jump = 0.7;
	}
	void allocOnDevice();
	void allocOnHost();
	__device__ Continuous2D* getWorld() const;
	__device__ void addToWorld(GAgent *ag, int idx);
};
class BaseBoid : public GAgent {
public:
	BoidModel *model;
	float2d_t lastd;
	bool dead;
	float DEFAULT_SPEED;
	BoidType btype;
	BoidState bstate;

	__device__ BaseBoid(){
		loc.x = 0;
		loc.y = 0;
		DEFAULT_SPEED = 0.7;
		this->dead = false;
	}
	__device__ float getOrientation2D();
	__device__ void  setOrientation2D(float val);
	__device__ float2d_t momentum();
	__device__ virtual void step(GModel* model) = 0;
};
class FoodBoid : public BaseBoid{
public:
	float scale;
	int amount;
	int respawnCount;
	__device__ FoodBoid(){
		scale = 100;
		amount = CONSTANT::FOOD_AMOUNT;
	}
	__device__ FoodBoid(float2d_t location){
		this->scale = 100;
		this->amount = CONSTANT::FOOD_AMOUNT;
		this->loc = location;
		this->btype = FOOD_BOID;
	}
	__device__ void reduce();
	__device__ void increase();
	__device__ void step(GModel* model);
};
class PreyBoid : public BaseBoid{
public:
	float fleeBonus;
	int hungerCount;
	int starveCount;
	int mateCount;
	bool horny;

	float SENSE_FOOD_RANGE;
	int HUNGER_LIMIT;
	int STARVE_LIMIT;

	__device__ PreyBoid(){
		this->loc.x = 0;
		this->loc.y = 0;
		this->btype = PREY_BOID;
		this->HUNGER_LIMIT = CONSTANT::PREY_HUNGER_LIMIT;
		this->STARVE_LIMIT = CONSTANT::PREY_STARVE_LIMIT;
		this->DEFAULT_SPEED = 0.7;
	}
	__device__ PreyBoid(float2d_t location){
		this->loc = location;
		this->btype = PREY_BOID;
		this->HUNGER_LIMIT = CONSTANT::PREY_HUNGER_LIMIT;
		this->STARVE_LIMIT = CONSTANT::PREY_STARVE_LIMIT;
		this->DEFAULT_SPEED = 0.7;
	}
	__device__ PreyBoid(PreyBoid *ag){
		this->loc = ag->loc;
		this->btype = ag->btype;
		this->HUNGER_LIMIT = ag->HUNGER_LIMIT;
		this->starveCount = ag->starveCount;
		this->DEFAULT_SPEED = ag->DEFAULT_SPEED;
		this->model = ag->model;
		this->dummy = ag;
	}
	__device__ bool hungry();
	__device__ void eat(FoodBoid *food);
	__device__ bool starved();
	__device__ bool readyToMate();
	__device__ void setRandomSpeed();
	__device__ float2d_t randomness(GRandomGen *gen);
	__device__ float2d_t consistency(const Continuous2D *world);
	__device__ float2d_t cohesion(const Continuous2D *world);
	__device__ float2d_t avoidance(const Continuous2D *world);
	//__device__ float2d_t flee(Continuous2D *world);
	//__device__ float2d_t searchFood(Continuous2D *world);
	//__device__ float2d_t conformSpeed(Continuous2D *world);
	//__device__ float2d_t searchMate(Continuous2D *world);
	__device__ void step(GModel *state);
};
class PredatorBoid : public BaseBoid{
public:
	int hungerCount;
	int accCount;
	int starveCount;

	float neighborRange;
	float surrounding;
	float jump;

	float MAX_SPEED;
	int ACC_DURATION;
	int ACC_COOLDOWN;
	int STARVE_LIMIT;
	int HUNGER_LIMIT;

	PreyBoid *lastTarget;

	__device__ float distanceToOther(BaseBoid* ag);
	__device__ void accelerate();
	__device__ void decelerate();
	__device__ bool hungry();
	__device__ void feast();
	__device__ bool starved();
	__device__ float2d_t randomness(GRandomGen *gen);
	__device__ float2d_t huntPrimitive();
	__device__ float2d_t huntByLockOnNearest();
	__device__ float2d_t huntByLockOnRandom();
	__device__ float2d_t stray();
	__device__ void step(GModel *model);
};

//BoidModel
void BoidModel::allocOnDevice(){
	//init scheduler
	GModel::allocOnDevice();
	//init Continuous2D
	worldH = new Continuous2D(BOARDER_R_H, BOARDER_D_H, this->neighborhood/1.5);
	worldH->allocOnDevice();
	cudaMalloc((void**)&world, sizeof(Continuous2D));
	cudaMemcpy(world, worldH, sizeof(Continuous2D), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(this->schedulerH->assignments, &this->worldH->neighborIdx,
		//sizeof(int), 0, cudaMemcpyDeviceToDevice);
	//init GRandomGen
	//rgenH = new GRandomGen();
	//rgenH->allocOnDevice();
	//cudaMalloc((void**)&rgen, sizeof(GRandomGen));
	//cudaMemcpy(rgen, rgenH, sizeof(GRandomGen), cudaMemcpyHostToDevice);
	//int gSize = GRID_SIZE;
	//rgenUtil::initStates<<<gSize, BLOCK_SIZE>>>(rgen, 1234);
	getLastCudaError("BoidModel()");
}
void BoidModel::allocOnHost(){
	GModel::allocOnHost();

	world = new Continuous2D(BOARDER_R_H, BOARDER_D_H, this->neighborhood/1.5);
	world->allocOnHost();
	this->scheduler = new GScheduler();
	this->scheduler->allocOnHost();
}
__device__ Continuous2D* BoidModel::getWorld() const {
	return this->world;
}
__device__ void BoidModel::addToWorld(GAgent *ag, int idx){
	this->world->add(ag, idx);
}

//BaseBoid
__device__ float BaseBoid::getOrientation2D(){
	if (lastd.x = 0 && lastd.y == 0)
		return 0;
	return atan2(lastd.y, lastd.x);
}
__device__ void BaseBoid::setOrientation2D(float val){
	lastd.x = cos(val);
	lastd.y = sin(val);
}
__device__ float2d_t BaseBoid::momentum(){
	return lastd;
}

//FoodBoid
__device__ void FoodBoid::reduce(){
	//if (this->amount <= 0)
	//	this->dead = true;
	//else {
	//	this->amount--;
	//	this->scale -= (this->scale/this->amount);
	//}
}
__device__ void FoodBoid::increase(){
	//if (this->amount <= 0)
	//	this->dead = true;
	//else if (this->amount < CONSTANT::FOOD_AMOUNT) {
	//	if (this->model->rgen->nextFloat() < 0.05){
	//		amount++;
	//		this->scale += (this->scale/this->amount);
	//	}
	//}
}
__device__ void FoodBoid::step(GModel *model){
	//if(this->dead) {
	//	this->model->getWorld()->remove(this);
	//	return;
	//}
	//if(this->model->rgen->nextFloat() > 0.3)
	//	this->increase();
}

//PreyBoid
__device__ bool PreyBoid::hungry(){
	if (this->hungerCount == HUNGER_LIMIT) {
		this->bstate = HUNGRY;
		this->starveCount++;
		return true;
	} else {
		this->hungerCount++;
		this->bstate = NORMAL;
		return false;
	}
}
__device__ void PreyBoid::eat(FoodBoid *food){
	//food->reduce();
	//this->hungerCount = 0;
	//this->starveCount = 0;
	//this->mateCount++;
	//float hornyValue = 
	//	(float)(this->mateCount/CONSTANT::PREY_MATING_TIME);
	//if (this->model->rgen->nextFloat() < hornyValue)
	//	this->horny = true;

}
__device__ bool PreyBoid::starved(){
	return this->starveCount == this->STARVE_LIMIT;
}
__device__ bool PreyBoid::readyToMate(){
	return this->horny;
}
__device__ void PreyBoid::setRandomSpeed(){
	this->DEFAULT_SPEED = CONSTANT::PREY_STD_SPEED + 
		this->model->rgen->nextGaussian() * 0.2;

}
__device__ float2d_t PreyBoid::randomness(GRandomGen *gen){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float x = randDebug[STRIP*idx];
	float y = randDebug[STRIP*idx+1];
	float l = sqrt(x*x + y*y);
	return float2d_t(0.05*x/l, 0.05*y/l);
}
__device__ float2d_t PreyBoid::consistency(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;

#if TRIAL_NEIGHBOR == 0
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while (nnc != STOP){
		PreyBoid *other = (PreyBoid*)world->obtainAgentByIterInfo(info.ptr);
		if(!other->dead){
			count++;
			float2d_t m = other->momentum();
			x += m.x;
			y += m.y;
		}
		nnc = world->nextNeighbor(info);
	}
#else
	NextNeighborControl nnc = world->nextNeighborInit2(this, this->model->neighborhood);
	while (nnc != STOP){
		PreyBoid *other = (PreyBoid*)world->obtainAgentByIterInfo2();
		if(!other->dead){
			count++;
			float2d_t m = other->momentum();
			x += m.x;
			y += m.y;
		}
		nnc = world->nextNeighbor2();
	}
#endif
	if (count > 0){
		x /= count;
		y /= count;
	}
	return float2d_t(x,y);
}
__device__ float2d_t PreyBoid::cohesion(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;

#if TRIAL_NEIGHBOR == 0
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while (nnc != STOP){
		PreyBoid *other = (PreyBoid*)world->obtainAgentByIterInfo(info.ptr);
		if (!other->dead){
			count++;
			x += world->tdx(this->loc.x, other->loc.x);
			y += world->tdy(this->loc.y, other->loc.y);
		}
		nnc = world->nextNeighbor(info);
	}
#else
	NextNeighborControl nnc = world->nextNeighborInit2(this, this->model->neighborhood);
	while (nnc != STOP){
		PreyBoid *other = (PreyBoid*)world->obtainAgentByIterInfo2();
		if (!other->dead){
			count++;
			x += world->tdx(this->loc.x, other->loc.x);
			y += world->tdy(this->loc.y, other->loc.y);
		}
		nnc = world->nextNeighbor2();
	}
#endif
	if (count > 0){
		x /= count;
		y /= count;
	}
	return float2d_t(-x/10,-y/10);
}
__device__ float2d_t PreyBoid::avoidance(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;

#if TRIAL_NEIGHBOR == 0
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while(nnc != STOP){
		PreyBoid *other = (PreyBoid*)world->obtainAgentByIterInfo(info.ptr);
		if (!other->dead){
			count++;
			float dx = world->tdx(this->loc.x, other->loc.x);
			float dy = world->tdy(this->loc.y, other->loc.y);
			float sqrDist = dx*dx + dy*dy;
			x += dx/(sqrDist*sqrDist + 1);
			y += dy/(sqrDist*sqrDist + 1);
		}
		nnc = world->nextNeighbor(info);
	}
#else
	NextNeighborControl nnc = world->nextNeighborInit2(this, this->model->neighborhood);
	while(nnc != STOP){
		PreyBoid *other = (PreyBoid*)world->obtainAgentByIterInfo2();
		if (!other->dead){
			count++;
			float dx = world->tdx(this->loc.x, other->loc.x);
			float dy = world->tdy(this->loc.y, other->loc.y);
			float sqrDist = dx*dx + dy*dy;
			x += dx/(sqrDist*sqrDist + 1);
			y += dy/(sqrDist*sqrDist + 1);
		}
		nnc = world->nextNeighbor2();
	}
	const iterInfo &info = infoArray[threadIdx.x];
#endif
	if (count > 0){
		x /= count;
		y /= count;
	}
	return float2d_t(400*x, 400*y);
}
//__device__ float2d_t PreyBoid::flee(Continuous2D *world){return float2d_t(0,0);}
//__device__ float2d_t PreyBoid::searchFood(Continuous2D *world){return float2d_t(0,0);}
//__device__ float2d_t PreyBoid::conformSpeed(Continuous2D *world){return float2d_t(0,0);}
//__device__ float2d_t PreyBoid::searchMate(Continuous2D *world){return float2d_t(0,0);}
__device__ void PreyBoid::step(GModel *model){
	const BoidModel *boidModel = (BoidModel*) model;
	if (this->dead)
		return;
	const Continuous2D *world = boidModel->getWorld();
	float2d_t avoid = this->avoidance(world);
	float2d_t cohes = this->cohesion(world);
	float2d_t consi = this->consistency(world);
	//float2d_t rdnes = this->randomness(model->rgen);
	float2d_t momen = this->momentum();
	float dx = 
		cohes.x * boidModel->cohesion +
		avoid.x * boidModel->avoidance +
		consi.x * boidModel->consistency +
		//rdnes.x * boidModel->randomness +
		momen.x * boidModel->momentum;
	float dy = 
		cohes.y * boidModel->cohesion +
		avoid.y * boidModel->avoidance +
		consi.y * boidModel->consistency +
		//rdnes.y * boidModel->randomness +
		momen.y * boidModel->momentum;
	float dist = sqrt(dx*dx + dy*dy);
	if (dist > 0){
		dx = dx / dist * boidModel->jump;
		dy = dy / dist * boidModel->jump;
	}
	BaseBoid *dummy = (BaseBoid*)this->dummy;
	dummy->lastd.x = dx;
	dummy->lastd.y = dy;
	dummy->loc.x = world->stx(loc.x + dx);
	dummy->loc.y = world->sty(loc.y + dy);

	if (dummy->loc.x <0 || dummy->loc.y < 0) {
		printf("%f, %f, %f, %f", dummy->loc.x, dummy->loc.y, dummy->lastd.x, dummy->lastd.y);
		dummy->loc.x = world->stx(loc.x + dx);
	}
	randDebug[STRIP*this->ag_id] = this->dummy->loc.x;
	randDebug[STRIP*this->ag_id+1] = this->dummy->loc.y;

//	printf("%f, %f\n", dummy->loc.x, dummy->loc.y);
}

//PredatorBoid
__device__ float PredatorBoid::distanceToOther(BaseBoid *b){return 0;}
__device__ void PredatorBoid::accelerate(){return;}
__device__ void PredatorBoid::decelerate(){return;}
__device__ bool PredatorBoid::hungry(){return true;}
__device__ void PredatorBoid::feast(){return;}
__device__ bool PredatorBoid::starved(){return true;}
__device__ float2d_t PredatorBoid::randomness(GRandomGen *gen){return float2d_t(0,0);}
__device__ float2d_t PredatorBoid::huntPrimitive(){return float2d_t(0,0);}
__device__ float2d_t PredatorBoid::huntByLockOnNearest(){return float2d_t(0,0);}
__device__ float2d_t PredatorBoid::huntByLockOnRandom(){return float2d_t(0,0);}
__device__ float2d_t PredatorBoid::stray(){return float2d_t(0,0);}
__device__ void PredatorBoid::step(GModel *model){}

#endif