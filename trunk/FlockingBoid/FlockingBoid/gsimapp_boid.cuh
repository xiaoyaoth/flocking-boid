#ifndef GSIMAPP_BOID_H
#define GSIMAPP_BOID_H

#include "gsimcore.cuh"
#include "header.cuh"

enum BoidType {PREY_BOID, PREDATOR_BOID, FOOD_BOID};
enum BoidState{HUNGRY, NORMAL, FLEEING, STARVING, SEEKING_MATE};

class BoidModel : public GModel{
public:
	GRandomGen * rgen;
	float cohesion;
	float avoidance;
	float randomness;
	float consistency;
	float momentum;
	float deadFlockerProbability;
	float neighborhood;
	float jump;
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
	__device__ bool hungry();
	__device__ void eat(FoodBoid *food);
	__device__ bool starved();
	__device__ bool readyToMate();
	__device__ void setRandomSpeed();
	__device__ float2d_t consistency();
	__device__ float2d_t cohesion();
	__device__ float2d_t avoidance();
	__device__ float2d_t flee();
	__device__ float2d_t searchFood();
	__device__ float2d_t conformSpeed();
	__device__ float2d_t searchMate();
	__device__ float2d_t randomness(GRandomGen *gen);
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

	__device__ PredatorBoid(float2d_t loc){

	}
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

//BaseBoid
/*v*/__device__ float BaseBoid::getOrientation2D(){
	if (lastd.x = 0 && lastd.y == 0)
		return 0;
	return atan2(lastd.y, lastd.x);
}
/*v*/__device__ void BaseBoid::setOrientation2D(float val){
	lastd.x = cos(val);
	lastd.y = sin(val);
}
/*v*/__device__ float2d_t BaseBoid::momentum(){
	return lastd;
}

//FoodBoid
/*v*/__device__ void FoodBoid::reduce(){
	if (this->amount <= 0)
		this->dead = true;
	else {
		this->amount--;
		this->scale -= (this->scale/this->amount);
	}
}
/*v*/__device__ void FoodBoid::increase(){
	if (this->amount <= 0)
		this->dead = true;
	else if (this->amount < CONSTANT::FOOD_AMOUNT) {
		if (this->model->rgen->nextFloat() < 0.05){
			amount++;
			this->scale += (this->scale/this->amount);
		}
	}
}
/*v*/__device__ void FoodBoid::step(GModel *model){
	if(this->dead) {
		this->model->getWorld()->remove(this);
		return;
	}
	if(this->model->rgen->nextFloat() > 0.3)
		this->increase();
}

//PreyBoid
/*v*/__device__ bool PreyBoid::hungry(){
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
/*v*/__device__ void PreyBoid::eat(FoodBoid *food){
	food->reduce();
	this->hungerCount = 0;
	this->starveCount = 0;
	this->mateCount++;
	float hornyValue = 
		(float)(this->mateCount/CONSTANT::PREY_MATING_TIME);
	if (this->model->rgen->nextFloat() < hornyValue)
		this->horny = true;

}
/*v*/__device__ bool PreyBoid::starved(){
	return this->starveCount == this->STARVE_LIMIT;
}
/*v*/__device__ bool PreyBoid::readyToMate(){
	return this->horny;
}
/*v*/__device__ void PreyBoid::setRandomSpeed(){
	this->DEFAULT_SPEED = CONSTANT::PREY_STD_SPEED + 
		this->model->rgen->nextGaussian() * 0.2;

}
/*v*/__device__ float2d_t PreyBoid::consistency(){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;

	//iterInfo info;
	//Continuous2D *w = this->model->world;
	//int ptr = w->nextNeighborInit(this, this->model->neighborhood, &info);
	//while (ptr != -1){
	//	agIdx = w->neighborIdx[ptr];
	//	BaseBoid *other = (BaseBoid*)w->allAgents[agIdx];
	//	if(other->bstate == PREY_BOID){
	//		if(!other->dead){
	//			count++;
	//			float2d_t m = other->momentum();
	//			x += m.x;
	//			y += m.y;
	//		}
	//	}
	//	ptr = w->nextNeighbor(&info);
	//}
	//if (count > 0){
	//	x /= count;
	//	y /= count;
	//}
	return float2d_t(x,y);
}
/*v*/__device__ float2d_t PreyBoid::cohesion(){
	float x = 0;
	float y = 0;
	//int count = 0;
	//int agIdx = -1;

	//iterInfo info;
	//Continuous2D *w = this->model->world;
	//int ptr = w->nextNeighborInit(this, 200, &info);
	//while (ptr != -1){
	//	agIdx = w->neighborIdx[ptr];
	//	BaseBoid *other = (BaseBoid*)w->allAgents[agIdx];
	//	if (!other->dead){
	//		x += this->model->world->tdx(this->loc.x, other->loc.x);
	//		y += this->model->world->tdy(this->loc.y, other->loc.y);
	//		count++;
	//	}
	//	ptr = w->nextNeighbor(&info);
	//}
	//if (count > 0){
	//	x /= count;
	//	y /= count;
	//}
	return float2d_t(x,y);
}
__device__ float2d_t PreyBoid::avoidance(){return float2d_t(0,0);}
__device__ float2d_t PreyBoid::flee(){return float2d_t(0,0);}
__device__ float2d_t PreyBoid::searchFood(){return float2d_t(0,0);}
__device__ float2d_t PreyBoid::conformSpeed(){return float2d_t(0,0);}
__device__ float2d_t PreyBoid::searchMate(){return float2d_t(0,0);}
__device__ float2d_t PreyBoid::randomness(GRandomGen *gen){return float2d_t(0,0);}

__device__ float *randDebug;
#define STRIP 4
__device__ void PreyBoid::step(GModel *model){
	iterInfo info;
	Continuous2D *w = model->getWorld();
	NextNeighborControl nnc = w->nextNeighborInit(this, 200, info);
	while (nnc != STOP){
		nnc = w->nextNeighbor(info);
	}
	info.ptr = 0;
	//float xrand = model->rgen->nextFloat();
	//float yrand = model->rgen->nextFloat();
	//loc.x += (xrand-1)*info.count;
	//loc.y += (yrand-1)*info.count;
	//if(loc.x < 0)
	//	loc.x += BOARDER_R;
	//if(loc.y < 0)
	//	loc.y += BOARDER_D;
	//const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//randDebug[STRIP*idx] = xrand;
	//randDebug[STRIP*idx+1] = yrand;
	//randDebug[STRIP*idx+2] = loc.x;
	//randDebug[STRIP*idx+3] = loc.y;
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