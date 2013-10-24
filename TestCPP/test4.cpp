#include "cppheader.h"
#include <vector>
#include <algorithm>
#include <map>
#include <Windows.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>
#include <iomanip>

#define VERBOSE	 0
#define SORTMETHOD 2
#define TRIAL_NEIGHBOR 1
#define TRIAL_GEOAGENT 1
#define TRIAL_ZCODE 1

#define SELECTION 0
#define FILE_GEN 0

class GModel;
class GAgent;
class Continuous2D;
typedef struct GAgentData GAgentData_t;
typedef struct PreyBoidData PreyBoidData_t;

struct GAgentData{
	int ag_id;
	float2d_t loc;
	float2d_t lastd;
	bool dead;
	float2d_t momentum(){
		return this->lastd;
	}
};
struct PreyBoidData : public GAgentData{
	int extra1;
	float extra2;
};
union dataUnion{
	GAgentData_t agData;
	PreyBoidData_t preyData;
};

typedef struct iter_info_per_thread
{
	int2d_t cellCur;
	int2d_t cellUL;
	int2d_t cellDR;

	int ptr;
	int boarder;
	int count;
	const GAgentData_t *agData; //subject agent data
	float range;
} iterInfo;

iterInfo *infoArray;
int tid = 0;
int bid = 0;
int hit = 0;
int nonhit = 0;
float *randDebugArray;
const int STRIP = 5;
enum NextNeighborControl{CONTINUE, STOP, FOUND};

class GAgent{
public:
	GAgentData_t *data;
	GAgent *dummy;
	GModel *model;

	static int agIdCount;
	GAgent(){
		GAgent(0, 0);
	}
	GAgent(float2d_t loc){
		GAgent(loc.x, loc.y);
	}
	GAgent(float x, float y){
		this->data = new GAgentData_t();
		this->data->ag_id = agIdCount;
		this->data->loc.x = x;
		this->data->loc.y = y;
		this->data->dead = false;
		agIdCount++;
	}
	GAgent(GAgent *ag){ //prepared for creating the dummy;
		this->data = new GAgentData_t();
		this->data->ag_id = ag->data->ag_id;
		this->data->loc = ag->data->loc;
		this->data->dead = ag->data->dead;
		this->model = ag->model;
		this->dummy = ag;
	}
	float2d_t randomness(int idx);
	float2d_t consistency(const Continuous2D *world);
	float2d_t cohesion(const Continuous2D *world);
	float2d_t avoidance(const Continuous2D *world);
	void step(const GModel *model);
};
class GAgentDummy : public GAgent {
	int2d_t dummyAttr1;
	char dummyAttr2;
};
class Continuous2D{
public:
	GAgent **allAgents;
	int *neighborIdx, *cellIdx;
	float width;
	float height;
	float discretization;

	Continuous2D(float w, float h, float disc){
		this->width = w;
		this->height = h;
		this->discretization = disc;
	}
	void allocOnDevice();
	void allocOnHost();
	//distance utilities
	float tdx(float ax, float bx) const;
	float tdy(float ay, float by) const;
	float tds(float2d_t loc1, float2d_t loc2) const;
	float Continuous2D::stx(const float x) const;
	float Continuous2D::sty(const float y) const;
	//Neighbor Searching
	NextNeighborControl nextNeighborPrimitive(iterInfo &info) const;
	bool foundPrimitive(iterInfo &info) const;
	int boarderPrimitive(iterInfo &info) const;
	int ptrPrimitive(iterInfo &info) const;
#if TRIAL_NEIGHBOR == 0
	NextNeighborControl nextNeighborInit(const GAgent* ag, const int range, iterInfo &info) const;
	NextNeighborControl nextNeighbor(iterInfo &info) const;
	GAgent* obtainAgentByIterInfo(const int ptr) const;
#elif TRIAL_NEIGHBOR == 1
	NextNeighborControl nextNeighborInit2(const GAgent* ag, const int range) const;
	NextNeighborControl nextNeighbor2() const;
	const GAgentData_t *obtainAgentDataByIterInfo2() const; 
#endif
#if TRIAL_GEOAGENT == 1
	GAgent* obtainAgentByGeo(int i) const;
#endif
};
class GModel{
public:
	Continuous2D *world;
	float neighborhood;
	float cohesion;
	float avoidance;
	float randomness;
	float consistency;
	float momentum;
	float deadFlockerProbability;
	float jump;

	GModel(){
		cohesion = 1.0;
		avoidance = 1.0;
		randomness = 1.0;
		consistency = 1.0;
		momentum = 1.0;
		deadFlockerProbability = 0.1;
		neighborhood = 150;
		jump = 0.7;
	}

	void allocOnHost(){
		world = new Continuous2D(BOARDER_R, BOARDER_D, this->neighborhood/1.5);
		world->allocOnHost();
	}
};

int GAgent::agIdCount = 0;
float2d_t GAgent::randomness(int idx){
	float x = 1;
	float y = 1;
	//float x = randDebugArray[idx*STRIP];
	//float y = randDebugArray[idx*STRIP+1];
	float l = sqrt(x*x + y*y);
	float2d_t res;
	res.x = 0.05*x/l;
	res.y = 0.05*y/l;
	return res;
}
float2d_t GAgent::consistency(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;

#if TRIAL_NEIGHBOR == 0
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while (nnc != STOP){
		GAgent *other = (GAgent*)world->obtainAgentByIterInfo(info.ptr);
		if(!other->data->dead){
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
		GAgentData_t *other = (GAgentData_t*)world->obtainAgentDataByIterInfo2();
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
	randDebugArray[STRIP*this->data->ag_id+2] = infoArray[tid].count;
	float2d_t res;
	res.x = x;
	res.y = y;
	return res;
}
float2d_t GAgent::cohesion(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;

#if TRIAL_NEIGHBOR == 0
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while (nnc != STOP){
		GAgent *other = (GAgent*)world->obtainAgentByIterInfo(info.ptr);
		if (!other->data->dead){
			count++;
			x += world->tdx(this->data->loc.x, other->data->loc.x);
			y += world->tdy(this->data->loc.y, other->data->loc.y);
		}
		nnc = world->nextNeighbor(info);
	}
#else
	NextNeighborControl nnc = world->nextNeighborInit2(this, this->model->neighborhood);
	while (nnc != STOP){
		GAgentData_t *other = (GAgentData_t*)world->obtainAgentDataByIterInfo2();
		if (!other->dead){
			count++;
			x += world->tdx(this->data->loc.x, other->loc.x);
			y += world->tdy(this->data->loc.y, other->loc.y);
		}
		nnc = world->nextNeighbor2();
	}
#endif
	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->data->ag_id+3] = infoArray[tid].count;
	float2d_t res;
	res.x = -x/10;
	res.y = -y/10;
	return res;
}
float2d_t GAgent::avoidance(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;
#if TRIAL_NEIGHBOR == 0
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while(nnc != STOP){
		GAgent *other = (GAgent*)world->obtainAgentByIterInfo(info.ptr);
		if (!other->data->dead){
			count++;
			float dx = world->tdx(this->data->loc.x, other->data->loc.x);
			float dy = world->tdy(this->data->loc.y, other->data->loc.y);
			float sqrDist = dx*dx + dy*dy;
			x += dx/(sqrDist*sqrDist + 1);
			y += dy/(sqrDist*sqrDist + 1);
		}
		nnc = world->nextNeighbor(info);
	}
#else
	NextNeighborControl nnc = world->nextNeighborInit2(this, this->model->neighborhood);
	while(nnc != STOP){
		GAgentData_t *other = (GAgentData_t*)world->obtainAgentDataByIterInfo2();
		if (!other->dead){
			count++;
			float dx = world->tdx(this->data->loc.x, other->loc.x);
			float dy = world->tdy(this->data->loc.y, other->loc.y);
			float sqrDist = dx*dx + dy*dy;
			x += dx/(sqrDist*sqrDist + 1);
			y += dy/(sqrDist*sqrDist + 1);
		}
		nnc = world->nextNeighbor2();
	}
#endif
	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->data->ag_id+4] = infoArray[tid].count;
	float2d_t res;
	res.x = 400*x;
	res.y = 400*y;
	return res;
}
void GAgent::step(const GModel *model){
	const GModel *boidModel = model;
	if (this->data->dead)
		return;
	const Continuous2D *world = model->world;
	float2d_t avoid = this->avoidance(world);
	float2d_t cohes = this->cohesion(world);
	float2d_t consi = this->consistency(world);
	//float2d_t rdnes = this->randomness(model->rgen);
	float2d_t momen = this->data->momentum();
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
	this->dummy->data->lastd.x = dx;
	this->dummy->data->lastd.y = dy;
	this->dummy->data->loc.x =
		world->stx(this->data->loc.x + dx);
	this->dummy->data->loc.y =
		world->sty(this->data->loc.y + dy);
	randDebugArray[this->data->ag_id*STRIP] = this->dummy->data->loc.x;
	randDebugArray[this->data->ag_id*STRIP+1] = this->dummy->data->loc.y;
}

float Continuous2D::stx(const float x) const{
	if (x >= 0){
		if (x < this->width)
			return x;
		return x - this->width;
	}
	return x + this->width;
}
float Continuous2D::sty(const float y) const {
	if (y >= 0) {
		if (y < this->height)
			return y;
		return y - height;
	}
	return y + height;
}
float Continuous2D::tdx(float ax, float bx) const {
	float dx = abs(ax-bx);
	if (dx < BOARDER_R/2)
		return dx;
	else
		return BOARDER_R-dx;
}
float Continuous2D::tdy(float ay, float by) const {
	float dy = abs(ay-by);
	if (dy < BOARDER_D/2)
		return dy;
	else
		return BOARDER_D-dy;
}
float Continuous2D::tds(float2d_t loc1, float2d_t loc2) const {
	float dx = loc1.x - loc2.x;
	float dy = loc1.y - loc2.y;
	return sqrt(dx*dx + dy*dy);
}
void Continuous2D::allocOnHost(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);
	this->neighborIdx = (int*)malloc(sizeAgArray);
	this->cellIdx = (int*)malloc(sizeCellArray);
	this->allAgents = (GAgent**)malloc(AGENT_NO*sizeof(GAgent*));
}
int Continuous2D::boarderPrimitive(iterInfo &info) const{
#if TRIAL_ZCODE == 1
	int cellIdBoarder = info.cellCur.zCode();
	int ptrBoarder = cellIdx[cellIdBoarder];
	if (ptrBoarder == -1) //if last cell is empty
		return -1;
	else if (cellIdBoarder == CELL_NO-1) // if last cell if not empty
		return AGENT_NO-1;
	cellIdBoarder++;
	if (cellIdBoarder == CELL_NO-1)
		return AGENT_NO-1;
	ptrBoarder = cellIdx[cellIdBoarder];
	while(ptrBoarder == -1){
		cellIdBoarder++;
		if (cellIdBoarder == CELL_NO-1)
			return AGENT_NO-1;
		ptrBoarder = cellIdx[cellIdBoarder];
	}
	if (cellIdBoarder >= CELL_NO)
		printf("%d, %d\n", ptrBoarder, cellIdBoarder);
	return ptrBoarder-1;
#else
	int cellIdBoarder = info.cellCur.y * CNO_PER_DIM + info.cellDR.x + 1;
	int ptrBoarder = -1;
	if (cellIdBoarder < CELL_NO)
		ptrBoarder = cellIdx[cellIdBoarder];
	else
		return AGENT_NO-1;

	while (ptrBoarder == -1) {
		cellIdBoarder++;
		if (cellIdBoarder < CELL_NO)
			ptrBoarder = cellIdx[cellIdBoarder];
		else
			return AGENT_NO-1;
	}
	return ptrBoarder-1;
#endif;
}
int Continuous2D::ptrPrimitive(iterInfo &info) const{
#if TRIAL_ZCODE == 1
	int ptr = cellIdx[info.cellCur.zCode()];
	return ptr;
#else
	int ptr = cellIdx[info.cellCur.cell_id()];
	while (ptr == -1){
		info.cellCur.x++;
		if (info.cellCur.x > info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if (info.cellCur.y > info.cellDR.y)
				return -2;
		}
	ptr = cellIdx[info.cellCur.cell_id()];
	}
	return ptr;
#endif
}
bool Continuous2D::foundPrimitive(iterInfo &info) const{
	if (info.ptr < 0)
		return false;

	GAgent *other = this->allAgents[this->neighborIdx[info.ptr]];
	float ds = tds(info.agData->loc, other->data->loc);
	if (ds < info.range){
		info.count++;
		return true;
	}
	return false;
}
NextNeighborControl Continuous2D::nextNeighborPrimitive(iterInfo &info) const{
#if TRIAL_ZCODE == 1
	info.ptr++;
	if (info.ptr > info.boarder){
		info.cellCur.x++;
		if (info.cellCur.x > info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if (info.cellCur.y > info.cellDR.y)
				return STOP;
		}
		info.ptr = this->ptrPrimitive(info);
		info.boarder = this->boarderPrimitive(info);
	}
	return CONTINUE;
#else
	info.ptr++;
	if (info.ptr > info.boarder){
		info.cellCur.x = info.cellUL.x;
		info.cellCur.y++;
		if (info.cellCur.y <= info.cellDR.y){
			info.ptr = this->ptrPrimitive(info);
			info.boarder = this->boarderPrimitive(info);
		} else
			return STOP;
	}
	return CONTINUE;
#endif
}
#if TRIAL_NEIGHBOR == 0
NextNeighborControl Continuous2D::nextNeighbor(iterInfo &info)const {
	NextNeighborControl nnc = this->nextNeighborPrimitive(info);
	GAgent *other;
	float ds;
	while (nnc == CONTINUE){
		if (info.ptr < 0)
			return STOP;
		if (this->foundPrimitive(info))
			return FOUND;
		nnc = this->nextNeighborPrimitive(info);
	}
	return nnc;
}
NextNeighborControl Continuous2D::nextNeighborInit(const GAgent* ag, 
	const int range, iterInfo &info) const {
		float2d_t pos = ag->data->loc;
		info.agData = ag->data;
		info.ptr = -1;
		info.boarder = -1;
		info.count = 0;
		info.range = range;

		info.cellUL.x = (pos.x-range)>BOARDER_L ? 
			(int)(pos.x-range)/CLEN_X : (int)BOARDER_L/CLEN_X;
		info.cellDR.x = (pos.x+range)<BOARDER_R ? 
			(int)(pos.x+range)/CLEN_X : (int)BOARDER_R/CLEN_X - 1;
		info.cellUL.y = (pos.y-range)>BOARDER_U ? 
			(int)(pos.y-range)/CLEN_Y : (int)BOARDER_U/CLEN_Y;
		info.cellDR.y = (pos.y+range)<BOARDER_D ? 
			(int)(pos.y+range)/CLEN_Y : (int)BOARDER_D/CLEN_Y - 1;
		info.cellCur.x = info.cellUL.x;
		info.cellCur.y = info.cellUL.y;

		info.ptr = this->ptrPrimitive(info);
		info.boarder = this->boarderPrimitive(info);

		if (this->foundPrimitive(info))
			return FOUND;
		else
			return this->nextNeighbor(info);
}
GAgent* Continuous2D::obtainAgentByIterInfo(const int ptr) const{
	if (ptr<AGENT_NO && ptr>=0){
		const int agIdx = this->neighborIdx[ptr];
		return this->allAgents[agIdx];
	}
	return NULL;
}
#elif TRIAL_NEIGHBOR == 1
NextNeighborControl Continuous2D::nextNeighbor2() const {
	iterInfo &info = infoArray[tid];
	NextNeighborControl nnc = this->nextNeighborPrimitive(info);
	GAgent *other;
	float ds;
	while (nnc == CONTINUE){
		if (this->foundPrimitive(info))
			return FOUND;
		nnc = this->nextNeighborPrimitive(info);
	}
	return nnc;
}
NextNeighborControl Continuous2D::nextNeighborInit2(const GAgent* ag, 
	const int range) const {
		iterInfo &info = infoArray[tid];
		float2d_t pos = ag->data->loc;
		info.agData = ag->data;
		info.ptr = -1;
		info.boarder = -1;
		info.count = 0;
		info.range = range;

		info.cellUL.x = (pos.x-range)>BOARDER_L ? 
			(int)(pos.x-range)/CLEN_X : (int)BOARDER_L/CLEN_X;
		info.cellDR.x = (pos.x+range)<BOARDER_R ? 
			(int)(pos.x+range)/CLEN_X : (int)BOARDER_R/CLEN_X - 1;
		info.cellUL.y = (pos.y-range)>BOARDER_U ? 
			(int)(pos.y-range)/CLEN_Y : (int)BOARDER_U/CLEN_Y;
		info.cellDR.y = (pos.y+range)<BOARDER_D ? 
			(int)(pos.y+range)/CLEN_Y : (int)BOARDER_D/CLEN_Y - 1;
		info.cellCur.x = info.cellUL.x;
		info.cellCur.y = info.cellUL.y;

		info.ptr = this->ptrPrimitive(info);
		info.boarder = this->boarderPrimitive(info);

		if (this->foundPrimitive(info))
			return FOUND;
		else
			return this->nextNeighbor2();
}
const GAgentData_t *Continuous2D::obtainAgentDataByIterInfo2() const{
	int ptr = infoArray[tid].ptr;
	if (ptr>=bid*256 && ptr<(bid+1)*256){
		hit++;
		dataUnion *unionArray = (dataUnion*)&infoArray[AGENT_NO];
		dataUnion &dataElem = unionArray[ptr];
		return (GAgentData_t*)&dataElem;
	}
	if (ptr<AGENT_NO && ptr>=0){
		nonhit++;
		const int agIdx = this->neighborIdx[ptr];
		return this->allAgents[agIdx]->data;
	}
	return NULL;
}
#endif
#if TRIAL_GEOAGENT == 1
GAgent* Continuous2D::obtainAgentByGeo(int i) const{
	int agIdx = this->neighborIdx[i];
	return this->allAgents[agIdx];
}
#endif

void init(float *x_pos_h, float *y_pos_h){

	std::ifstream fin("pos_data.txt.10240");
	std::string rec;

	char *cstr, *p;
	int i = 0;
	cstr = (char *)malloc((rec.length() + 1) * sizeof(char));
	while (!fin.eof() && i<AGENT_NO) {
		std::getline(fin, rec);
		std::strcpy(cstr, rec.c_str());
		if(strcmp(cstr,"")==0)
			break;
		p=strtok(cstr, " ");
		x_pos_h[i] = atof(p);
		p=strtok(NULL, " ");
		y_pos_h[i] = atof(p);
		i++;
	}
}
int genZCode(int x, int y){
	x &= 0x0000ffff;                 // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	
	y &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
    y = (y ^ (y << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    y = (y ^ (y << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    y = (y ^ (y << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    y = (y ^ (y << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

	return x | (y << 1);
}
void genHash(int *hash, Continuous2D *c2d){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = c2d->allAgents[i];
		int idx = ag->data->ag_id;
		int xhash = (int)(ag->data->loc.x/CLEN_X);
		int yhash = (int)(ag->data->loc.y/CLEN_Y);
		//hash[idx] = (int)(ag->loc.x/CELL_RESO) + CNO_PER_DIM * (int)(ag->loc.y/CELL_RESO);
		hash[idx] = genZCode(xhash, yhash);
		c2d->neighborIdx[idx] = ag->data->ag_id;
	}
}
void sortHash1(int *hash, Continuous2D *c2d){
	struct compStruct{
		int key;
		float val;
		compStruct(int k, float v):key(k), val(v){}
		bool operator < (const compStruct& other) const{
			return key < other.key;
		}
	};
	std::vector <compStruct> vec;
	for (int i=0; i<AGENT_NO; i++)
		vec.push_back(compStruct(hash[i], c2d->neighborIdx[i]));
	std::sort(vec.begin(), vec.end());
	int marker = 0;
	c2d->cellIdx[0] = 0;
	for (int i=0; i<AGENT_NO; i++){
		if(vec[i].key != marker){
			marker = vec[i].key;
			c2d->cellIdx[marker] = i;
		}
		c2d->neighborIdx[i] = vec[i].val;
#if	VERBOSE == 1
		std::cout<<"("<<vec[i].key<<","<<vec[i].val<<")"<<" ";
	}
	std::cout<<std::endl;
	for (int i=0; i<CELL_NO; i++)
		std::cout<<c2d->cellIdx[i]<<" ";
	std::cout<<std::endl;
#else
	}
#endif
	vec.erase(vec.begin(), vec.end());
}
void sortHash2(int *hash, Continuous2D *c2d){
	//std::vector<int> hashMap[CELL_NO];
	std::vector <std::vector<int>> hashMap(CELL_NO);
	for(int i=0; i<AGENT_NO; i++)
		hashMap[hash[i]].push_back(c2d->neighborIdx[i]);
	int count = 0;

	for(int i=0; i<CELL_NO; i++){
		c2d->cellIdx[i] = -1;
		std::vector<int> seg = hashMap[i];
		if (!seg.empty()){
			c2d->cellIdx[i] = count;
			std::copy(seg.begin(), seg.end(), &c2d->neighborIdx[count]);
			count+= seg.size();
		}
	}
#if FILE_GEN == 1
	std::fstream fout;
	char *outfname = new char[20];
	sprintf(outfname, "cppout_%s.txt", "sortHash2_1");
	fout.open(outfname, std::ios::out);
	for(int i=0; i<AGENT_NO; i++){
		//GAgent *ag = c2d->allAgents[idx];
		int idx = c2d->neighborIdx[i];
		fout<<std::setw(5)
			<<i<<" "<<idx<<" "<<hash[idx]
			<<std::endl;
		fout.flush();
	}
	fout.close();
	sprintf(outfname, "cppout_%s.txt", "sortHash2_2");
	fout.open(outfname, std::ios::out);
	for(int i=0; i<CELL_NO; i++){
		fout<<std::setw(5)
			<<i<<" "<<c2d->cellIdx[i]
			<<std::endl;
		fout.flush();
	}
	fout.close();
#endif
}
void sortHash(int *hash, Continuous2D *c2d){
#if SORTMETHOD == 1
	sortHash1(hash, c2d);
#else
	sortHash2(hash, c2d);
#endif

}

void genNeighbor(Continuous2D *c2d){
	size_t agArraySize = AGENT_NO*sizeof(int);
	int *hash = (int*)malloc(agArraySize);
	genHash(hash, c2d);
	sortHash(hash, c2d);
}
void queryNeighbor(Continuous2D *c2d){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = c2d->allAgents[i];
		NextNeighborControl nnc;
#if TRIAL_NEIGHBOR == 0
		iterInfo info;
		nnc = c2d->nextNeighborInit(ag, 200, info);
		while(nnc != STOP){
			nnc = c2d->nextNeighbor(info);
		}
#endif
		//std::cout<<info.count<<" ";
		float rand1 = randDebugArray[i*STRIP];
		float rand2 = randDebugArray[i*STRIP+1];
		ag->dummy->data->loc.x += (rand1-1) * infoArray[tid].count;
		ag->dummy->data->loc.y += (rand2-1) * infoArray[tid].count;
		if(ag->dummy->data->loc.x < 0)
			ag->dummy->data->loc.x += BOARDER_R;
		if(ag->dummy->data->loc.y < 0)
			ag->dummy->data->loc.y += BOARDER_D;
		randDebugArray[i*STRIP] = infoArray[tid].count;
		randDebugArray[i*STRIP+1] = ag->dummy->data->loc.x;
		randDebugArray[i*STRIP+2] = ag->dummy->data->loc.y;
	}
	//std::cout<<std::endl;
}
void initRandDebugArray(std::string str1, std::string str2){
	std::istringstream buf1(str1);
	std::istringstream buf2(str2);
	std::istream_iterator<std::string> begin1(buf1), end1;
	std::istream_iterator<std::string> begin2(buf2), end2;
	std::vector<std::string> tokens1(begin1, end1); // done!
	std::vector<std::string> tokens2(begin2, end2); // done!
	for(int i=0; i<AGENT_NO; i++){
		randDebugArray[i*STRIP] = atof(tokens1[i].c_str());
		randDebugArray[i*STRIP+1] = atof(tokens2[i].c_str());
	}
}
void writeRandDebugArray(int i){
	if (i == SELECTION){
		std::fstream randDebugOut3;
		char *outfname = new char[10];
		sprintf(outfname, "cppout_%d.txt", i);
		randDebugOut3.open(outfname, std::ios::out);
		for(int j=0; j<AGENT_NO; j++){
			randDebugOut3
				<<std::setw(4)
				<<j<<"\t"
				<<randDebugArray[STRIP*j]<<"\t"
				<<randDebugArray[STRIP*j+1]<<"\t"
				<<randDebugArray[STRIP*j+2]<<"\t"
				<<randDebugArray[STRIP*j+3]<<"\t"
				<<randDebugArray[STRIP*j+4]<<"\t"
				<<std::endl;
			randDebugOut3.flush();
		}
		randDebugOut3.close();
		std::cout<<std::endl;
		std::cout<<"hit: "<<hit<<std::endl;
		std::cout<<"non-hit: "<<nonhit<<std::endl;
		system("PAUSE");
		exit(1);
	}
}

void swapDummy(Continuous2D *world){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = world->allAgents[i];
		world->allAgents[i] = ag->dummy;
	}
}
void setupSharedInfo(const GModel *model){
	for (int i=0; i<AGENT_NO; i++){
		GAgent *ag = model->world->obtainAgentByGeo(i);
		dataUnion *unionArray = (dataUnion*)&infoArray[AGENT_NO];
		unionArray[i].agData = *ag->data;
	}
}
void stepAllAgents(const GModel *model){
	for(int i=0; i<AGENT_NO; i++){
#if TRIAL_GEOAGENT == 0
		GAgent *ag = model->world->allAgents[i];
#elif TRIAL_GEOAGENT == 1
		GAgent *ag = model->world->obtainAgentByGeo(i);
#endif
		ag->step(model);
		tid++;
		bid = tid/256;
	}
	tid=0;
}

void test2() {
	printf("size of iterInfo: %d\n", sizeof(iterInfo));
	printf("size of dataUnion: %d\n", sizeof(dataUnion));
	randDebugArray = (float*)malloc(AGENT_NO*STRIP*sizeof(float));
	infoArray = (iterInfo*)malloc(AGENT_NO*sizeof(iterInfo)+AGENT_NO*sizeof(dataUnion));

	GModel *model = new GModel();
	model->allocOnHost();
	float *x_pos_h, *y_pos_h;
	x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	init(x_pos_h, y_pos_h);
	for(int i = 0; i<AGENT_NO; i++){
		GAgent *ag = new GAgent(x_pos_h[i], y_pos_h[i]);
		ag->model = model;
		GAgent *dummy = new GAgent(ag);
		ag->dummy = dummy;
		model->world->allAgents[i] = ag;
	}

	//	std::ifstream fin("randDebugOut2.txt");
	//	std::string str1;
	//	std::string str2;
	for(int i=0; i<200; i++){
		//printf("STEP: %d\n", i);
		genNeighbor(model->world);
		//std::getline(fin, str1);
		//std::getline(fin, str2);
		//initRandDebugArray(str1, str2);
		//queryNeighbor(model->world);
		setupSharedInfo(model);
		stepAllAgents(model);
		swapDummy(model->world);
		writeRandDebugArray(i);
	}

	
}
int main(){ 
	int start = GetTickCount();
	test2();
	int end = GetTickCount();
	int diff = end-start;
	std::cout<<"Took "<<diff<<" ms"<<std::endl;
	system("PAUSE");
	return 0;
	dataUnion *temp = (dataUnion*)&infoArray[0];
}
