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

float *randDebugArray;
const int STRIP = 5;

class GModel;
class GAgent;
class Continuous2D;

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
} iterInfo;
enum NextNeighborControl{CONTINUE, STOP, FOUND};

class GAgent{
public:
	int ag_id;
	float2d_t loc;
	float2d_t lastd;
	GAgent *dummy;
	GModel *model;
	bool dead;

	static int agIdCount;
	GAgent(){
		GAgent(0, 0);
	}
	GAgent(float2d_t loc){
		GAgent(loc.x, loc.y);
	}
	GAgent(float x, float y){
		this->ag_id = agIdCount;
		this->loc.x = x;
		this->loc.y = y;
		this->dead = false;
		agIdCount++;
	}
	GAgent(GAgent *ag){ //prepared for creating the dummy;
		this->ag_id = ag->ag_id;
		this->loc = ag->loc;
		this->model = ag->model;
		this->dead = ag->dead;
		this->dummy = ag;
	}
	float2d_t momentum();
	float2d_t randomness(int idx);
	float2d_t consistency(const Continuous2D *world);
	float2d_t cohesion(const Continuous2D *world);
	float2d_t avoidance(const Continuous2D *world);
	void step(const GModel *model);

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
	float tdx(float ax, float bx) const;
	float tdy(float ay, float by) const;
	float tds(float2d_t loc1, float2d_t loc2) const;

	NextNeighborControl nextNeighborInit(const GAgent* ag, const int range, iterInfo &info) const;
	NextNeighborControl nextNeighbor(iterInfo &info) const;
	NextNeighborControl nextNeighborPrimitive(iterInfo &info) const;
	int boarderPrimitive(iterInfo &info) const;
	int ptrPrimitive(iterInfo &info) const;

	float Continuous2D::stx(const float x) const;
	float Continuous2D::sty(const float y) const;

	GAgent* obtainAgentFromNeighborIdx(const int ptr) const;
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
float2d_t GAgent::momentum(){
	return lastd;
}
float2d_t GAgent::randomness(int idx){
	float x = 1;
	float y = 1;
	//float x = randDebugArray[idx*STRIP];
	//float y = randDebugArray[idx*STRIP+1];
	float l = sqrt(x*x + y*y);
	return float2d_t(0.05*x/l, 0.05*y/l);
}
float2d_t GAgent::consistency(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;

	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while (nnc != STOP){
		GAgent *other = (GAgent*)world->obtainAgentFromNeighborIdx(info.ptr);
		if(!other->dead){
			count++;
			float2d_t m = other->momentum();
			x += m.x;
			y += m.y;
		}
		nnc = world->nextNeighbor(info);
	}
	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->ag_id+2] = info.count;
	return float2d_t(x,y);
}
float2d_t GAgent::cohesion(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;

	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while (nnc != STOP){
		GAgent *other = (GAgent*)world->obtainAgentFromNeighborIdx(info.ptr);
		if (!other->dead){
			count++;
			x += world->tdx(this->loc.x, other->loc.x);
			y += world->tdy(this->loc.y, other->loc.y);
		}
		nnc = world->nextNeighbor(info);
	}
	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->ag_id+3] = info.count;
	return float2d_t(-x/10,-y/10);
}
float2d_t GAgent::avoidance(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	int agIdx = -1;
	iterInfo info;
	NextNeighborControl nnc = world->nextNeighborInit(this, this->model->neighborhood, info);
	while(nnc != STOP){
		GAgent *other = (GAgent*)world->obtainAgentFromNeighborIdx(info.ptr);
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
	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->ag_id+4] = info.count;
	return float2d_t(400*x, 400*y);
}
void GAgent::step(const GModel *model){
	const GModel *boidModel = model;
	if (this->dead)
		return;
	const Continuous2D *world = model->world;
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
	this->dummy->lastd = float2d_t(dx, dy);
	this->dummy->loc = float2d_t(
		world->stx(loc.x + dx),
		world->sty(loc.y + dy)
		);
	randDebugArray[this->ag_id*STRIP] = this->dummy->loc.x;
	randDebugArray[this->ag_id*STRIP+1] = this->dummy->loc.y;
}

int Continuous2D::boarderPrimitive(iterInfo &info) const{
	int cellIdBoarder = info.cell_cur.y * XLENGTH + info.cell_dr.x + 1;
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
}

int Continuous2D::ptrPrimitive(iterInfo &info) const{
	int ptr = cellIdx[info.cell_cur.cell_id()];
	while (ptr == -1){
		info.cell_cur.x++;
		if (info.cell_cur.x > info.cell_dr.x){
			info.cell_cur.x = info.cell_ul.x;
			info.cell_cur.y++;
			if (info.cell_cur.y > info.cell_dr.y)
				return -1;
		}
		ptr = cellIdx[info.cell_cur.cell_id()];
	}
	return ptr;
}

NextNeighborControl Continuous2D::nextNeighborPrimitive(iterInfo &info) const{
	info.ptr++;
	if (info.ptr > info.boarder){
		info.cell_cur.x = info.cell_ul.x;
		info.cell_cur.y++;
		if (info.cell_cur.y <= info.cell_dr.y){
			info.ptr = this->ptrPrimitive(info);
			info.boarder = this->boarderPrimitive(info);
		} else
			return STOP;
	}
	return CONTINUE;
}

NextNeighborControl Continuous2D::nextNeighbor(iterInfo &info)const {
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

NextNeighborControl Continuous2D::nextNeighborInit(const GAgent* ag, 
	const int range, iterInfo &info) const {
		float2d_t pos = ag->loc;
		info.agent = ag;
		info.ptr = -1;
		info.boarder = -1;
		info.count = 0;
		info.range = range;

		info.cell_ul.x = (pos.x-range)>BOARDER_L ? 
			(int)(pos.x-range)/CELL_RESO : (int)BOARDER_L/CELL_RESO;
		info.cell_dr.x = (pos.x+range)<BOARDER_R ? 
			(int)(pos.x+range)/CELL_RESO : (int)BOARDER_R/CELL_RESO - 1;
		info.cell_ul.y = (pos.y-range)>BOARDER_U ? 
			(int)(pos.y-range)/CELL_RESO : (int)BOARDER_U/CELL_RESO;
		info.cell_dr.y = (pos.y+range)<BOARDER_D ? 
			(int)(pos.y+range)/CELL_RESO : (int)BOARDER_D/CELL_RESO - 1;
		info.cell_cur.x = info.cell_ul.x;
		info.cell_cur.y = info.cell_ul.y;

		info.ptr = this->ptrPrimitive(info);
		info.boarder = this->boarderPrimitive(info);

		GAgent *other = this->allAgents[this->neighborIdx[info.ptr]];
		float ds = tds(pos, other->loc);
		if (ds < range){
			info.count++;
			return FOUND;
		} else
			return this->nextNeighbor(info);
}

GAgent* Continuous2D::obtainAgentFromNeighborIdx(const int ptr) const{
	if (ptr<AGENT_NO && ptr>=0){
		const int agIdx = this->neighborIdx[ptr];
		return this->allAgents[agIdx];
	}
	return NULL;
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
void genHash(int *hash, Continuous2D *c2d){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = c2d->allAgents[i];
		int idx = ag->ag_id;
		hash[idx] = (int)(ag->loc.x/CELL_RESO) + XLENGTH * (int)(ag->loc.y/CELL_RESO);
		c2d->neighborIdx[idx] = ag->ag_id;
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
	std::vector<int> hashMap[CELL_NO];
	//std::vector <std::vector<int>> hashMap(CELL_NO);
	for(int i=0; i<AGENT_NO; i++)
		hashMap[hash[i]].push_back(c2d->neighborIdx[i]);

	int count = 0;
	
	std::fstream fout;
	char *outfname = new char[10];
	sprintf(outfname, "cppout_%s.txt", "sortHash2");
	fout.open(outfname, std::ios::out);

	for(int i=0; i<CELL_NO; i++){
		c2d->cellIdx[i] = -1;
		std::vector<int> seg = hashMap[i];
		if (!seg.empty()){
			c2d->cellIdx[i] = count;
			std::copy(seg.begin(), seg.end(), &c2d->neighborIdx[count]);
			count+= seg.size();
		}
		fout<<c2d->cellIdx[i]<<std::endl;
		fout.flush();
	}
	fout.close();
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
#if VERBOSE == 1
	std::cout<<std::endl;
#endif
}

void queryNeighbor(Continuous2D *c2d){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = c2d->allAgents[i];
		iterInfo info;
		NextNeighborControl nnc;
		nnc = c2d->nextNeighborInit(ag, 200, info);
		while(nnc != STOP){
			nnc = c2d->nextNeighbor(info);
		}
		//std::cout<<info.count<<" ";
		float rand1 = randDebugArray[i*STRIP];
		float rand2 = randDebugArray[i*STRIP+1];
		ag->dummy->loc.x += (rand1-1) * info.count;
		ag->dummy->loc.y += (rand2-1) * info.count;
		if(ag->dummy->loc.x < 0)
			ag->dummy->loc.x += BOARDER_R;
		if(ag->dummy->loc.y < 0)
			ag->dummy->loc.y += BOARDER_D;
		randDebugArray[i*STRIP] = info.count;
		randDebugArray[i*STRIP+1] = ag->dummy->loc.x;
		randDebugArray[i*STRIP+2] = ag->dummy->loc.y;
	}
	//std::cout<<std::endl;
}
void stepAllAgents(const GModel *model){

	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = model->world->allAgents[i];
		ag->step(model);
	}
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
void swapDummy(Continuous2D *world){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = world->allAgents[i];
		world->allAgents[i] = ag->dummy;
	}
}
void test2() {
	randDebugArray = (float*)malloc(AGENT_NO*STRIP*sizeof(float));

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
	std::fstream randDebugOut3;
	for(int i=0; i<200; i++){
		//printf("STEP: %d\n", i);
		genNeighbor(model->world);
		//std::getline(fin, str1);
		//std::getline(fin, str2);
		//initRandDebugArray(str1, str2);
		//queryNeighbor(model->world);
		stepAllAgents(model);
		swapDummy(model->world);

		if (i == 0 ){
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
			exit(1);
		}
		//randDebugOut3<<std::endl;
		//std::cout<<i<<" ";
	}
	//randDebugOut3.close();
	//std::cout<<std::endl;
}

int main(){
	int start = GetTickCount();
	test2();
	int end = GetTickCount();
	int diff = end-start;
	std::cout<<"Took "<<diff<<" ms"<<std::endl;
	system("PAUSE");
	return 0;
}
