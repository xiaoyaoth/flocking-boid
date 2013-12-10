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

#define SELECTION 10
#define FILE_GEN 1

class GModel;
class GAgent;
class Continuous2D;

typedef struct iter_info_per_thread
{
	int2d_t cellCur;
	int2d_t cellUL;
	int2d_t cellDR;

	int ptr;
	int boarder;
	int count;
	int ag_id;
	float2d_t loc;
	const GAgent* ag;
	float range;
} iterInfo;

int tid = 0;
int bid = 0;
int hit = 0;
int nonhit = 0;
float *randDebugArray;
const int STRIP = 5;

class GAgent{
public:
	int ag_id;
	float2d_t loc;
	float2d_t lastd;
	bool dead;
	float2d_t momentum(){
		return this->lastd;
	}
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
		this->ag_id = agIdCount;
		this->loc.x = x;
		this->loc.y = y;
		this->dead = false;
		agIdCount++;
	}
	GAgent(GAgent *ag){ //prepared for creating the dummy;
		this->ag_id = ag->ag_id;
		this->loc = ag->loc;
		this->dead = ag->dead;
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
	int *neighborIdx;
	int *cellIdxStart;
	int *cellIdxEnd;
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
	void calcPtrAndBoarder(iterInfo &info) const;
	void nextNeighborInit(const GAgent* ag, const float range, iterInfo &info) const;
	GAgent* nextNeighbor(iterInfo &info) const;
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
		cohesion = 1.0f;
		avoidance = 1.0f;
		randomness = 1.0f;
		consistency = 1.0f;
		momentum = 1.0f;
		deadFlockerProbability = 0.1f;
		neighborhood = 150;
		jump = 0.7f;
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
	res.x = 0.05f*x/l;
	res.y = 0.05f*y/l;
	return res;
}
float2d_t GAgent::consistency(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	float ds = 0;;

	iterInfo info;
	world->nextNeighborInit(this, 150, info);
	GAgent *other = world->nextNeighbor(info);
	while(other != NULL){
		ds = world->tds(this->loc, other->loc);
		if (ds < 150) {
			if(!other->dead){
				count++;
				float2d_t m = other->momentum();
				x += m.x;
				y += m.y;
			}
		}
		other = world->nextNeighbor(info);
	}

	if (count > 0){
		x /= count;
		y /= count;
	}

	randDebugArray[STRIP*this->ag_id+2] = count;
	float2d_t res;
	res.x = x;
	res.y = y;
	return res;
}
float2d_t GAgent::cohesion(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	float ds = 0;;

	iterInfo info;
	world->nextNeighborInit(this, 150, info);
	GAgent *other = world->nextNeighbor(info);
	while (other != NULL){
		ds = world->tds(this->loc, other->loc);
		if (ds < 150) {
		if (!other->dead){
			count++;
			x += world->tdx(this->loc.x, other->loc.x);
			y += world->tdy(this->loc.y, other->loc.y);
		}
		}
		other = world->nextNeighbor(info);
	}

	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->ag_id+3] = count;
	float2d_t res;
	res.x = -x/10;
	res.y = -y/10;
	return res;
}
float2d_t GAgent::avoidance(const Continuous2D *world){
	float x = 0;
	float y = 0;
	int count = 0;
	float ds = 0;;

	iterInfo info;
	world->nextNeighborInit(this, 150, info);
	GAgent *other = world->nextNeighbor(info);
	while(other != NULL){
		ds = world->tds(this->loc, other->loc);
		if (ds < 150) {
			if (!other->dead){
				count++;
				float dx = world->tdx(this->loc.x, other->loc.x);
				float dy = world->tdy(this->loc.y, other->loc.y);
				float sqrDist = dx*dx + dy*dy;
				x += dx/(sqrDist*sqrDist + 1);
				y += dy/(sqrDist*sqrDist + 1);
			}
		}
		other = world->nextNeighbor(info);
	}
	if (count > 0){
		x /= count;
		y /= count;
	}
	randDebugArray[STRIP*this->ag_id+4] = count;
	float2d_t res;
	res.x = 400*x;
	res.y = 400*y;
	return res;
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
	this->dummy->lastd.x = dx;
	this->dummy->lastd.y = dy;
	this->dummy->loc.x =
		world->stx(this->loc.x + dx);
	this->dummy->loc.y =
		world->sty(this->loc.y + dy);
	randDebugArray[this->ag_id*STRIP] = this->dummy->loc.x;
	randDebugArray[this->ag_id*STRIP+1] = this->dummy->loc.y;
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
	this->cellIdxStart = (int*)malloc(sizeCellArray);
	this->cellIdxEnd = (int*)malloc(sizeCellArray);
	this->allAgents = (GAgent**)malloc(AGENT_NO*sizeof(GAgent*));
}

void Continuous2D::calcPtrAndBoarder(iterInfo &info) const{
	int hash = info.cellCur.cell_id();
	if (hash < CELL_NO && hash >= 0) {
		info.ptr = this->cellIdxStart[hash];
		info.boarder = this->cellIdxEnd[hash];
	} else
		printf("calcPtrAndBoarder\n");
}
void Continuous2D::nextNeighborInit(const GAgent* ag,  const float range, iterInfo &info) const {
	float2d_t pos = ag->loc;
	info.ag = ag;
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

	this->calcPtrAndBoarder(info);
}
GAgent* Continuous2D::nextNeighbor(iterInfo &info) const{
	if (info.ptr > info.boarder){
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	while (info.ptr == -1) {
		info.cellCur.x++;
		if(info.cellCur.x>info.cellDR.x){
			info.cellCur.x = info.cellUL.x;
			info.cellCur.y++;
			if(info.cellCur.y>info.cellDR.y)
				return NULL;
		}
		this->calcPtrAndBoarder(info);
	}

	int agIdx = this->neighborIdx[info.ptr];
	GAgent *ag = this->allAgents[agIdx];
	info.ptr++;
	return ag;
}

void init(float *x_pos_h, float *y_pos_h){
	std::ifstream fin(DATA_FILENAME);
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
		int xhash = (int)(ag->loc.x/CLEN_X);
		int yhash = (int)(ag->loc.y/CLEN_Y);
		hash[idx] = (int)(ag->loc.x/CLEN_X) + CNO_PER_DIM * (int)(ag->loc.y/CLEN_Y);
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
	c2d->cellIdxStart[0] = 0;
	for (int i=0; i<AGENT_NO; i++){
		if(vec[i].key != marker){
			marker = vec[i].key;
			c2d->cellIdxStart[marker] = i;
		}
		c2d->neighborIdx[i] = vec[i].val;
#if	VERBOSE == 1
		std::cout<<"("<<vec[i].key<<","<<vec[i].val<<")"<<" ";
	}
	std::cout<<std::endl;
	for (int i=0; i<CELL_NO; i++)
		std::cout<<c2d->cellIdxStart[i]<<" ";
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
		c2d->cellIdxStart[i] = -1;
		c2d->cellIdxEnd[i] = -1;
		std::vector<int> seg = hashMap[i];
		if (!seg.empty()){
			c2d->cellIdxStart[i] = count;
			c2d->cellIdxEnd[i] = count+seg.size()-1;
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
			<<i<<" "<<c2d->cellIdxStart[i]
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
	if (i == SELECTION && FILE_GEN == 1){
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
void stepAllAgents(const GModel *model){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = model->world->allAgents[i];

		ag->step(model);
		tid++;
		bid = tid/256;
	}
	tid=0;
}

void test2() {
	printf("size of iterInfo: %d\n", sizeof(iterInfo));
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

	for(int i=0; i<100; i++){
		printf("STEP: %d\n", i);
		genNeighbor(model->world);
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
}
