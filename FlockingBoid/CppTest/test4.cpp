#include "cppheader.h"
#include <vector>
#include <algorithm>
#include <map>
#include <Windows.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>

#define VERBOSE	 0
#define SORTMETHOD 2

class GModel;
class GAgent;

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
	GAgent *dummy;
	static int agIdCount;
	GAgent(){
		this->ag_id = agIdCount;
		agIdCount++;
	}
	GAgent(float2d_t loc){
		this->ag_id = agIdCount;
		this->loc = loc;
		agIdCount++;
	}
	GAgent(float x, float y){
		this->ag_id = agIdCount;
		this->loc.x = x;
		this->loc.y = y;
		agIdCount++;
	}
	GAgent(GAgent *ag){ //prepared for creating the dummy;
		this->ag_id = ag->ag_id;
		this->loc = ag->loc;
		this->dummy = ag;
	}
	void step(GModel *model);
};
class Continuous2D{
public:
	GAgent **allAgents;
	int *neighborIdx, *cellIdx;
	void allocOnDevice();
	void allocOnHost();
	NextNeighborControl nextNeighborInit(const GAgent* ag, const int range, iterInfo &info);
	NextNeighborControl nextNeighbor(iterInfo &info);
	NextNeighborControl nextNeighborPrimitive(iterInfo &info);
	float tds(float2d_t loc1, float2d_t loc2){
		float dx = loc1.x - loc2.x;
		float dy = loc1.y - loc2.y;
		return sqrt(dx*dx + dy*dy);
	}
};
class GModel{
public:
	Continuous2D *world;
#ifdef __CUDACC__
	void allocOnDevice(){
		Continuous2D *world_d = new Continuous2D();
		world_d->allocOnDevice();
		cudaMalloc((void**)&world, sizeof(Continuous2D));
		cudaCheckErrors("GModel():cudaMalloc:world");
		cudaMemcpy(world, world_d, sizeof(Continuous2D), cudaMemcpyHostToDevice);
		cudaCheckErrors("GModel():cudaMemcpy:world");
	}
#endif
	void allocOnHost(){
		world = new Continuous2D();
		world->allocOnHost();
	}
};

int GAgent::agIdCount = 0;

NextNeighborControl Continuous2D::nextNeighborInit(const GAgent* ag, const int range, iterInfo &info){
	float2d_t pos = ag->loc;
	info.agent = ag;
	info.ptr = -1;
	info.boarder = -1;
	info.count = 0;
	info.range = range;
	info.cell_ul.x = (pos.x-range)>BOARDER_L ? (int)(pos.x-range)/CELL_RESO : (int)BOARDER_L/CELL_RESO;
	info.cell_dr.x = (pos.x+range)<BOARDER_R ? (int)(pos.x+range)/CELL_RESO : (int)BOARDER_R/CELL_RESO-1;
	info.cell_ul.y = (pos.y-range)>BOARDER_U ? (int)(pos.y-range)/CELL_RESO : (int)BOARDER_U/CELL_RESO;
	info.cell_dr.y = (pos.y+range)<BOARDER_D ? (int)(pos.y+range)/CELL_RESO : (int)BOARDER_D/CELL_RESO-1;
	info.cell_cur.x = info.cell_ul.x;
	info.cell_cur.y = info.cell_ul.y;
	info.ptr = cellIdx[info.cell_cur.cell_id()];
	if (info.cell_cur.cell_id() == CELL_NO-1)
		info.boarder = AGENT_NO-1;
	else
		info.boarder = cellIdx[info.cell_cur.cell_id()+1];

	GAgent *other = this->allAgents[this->neighborIdx[info.ptr]];
	float ds = tds(ag->loc, other->loc);
	if (info.agent->ag_id == 576)
		printf("%d %f\n", other->ag_id, ds);
	if (ds < range){
		info.count++;
		return FOUND;
	} else
		return this->nextNeighbor(info);
}
NextNeighborControl Continuous2D::nextNeighborPrimitive(iterInfo &info){
	info.ptr++;

	if (info.ptr >= info.boarder) {
		info.cell_cur.x++;
		//if (info.cell_cur.x <= info.cell_dr.x) {
		//	info.ptr = cellIdx[info.cell_cur.cell_id()];
		//} else {
		if (info.cell_cur.x > info.cell_dr.x) {
			info.cell_cur.x = info.cell_ul.x;
			info.cell_cur.y++;
			if(info.cell_cur.y <= info.cell_dr.y)
				info.ptr = cellIdx[info.cell_cur.cell_id()];
			else
				return STOP;
		}
		if (info.cell_cur.cell_id() == CELL_NO-1)
			info.boarder = AGENT_NO-1;
		else
			info.boarder = cellIdx[info.cell_cur.cell_id()+1];
	}
	//info.print();
	return CONTINUE;
}
NextNeighborControl Continuous2D::nextNeighbor(iterInfo &info){
	NextNeighborControl nnc = this->nextNeighborPrimitive(info);
	GAgent *other;
	float ds;
	while (nnc == CONTINUE){
		other = this->allAgents[this->neighborIdx[info.ptr]];
		ds = tds(info.agent->loc, other->loc);
		if (info.agent->ag_id == 576)
			printf("%d %f\n", other->ag_id, ds);
		if (ds < info.range){
			info.count++;
			return FOUND;
		}
		nnc = this->nextNeighborPrimitive(info);
	}
	return nnc;
}

#ifdef __CUDACC__
void Continuous2D::allocOnDevice(){
	size_t sizeAgArray = AGENT_NO*sizeof(int);
	size_t sizeCellArray = CELL_NO*sizeof(int);

	cudaMalloc((void**)&neighborIdx, sizeAgArray);
	cudaCheckErrors("Continuous2D():cudaMalloc:neighborIdx");
	cudaMalloc((void**)&cellIdx, sizeCellArray);
	cudaCheckErrors("Continuous2D():cudaMalloc:cellIdx");


	int *neighborIdx_h = (int*)malloc(sizeAgArray);
	int *cellIdx_h = (int*)malloc(sizeCellArray);
	for(int i=0; i<AGENT_NO; i++)
		neighborIdx_h[i] = 2;
	for(int i = 0; i<CELL_NO; i++)
		cellIdx_h[i] = 3;
	cudaMemcpy(this->neighborIdx, neighborIdx_h, sizeAgArray, cudaMemcpyHostToDevice);
	cudaCheckErrors("Continuous2D():cudaMemcpy:neighborIdx");
	cudaMemcpy(this->cellIdx, cellIdx_h, sizeCellArray, cudaMemcpyHostToDevice);
	cudaCheckErrors("Continuous2D():cudaMemcpy:cellIdx");
}
#endif
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
		hash[idx] = (int)(ag->loc.x/100) + 10 * (int)(ag->loc.y/100);
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
	//std::vector <std::vector<int>> hashMap;
	for(int i=0; i<AGENT_NO; i++)
		hashMap[hash[i]].push_back(c2d->neighborIdx[i]);
	int count = 0;
	c2d->cellIdx[0]=0;
	for(int i=0; i<CELL_NO; i++){
		std::vector<int> seg = hashMap[i];
		std::copy(seg.begin(), seg.end(), &c2d->neighborIdx[count]);
		count+= seg.size();
		if(i+1<CELL_NO)
			c2d->cellIdx[i+1]=count;
	}
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

float *randDebugArray;
int strip = 2;
void queryNeighbor(Continuous2D *c2d, std::string str1, std::string str2){
	std::istringstream buf1(str1);
	std::istringstream buf2(str2);
	std::istream_iterator<std::string> begin1(buf1), end1;
	std::istream_iterator<std::string> begin2(buf2), end2;
	std::vector<std::string> tokens1(begin1, end1); // done!
	std::vector<std::string> tokens2(begin2, end2); // done!

	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = c2d->allAgents[i];
		iterInfo info;
		NextNeighborControl nnc;
		nnc = c2d->nextNeighborInit(ag, 200, info);
		while(nnc != STOP){
			nnc = c2d->nextNeighbor(info);
		}
		//std::cout<<info.count<<" ";
		float rand1 = atof(tokens1[i].c_str());
		float rand2 = atof(tokens2[i].c_str());
		ag->dummy->loc.x += (rand1-1) * info.count;
		ag->dummy->loc.y += (rand2-1) * info.count;
		if(ag->dummy->loc.x < 0)
			ag->dummy->loc.x += BOARDER_R;
		if(ag->dummy->loc.y < 0)
			ag->dummy->loc.y += BOARDER_D;
		randDebugArray[i*strip] = info.count;
		randDebugArray[i*strip+1] = ag->loc.y;
	}
	//std::cout<<std::endl;
}
void swapDummy(Continuous2D *world){
	for(int i=0; i<AGENT_NO; i++){
		GAgent *ag = world->allAgents[i];
		GAgent *dummy = ag->dummy;
		world->allAgents[i] = dummy;
		dummy->dummy = ag;
	}
}
void test2() {
	GModel *model = new GModel();
	model->allocOnHost();
	float *x_pos_h, *y_pos_h;
	x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	init(x_pos_h, y_pos_h);
	for(int i = 0; i<AGENT_NO; i++){
		GAgent *ag = new GAgent(x_pos_h[i], y_pos_h[i]);
		GAgent *dummy = new GAgent(ag);
		ag->dummy = dummy;
		model->world->allAgents[i] = ag;
	}

	std::ifstream fin("randDebugOut2.txt");
	std::string str1;
	std::string str2;
	std::fstream randDebugOut3;
#if SORTMETHOD == 1
	randDebugOut3.open("randDebugOut3.txt", std::ios::out);
#else
	randDebugOut3.open("randDebugOut4.txt", std::ios::out);
#endif
	for(int i=0; i<3; i++){
		genNeighbor(model->world);
		std::getline(fin, str1);
		std::getline(fin, str2);
		queryNeighbor(model->world, str1, str2);
		swapDummy(model->world);
		for(int j=0; j<AGENT_NO; j++){
			randDebugOut3
				<<randDebugArray[strip*j]<<"\t"
				<<std::endl;
			randDebugOut3.flush();
		}
		//randDebugOut3<<std::endl;
		//std::cout<<i<<" ";
	}
	randDebugOut3.close();
	//std::cout<<std::endl;
}
int test3(){
	std::ifstream fin("randDebugOut2.txt");
	std::string str;
	std::getline(fin, str);
	std::istringstream buf(str);
	std::istream_iterator<std::string> beg(buf), end;
	std::vector<std::string> tokens(beg, end); // done!
	for(int i=0; i<1024; i++)
		std::cout<<tokens[i]<<" ";
	std::cout<<std::endl;
	return 0;
}
int main(){
	randDebugArray = (float*)malloc(AGENT_NO*strip*sizeof(float));
	int start = GetTickCount();
	test2();
	int end = GetTickCount();
	int diff = end-start;
	std::cout<<"Took "<<diff<<" ms"<<std::endl;
	system("PAUSE");
	return 0;
}
