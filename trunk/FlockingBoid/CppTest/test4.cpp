#include "cppheader.h"
#include <vector>
#include <algorithm>
#include <map>
#include <Windows.h>

#define VERBOSE	 0

typedef struct iter_info_per_thread
{
	int2d_t cell_cur;
	int2d_t cell_ul;
	int2d_t cell_dr;

	int ag_id;
	int ptr;
	int boarder;
	int count;

	void print(){
		printf("======iter info======\n");
		printf("agent_id: %d", ag_id);
		printf("cell_cur: ");
		cell_cur.print();
		printf("\ncell_ul: ");
		cell_ul.print();
		printf("\ncell_dr: ");
		cell_dr.print();
		printf("\nptr: %d\ncellIdx_border: %d\n", ptr, boarder);
		printf("=====================\n");
	}
} iterInfo;

class GModel;
class GAgent{
public:
	int ag_id;
	float2d_t loc;
	static int agIdCount;
	GAgent(){
		this->ag_id = agIdCount;
		agIdCount++;
	}
	void step(GModel *model);
};
class Continuous2D{
public:
	GAgent **allAgents;
	int *neighborIdx, *cellIdx;
	void allocOnDevice();
	void allocOnHost();
	int nextNeighborInit(const GAgent* ag, const int range, iterInfo *info);
	int nextNeighbor(iterInfo *info);
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

int Continuous2D::nextNeighborInit(const GAgent* ag, const int range, iterInfo *info){
	float2d_t pos = ag->loc;
	info->ptr = -1;
	info->boarder = -1;
	info->count = 0;
	info->cell_ul.x = (pos.x-range)>BOARDER_L ? (int)(pos.x-range)/CELL_RESO : (int)BOARDER_L/CELL_RESO;
	info->cell_dr.x = (pos.x+range)<BOARDER_R ? (int)(pos.x+range)/CELL_RESO : (int)BOARDER_R/CELL_RESO-1;
	info->cell_ul.y = (pos.y-range)>BOARDER_U ? (int)(pos.y-range)/CELL_RESO : (int)BOARDER_U/CELL_RESO;
	info->cell_dr.y = (pos.y+range)<BOARDER_D ? (int)(pos.y+range)/CELL_RESO : (int)BOARDER_D/CELL_RESO-1;
	info->cell_cur.x = info->cell_ul.x;
	info->cell_cur.y = info->cell_ul.y;
	info->ptr = cellIdx[info->cell_cur.cell_id()];
	if (info->cell_cur.cell_id() == CELL_NO-1)
		info->boarder = AGENT_NO-1;
	else
		info->boarder = cellIdx[info->cell_cur.cell_id()+1];
	//info->print();
	info->count++;
	return info->ptr;
}
int Continuous2D::nextNeighbor(iterInfo *info){
	info->ptr++;
	if (info->ptr > info->boarder) {
		info->cell_cur.x++;
		if (info->cell_cur.x <= info->cell_dr.x)
			info->ptr = cellIdx[info->cell_cur.cell_id()];
		else {
			info->cell_cur.x = info->cell_ul.x;
			info->cell_cur.y++;
			if(info->cell_cur.y <= info->cell_dr.y)
				info->ptr = cellIdx[info->cell_cur.cell_id()];
			else
				return -1;
		}
		if (info->cell_cur.cell_id() == CELL_NO-1)
			info->boarder = AGENT_NO-1;
		else
			info->boarder = cellIdx[info->cell_cur.cell_id()+1];
	}
	//info->print();
	info->count++;
	return info->ptr;
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
	sortHash1(hash, c2d);
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
		int ptr = c2d->nextNeighborInit(ag, 200, &info);
		while(ptr != -1){
			ptr = c2d->nextNeighbor(&info);
		}
		//std::cout<<info.count<<" ";
	}
	//std::cout<<std::endl;
}

void test2() {
	GModel *model = new GModel();
	model->allocOnHost();
	float *x_pos_h, *y_pos_h;
	x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	init(x_pos_h, y_pos_h);
	for(int i = 0; i<AGENT_NO; i++){
		GAgent *ag = new GAgent();
		ag->loc.x = x_pos_h[i];
		ag->loc.y = y_pos_h[i];
		model->world->allAgents[i] = ag;
	}
	for(int i=0; i<1000; i++){
		genNeighbor(model->world);
		queryNeighbor(model->world);
		//std::cout<<i<<" ";
	}
	//std::cout<<std::endl;
}

int main4(){
	int start = GetTickCount();
	test2();
	int end = GetTickCount();
	int diff = end-start;
	std::cout<<"Took "<<diff<<" ms"<<std::endl;
	std::cin>>diff;
	return 0;
}
