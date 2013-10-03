#include "gsimcore.cuh"
#include "gsimapp_boid.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>

__global__ void seeAllAgents(BoidModel *gm){
	GAgent *ag = gm->getScheduler()->obtainAgentPerThread();
	if (ag != NULL)
		ag->getAgId();
}

void initOnDevice(float *x_pos, float *y_pos){
	float *x_pos_h, *y_pos_h;
	x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
	y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));

	std::ifstream fin("pos_data.txt.10240");
	std::string rec;

	char *cstr, *p;
	int i = 0;
	cstr = (char *)malloc(20 * sizeof(char));
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
	size_t floatDataSize = AGENT_NO*sizeof(float);
	cudaMemcpy(x_pos, x_pos_h, floatDataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(y_pos, y_pos_h, floatDataSize, cudaMemcpyHostToDevice);
	cudaCheckErrors("initOnDevice");
}
__global__ void addAgentsOnDevice(BoidModel *gm, float *x_pos, float *y_pos){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		PreyBoid *ag = new PreyBoid();
		ag->loc.x = x_pos[idx];
		ag->loc.y = y_pos[idx];
		ag->time = 0;
		ag->rank = 0;
		ag->model = gm;
		PreyBoid *dummy = new PreyBoid(ag);
		ag->setDummy(dummy);
		gm->addToScheduler(ag, idx);
		gm->addToWorld(ag, idx);
	}
}

__global__ void swapDummy(GModel *gm){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
}

void test1(){
	int gSize = GRID_SIZE;
	printf("sizeof(GModel*): %d\n", sizeof(GModel*));
	printf("sizeof(void*):   %d\n", sizeof(void*));

	BoidModel *model_h = new BoidModel();
	model_h->allocOnDevice();
	BoidModel *model;
	cudaMalloc((void**)&model, sizeof(BoidModel));
	cudaMemcpy(model, model_h, sizeof(BoidModel), cudaMemcpyHostToDevice);
	delete model_h;

	float *x_pos, *y_pos;
	size_t floatDataSize = AGENT_NO*sizeof(float);
	cudaMalloc((void**)&x_pos, floatDataSize);
	cudaMalloc((void**)&y_pos, floatDataSize);
	initOnDevice(x_pos, y_pos);
	addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(model, x_pos, y_pos);
	seeAllAgents<<<gSize, BLOCK_SIZE>>>(model);

	schUtil::scheduleRepeatingAllAgents<<<gSize, BLOCK_SIZE>>>(model);
	int steps;
	std::cout<<"steps: ";
	std::cin>>steps;
	for (int i = 0; i < steps; i++)
		schUtil::step<<<gSize, BLOCK_SIZE>>>(model);
}

void readConfig(){
	std::ifstream fin;
	fin.open("config.txt");
	std::string rec;
	char *cstr, *p;
	cstr = (char *)malloc(100 * sizeof(char));
	int temp;

	while (!fin.eof()) {
		std::getline(fin, rec);
		std::strcpy(cstr, rec.c_str());
		if(strcmp(cstr,"")==0)
			break;
		p=strtok(cstr, "=");
		if(strcmp(p, "AGENT_NO")==0){
			p=strtok(NULL, "=");
			AGENT_NO = atoi(p);
			cudaMemcpyToSymbol(AGENT_NO_D, &AGENT_NO, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "STEPS")==0){
			p=strtok(NULL, "=");
			STEPS = atoi(p);
		}
		if(strcmp(p, "CELL_NO")==0){
			p=strtok(NULL, "=");
			CELL_NO = atoi(p);
			cudaMemcpyToSymbol(CELL_NO_D, &CELL_NO, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "CELL_RESO")==0){
			p=strtok(NULL, "=");
			temp = atoi(p);
			cudaMemcpyToSymbol(CELL_RESO, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_L")==0){
			p=strtok(NULL, "=");
			temp = atoi(p);
			cudaMemcpyToSymbol(BOARDER_L, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_R")==0){
			p=strtok(NULL, "=");
			temp = atoi(p);
			cudaMemcpyToSymbol(BOARDER_R, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_U")==0){
			p=strtok(NULL, "=");
			temp = atoi(p);
			cudaMemcpyToSymbol(BOARDER_U, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_D")==0){
			p=strtok(NULL, "=");
			temp = atoi(p);
			cudaMemcpyToSymbol(BOARDER_D, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "XLENGTH")==0){
			p=strtok(NULL, "=");
			temp = atoi(p);
			cudaMemcpyToSymbol(XLENGTH, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
	}
	cudaCheckErrors("readConfig");
	free(cstr);
	fin.close();
}

void readRandDebug(float *devRandDebug, std::string str1, std::string str2){
	int gSize = GRID_SIZE;
	float *hostRandDebug = (float*)malloc(STRIP*gSize*BLOCK_SIZE*sizeof(float));

	std::istringstream buf1(str1);
	std::istringstream buf2(str2);
	std::istream_iterator<std::string> begin1(buf1), end1;
	std::istream_iterator<std::string> begin2(buf2), end2;
	std::vector<std::string> tokens1(begin1, end1); // done!
	std::vector<std::string> tokens2(begin2, end2); // done!

	for (int i=0; i<AGENT_NO; i++){
		float rand1 = atof(tokens1[i].c_str());
		float rand2 = atof(tokens2[i].c_str());
		hostRandDebug[STRIP*i] = rand1;
		hostRandDebug[STRIP*i+1] = rand2;
	}
	cudaMemcpy(devRandDebug, hostRandDebug, 
		STRIP*gSize*BLOCK_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(randDebug, &devRandDebug, sizeof(devRandDebug),
		0, cudaMemcpyHostToDevice);
	
	free(hostRandDebug);
	buf1.clear();
	buf2.clear();
	tokens1.clear();
	tokens2.clear();
}

int main(int argc, char *argv[]){
	readConfig();

	int gSize = GRID_SIZE;
	BoidModel *model_h = new BoidModel();
	model_h->allocOnDevice();
	BoidModel *model;
	cudaMalloc((void**)&model, sizeof(BoidModel));
	cudaMemcpy(model, model_h, sizeof(BoidModel), cudaMemcpyHostToDevice);

	float *x_pos, *y_pos;
	size_t floatDataSize = AGENT_NO*sizeof(float);
	cudaMalloc((void**)&x_pos, floatDataSize);
	cudaMalloc((void**)&y_pos, floatDataSize);
	initOnDevice(x_pos, y_pos);

	printf("size taken by the one agent:%d and all agents: %d\n",
		sizeof(PreyBoid), AGENT_NO*sizeof(PreyBoid));
	addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(model, x_pos, y_pos);

	//schUtil::scheduleRepeatingAllAgents<<<1, BLOCK_SIZE>>>(model);
	cudaCheckErrors("before going into the big loop");
	printf("steps: %d\n", STEPS);

	std::ifstream fin("randDebugOut2.txt");
	std::string str1;
	std::string str2;
	float *devRandDebug;
	cudaMalloc((void**)&devRandDebug, STRIP*gSize*BLOCK_SIZE*sizeof(float));

	for (int i=0; i<STEPS; i++){
		std::getline(fin, str1);
		std::getline(fin, str2);
		readRandDebug(devRandDebug, str1, str2);

		c2dUtil::genNeighbor(model_h->world);
		schUtil::step<<<gSize, BLOCK_SIZE>>>(model);

		c2dUtil::swapAgentsInWorld<<<gSize, BLOCK_SIZE>>>(model_h->world);
		schUtil::swapAgentsInScheduler<<<gSize, BLOCK_SIZE>>>(model);
	}
	cudaCheckErrors("finished");
	system("PAUSE");
	return 0;
}

void backupcode1(){ //devRand
	int gSize = GRID_SIZE;
	float *devRandDebug;
	cudaMalloc((void**)&devRandDebug, STRIP*gSize*BLOCK_SIZE*sizeof(float));
	cudaMemcpyToSymbol(randDebug, &devRandDebug, sizeof(devRandDebug),
		0, cudaMemcpyHostToDevice);

	std::fstream randDebugOut;
	std::fstream randDebugOut2;
	randDebugOut.open("randDebugOut.txt", std::ios::out);
	randDebugOut2.open("randDebugOut2.txt", std::ios::out);
	float *hostRandDebug = (float*)malloc(STRIP*gSize*BLOCK_SIZE*sizeof(float));

	cudaMemcpy(hostRandDebug, devRandDebug, 
		STRIP*gSize*BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<gSize*BLOCK_SIZE; i++) {
		randDebugOut2<<hostRandDebug[STRIP*i]<<"\t";
		randDebugOut2.flush();
	}
	randDebugOut2<<std::endl;
	for(int i=0; i<gSize*BLOCK_SIZE; i++) {
		randDebugOut2<<hostRandDebug[STRIP*i+1]<<"\t";
		randDebugOut2.flush();
	}
	randDebugOut2<<std::endl;
	for(int i=0; i<gSize*BLOCK_SIZE; i++) {
		randDebugOut<<
			hostRandDebug[STRIP*i]<<" \t"<<
			hostRandDebug[STRIP*i+1]<<" \t"<<
			hostRandDebug[STRIP*i+2]<<" \t"<<
			hostRandDebug[STRIP*i+3]<<" \t"<<
			std::endl;
		randDebugOut.flush();
	}
	randDebugOut.close();
}