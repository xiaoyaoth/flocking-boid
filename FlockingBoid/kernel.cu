#include "gsimcore.cuh"
#include "gsimapp_boid.cuh"
#include "gsimvisual.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>
#include <iomanip>

__global__ void seeAllAgents(BoidModel *gm){
	GAgent *ag = gm->getScheduler()->obtainAgentPerThread();
	if (ag != NULL)
		ag->getId();
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
	getLastCudaError("initOnDevice");
}
__global__ void addAgentsOnDevice(BoidModel *gm, float *x_pos, float *y_pos){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){ // user init step
		PreyBoid *ag = new PreyBoid(x_pos[idx], y_pos[idx], gm);

		PreyBoid *dummy = new PreyBoid(*ag);
		dummy->model = gm;

		ag->setDummy(dummy);
		gm->addToScheduler(ag, idx);
		gm->addToWorld(ag, idx);
	}
	gm->getScheduler()->setAssignments(
		gm->getWorld()->getNeighborIdx()
		);
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
	int CELL_RESO_TEMP;

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
		if(strcmp(p, "BOARDER_L")==0){
			p=strtok(NULL, "=");
			BOARDER_L_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_L_D, &BOARDER_L_H, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_R")==0){
			p=strtok(NULL, "=");
			BOARDER_R_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_R_D, &BOARDER_R_H, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_U")==0){
			p=strtok(NULL, "=");
			BOARDER_U_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_U_D, &BOARDER_U_H, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "BOARDER_D")==0){
			p=strtok(NULL, "=");
			BOARDER_D_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_D_D, &BOARDER_D_H, sizeof(int), 0, cudaMemcpyHostToDevice);
		}
		if(strcmp(p, "DISCRETI")==0){
			p=strtok(NULL, "=");
			DISCRETI = atoi(p);
		}
		if(strcmp(p, "STEPS")==0){
			p=strtok(NULL, "=");
			STEPS = atoi(p);
		}
		if(strcmp(p, "VERBOSE")==0){
			p=strtok(NULL, "=");
			VERBOSE = atoi(p);
		}
		if(strcmp(p, "SELECTION")==0){
			p=strtok(NULL, "=");
			SELECTION = atoi(p);
		}
		if(strcmp(p, "VISUALIZE")==0){
			p=strtok(NULL, "=");
			VISUALIZE = atoi(p);
		}
		if(strcmp(p, "FILE_GEN")==0){
			p=strtok(NULL, "=");
			FILE_GEN = atoi(p);
		}
		if(strcmp(p, "BLOCK_SIZE")==0){
			p=strtok(NULL, "=");
			BLOCK_SIZE = atoi(p);
		}
	}
	getLastCudaError("readConfig");
	free(cstr);
	fin.close();

	int CNO_PER_DIM_H = (int)pow((float)2, DISCRETI);
	cudaMemcpyToSymbol(CNO_PER_DIM, &CNO_PER_DIM_H, 
		sizeof(int), 0, cudaMemcpyHostToDevice);
	
	CELL_NO = CNO_PER_DIM_H * CNO_PER_DIM_H;
	cudaMemcpyToSymbol(CELL_NO_D, &CELL_NO, 
		sizeof(int), 0, cudaMemcpyHostToDevice);
	
	float CLEN_X_H = (float)(BOARDER_R_H-BOARDER_L_H)/CNO_PER_DIM_H;
	float CLEN_Y_H = (float)(BOARDER_D_H-BOARDER_U_H)/CNO_PER_DIM_H;
	cudaMemcpyToSymbol(CLEN_X, &CLEN_X_H, 
		sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CLEN_Y, &CLEN_Y_H, 
		sizeof(int), 0, cudaMemcpyHostToDevice);

	GRID_SIZE = AGENT_NO%BLOCK_SIZE==0 ? 
		AGENT_NO/BLOCK_SIZE : AGENT_NO/BLOCK_SIZE + 1;

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
	
	free(hostRandDebug);
	buf1.clear();
	buf2.clear();
	tokens1.clear();
	tokens2.clear();
}

void writeRandDebug(int i, float* devRandDebug){
	//float *hostRandDebug2 = (float*)malloc(sizeof(float));
	//cudaMemcpy(hostRandDebug2, devRandDebug, sizeof(float), cudaMemcpyDeviceToHost);
	//if (hostRandDebug2[0] != 0)
	//	std::cout<<"hostRandDebug2 "<<hostRandDebug2[0]<<std::endl;
	if (FILE_GEN == 1){
		int gSize = GRID_SIZE;
		if (i == SELECTION) {		
			char *outfname = new char[10];		
			sprintf(outfname, "gpuout%d.txt", i);		
			printf("SELECTION\n");		
			std::fstream randDebugOut;		
			randDebugOut.open(outfname, std::ios::out);		
			float *hostRandDebug = (float*)malloc(STRIP*gSize*BLOCK_SIZE*sizeof(float));		
			cudaMemcpy(hostRandDebug, devRandDebug,		
				STRIP*gSize*BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost);		
			for(int i=0; i<AGENT_NO; i++) {		
				randDebugOut
					<<std::setw(4)
					<<i<< "\t"
					//<<hostRandDebug[STRIP*i]<<"\t"
					//<<hostRandDebug[STRIP*i+1]<<"\t"
					<<hostRandDebug[STRIP*i+2]<<"\t"
					<<hostRandDebug[STRIP*i+3]<<"\t"
					<<hostRandDebug[STRIP*i+4]<<"\t"
					<<std::endl;		
				randDebugOut.flush();		
			}		
			randDebugOut.close();		
			free(hostRandDebug);		
			exit(1);		
		}	
	}
}

__global__ void checkEverything(BoidModel *model){
	//const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	Continuous2D *world = model->getWorld();
	GScheduler * sch = model->getScheduler();
	const int *localNeiIdx = world->getNeighborIdx();
	for (int i=0; i<AGENT_NO_D; i++) {
		int agIdx = localNeiIdx[i];
		GAgent *ag = sch->obtainAgentById(agIdx);
		ag->getData()->id = agIdx;
	}
}

void oneStep(BoidModel *model, BoidModel *model_h){
	int gSize = GRID_SIZE;
	size_t sizeOfSmem = BLOCK_SIZE * (
		4*sizeof(int)
		+ sizeof(dataUnion)
		);
	getLastCudaError("before loop");
	c2dUtil::genNeighbor(model_h->world, model_h->worldH);
	//checkEverything<<<gSize, BLOCK_SIZE, sizeOfSmem>>>(model);
	schUtil::step<<<gSize, BLOCK_SIZE, sizeOfSmem>>>(model);
	c2dUtil::swapAgentsInWorld<<<gSize, BLOCK_SIZE>>>(model_h->world);
	schUtil::swapAgentsInScheduler<<<gSize, BLOCK_SIZE>>>(model);
	getLastCudaError("end loop");
}

int main(int argc, char *argv[]){
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
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
	printf("size taken by one iterInfo: %d\n", sizeof(iterInfo));
	printf("size taken by one dataUnion: %d\n", sizeof(dataUnion));
	size_t pVal;
	cudaDeviceGetLimit(&pVal, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %d", pVal);

	addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(model, x_pos, y_pos);
	//schUtil::scheduleRepeatingAllAgents<<<1, BLOCK_SIZE>>>(model);
	getLastCudaError("before going into the big loop");
	printf("steps: %d\n", STEPS);

	std::ifstream fin("randDebugOut2.txt");
	float *devRandDebug;
	cudaMalloc((void**)&devRandDebug, STRIP*gSize*BLOCK_SIZE*sizeof(float));
	cudaMemcpyToSymbol(randDebug, &devRandDebug, sizeof(devRandDebug),
		0, cudaMemcpyHostToDevice);

	GSimVisual::getInstance().setWorld(model_h->world);
	for (int i=0; i<STEPS; i++){
		printf("STEP:%d\n", i);
		oneStep(model, model_h);
		GSimVisual::getInstance().animate();
		writeRandDebug(i, devRandDebug);
	}
	GSimVisual::getInstance().stop();
	getLastCudaError("finished");
	//system("PAUSE");
	return 0;
}
