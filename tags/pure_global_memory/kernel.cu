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

void readConfig(char *config_file){
	std::ifstream fin;
	fin.open(config_file);
	std::string rec;
	char *cstr, *p;
	cstr = (char *)malloc(100 * sizeof(char));

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
			cudaCheckErrors("readConfig");
		}
		if(strcmp(p, "BOARDER_L")==0){
			p=strtok(NULL, "=");
			BOARDER_L_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_L_D, &BOARDER_L_H, sizeof(int));
			cudaCheckErrors("readConfig");
		}
		if(strcmp(p, "BOARDER_R")==0){
			p=strtok(NULL, "=");
			BOARDER_R_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_R_D, &BOARDER_R_H, sizeof(int));
			cudaCheckErrors("readConfig");
		}
		if(strcmp(p, "BOARDER_U")==0){
			p=strtok(NULL, "=");
			BOARDER_U_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_U_D, &BOARDER_U_H, sizeof(int));
			cudaCheckErrors("readConfig");
		}
		if(strcmp(p, "BOARDER_D")==0){
			p=strtok(NULL, "=");
			BOARDER_D_H = atoi(p);
			cudaMemcpyToSymbol(BOARDER_D_D, &BOARDER_D_H, sizeof(int));
			cudaCheckErrors("readConfig");
		}
		if(strcmp(p, "RANGE")==0){
			p=strtok(NULL, "=");
			RANGE_H = atof(p);
			cudaMemcpyToSymbol(RANGE, &RANGE_H, sizeof(float));
			cudaCheckErrors("readConfig");
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
		if(strcmp(p, "HEAP_SIZE")==0){
			p=strtok(NULL, "=");
			HEAP_SIZE = atoi(p);
		}
		if(strcmp(p, "DATA_FILENAME")==0){
			dataFileName = new char[20];
			p=strtok(NULL, "=");
			strcpy(dataFileName, p);
		}
	}
	free(cstr);
	fin.close();

	int CNO_PER_DIM_H = (int)pow((float)2, DISCRETI);
	cudaMemcpyToSymbol(CNO_PER_DIM, &CNO_PER_DIM_H, sizeof(int));
	cudaCheckErrors("readConfig");
	
	CELL_NO = CNO_PER_DIM_H * CNO_PER_DIM_H;
	cudaMemcpyToSymbol(CELL_NO_D, &CELL_NO, sizeof(int));
	cudaCheckErrors("readConfig");
	
	float CLEN_X_H = (float)(BOARDER_R_H-BOARDER_L_H)/CNO_PER_DIM_H;
	float CLEN_Y_H = (float)(BOARDER_D_H-BOARDER_U_H)/CNO_PER_DIM_H;
	cudaMemcpyToSymbol(CLEN_X, &CLEN_X_H, sizeof(int));
	cudaCheckErrors("readConfig");
	cudaMemcpyToSymbol(CLEN_Y, &CLEN_Y_H, sizeof(int));
	cudaCheckErrors("readConfig");

	GRID_SIZE = AGENT_NO%BLOCK_SIZE==0 ? AGENT_NO/BLOCK_SIZE : AGENT_NO/BLOCK_SIZE + 1;
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
					<<hostRandDebug[STRIP*i]<<" \t"
					<<hostRandDebug[STRIP*i+1]<<" \t"
					<<std::endl;		
				randDebugOut.flush();		
			}		
			randDebugOut.close();		
			free(hostRandDebug);		
			exit(1);		
		}	
	}
}

void oneStep(BoidModel *model, BoidModel *model_h){
	int gSize = GRID_SIZE;
	c2dUtil::genNeighbor(model_h->world, model_h->worldH);
	schUtil::step<<<gSize, BLOCK_SIZE>>>(model);

	c2dUtil::swapAgentsInWorld<<<gSize, BLOCK_SIZE>>>(model_h->world);
	schUtil::swapAgentsInScheduler<<<gSize, BLOCK_SIZE>>>(model);
}

int mainWork(char *config_file){
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	readConfig(config_file);
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
	cudaMemcpyToSymbol(randDebug, &devRandDebug, sizeof(devRandDebug),
		0, cudaMemcpyHostToDevice);

#ifdef _WIN32
	GSimVisual::getInstance().setWorld(model_h->world);
	for (int i=0; i<STEPS; i++){
		if ((i%(STEPS/10))==0) printf("STEP:%d ", i);
		oneStep(model, model_h);
		GSimVisual::getInstance().animate();
		writeRandDebug(i, devRandDebug);
	}
	GSimVisual::getInstance().stop();
#else
	for (int i=0; i<STEPS; i++){
	 	if ((i%(STEPS/10))==0) printf("STEP:%d ", i);
		oneStep(model, model_h);
		writeRandDebug(i, devRandDebug);
	}
#endif
	return 0;
}

int main(int argc, char *argv[]){
#ifndef _WIN32
	struct timeval start, end;
	gettimeofday(&start, NULL);
	mainWork(argv[1]);
	gettimeofday(&end, NULL);
	printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec)));
#else
	int start = GetTickCount();
	mainWork(argv[1]);
	int end = GetTickCount();
	int diff = end-start;
	std::cout<<"Took "<<diff<<" ms"<<std::endl;
	system("PAUSE");
#endif
}