#include "gsimcore.cuh"
#include "gsimapp_boid.cuh"
#include <fstream>

__global__ void seeAllAgents(GModel *gm){
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
		//std::cout<<x_pos_h[i]<<" "<<y_pos_h[i]<<std::endl;
		i++;
	}

	size_t floatDataSize = AGENT_NO*sizeof(float);
	cudaMemcpy(x_pos, x_pos_h, floatDataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(y_pos, y_pos_h, floatDataSize, cudaMemcpyHostToDevice);
	cudaCheckErrors("initOnDevice");
}
__global__ void addAgentsOnDevice(GModel *gm, float *x_pos, float *y_pos){
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < AGENT_NO_D){
		PreyBoid *ag = new PreyBoid();
		//GAgent *ag = new GAgent();
		ag->loc.x = x_pos[idx];
		ag->loc.y = y_pos[idx];
		ag->time = 0;
		ag->rank = 0;
		gm->addToScheduler(ag, idx);
		gm->addToWorld(ag, idx);
	}
}

void test1(){
	int gSize = GRID_SIZE;
	printf("sizeof(GModel*): %d\n", sizeof(GModel*));
	printf("sizeof(void*):   %d\n", sizeof(void*));

	GModel *model_h = new GModel();
	model_h->allocOnDevice();
	GModel *model;
	cudaMalloc((void**)&model, sizeof(GModel));
	cudaMemcpy(model, model_h, sizeof(GModel), cudaMemcpyHostToDevice);
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

int main(int argc, char *argv[]){
	readConfig();
	size_t mallocHeapSize;
	cudaDeviceGetLimit(&mallocHeapSize, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %d\n", mallocHeapSize);

	int gSize = GRID_SIZE;
	GModel *model_h = new GModel();
	model_h->allocOnDevice();
	GModel *model;
	cudaMalloc((void**)&model, sizeof(GModel));
	cudaMemcpy(model, model_h, sizeof(GModel), cudaMemcpyHostToDevice);
	cudaCheckErrors("before read x_pos[] and y_pos");

	float *x_pos, *y_pos;
	size_t floatDataSize = AGENT_NO*sizeof(float);
	cudaMalloc((void**)&x_pos, floatDataSize);
	cudaMalloc((void**)&y_pos, floatDataSize);
	initOnDevice(x_pos, y_pos);
	cudaCheckErrors("before init random states");

	//rgenUtil::initStates<<<gSize, BLOCK_SIZE>>>(model, 1234);
	cudaCheckErrors("before add agents on device");
	printf("size taken by the one agent:%d and all agents: %d\n",
		sizeof(PreyBoid), AGENT_NO*sizeof(PreyBoid));
	addAgentsOnDevice<<<gSize, BLOCK_SIZE>>>(model, x_pos, y_pos);
	cudaCheckErrors("before going into the big loop");

	//schUtil::scheduleRepeatingAllAgents<<<1, BLOCK_SIZE>>>(model);
	printf("steps: %d\n", STEPS);
	for (int i=0; i<STEPS; i++){
		c2dUtil::genNeighbor(model);
		schUtil::step<<<gSize, BLOCK_SIZE>>>(model);
	}

	cudaCheckErrors("finished");
	system("PAUSE");
	return 0;
}

//namespace backupCode{
//	void addAgentsToC2DOnDeviceSide(){//add on device side
//		GModel *model_h = new GModel();
//		model_h->allocOnDevice();
//		GModel *model_d;
//		cudaMalloc((void**)&model_d, sizeof(GModel));
//		cudaMemcpy(model_d, model_h, sizeof(GModel), cudaMemcpyHostToDevice);
//		//gmUtil::addAgents_kernel<<<1, BLOCK_SIZE>>>(model_d);
//	}
//
//	void initOnHost(GAgent **allAg_h){
//
//		float *x_pos_h, *y_pos_h;
//		x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
//		y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
//
//		std::ifstream fin("pos_data.txt");
//		std::string rec;
//
//		char *cstr, *p;
//		int i = 0;
//		cstr = (char *)malloc((rec.length() + 1) * sizeof(char));
//		while (!fin.eof()) {
//			std::getline(fin, rec);
//			std::strcpy(cstr, rec.c_str());
//			if(strcmp(cstr,"")==0)
//				break;
//			p=strtok(cstr, " ");
//			x_pos_h[i] = atof(p);
//			p=strtok(NULL, " ");
//			y_pos_h[i] = atof(p);
//			i++;
//		}
//
//		for(int i=0; i<AGENT_NO; i++){
//			GAgent *ag
//				//= new GAgent()
//				;
//			ag->time = i;
//			ag->rank = AGENT_NO-i;
//			ag->loc.x = x_pos_h[i];
//			ag->loc.y = y_pos_h[i];
//			GAgent *ag_d;
//			cudaMalloc((void**)&ag_d, sizeof(GAgent));
//			cudaMemcpy(ag_d, ag, sizeof(GAgent), cudaMemcpyHostToDevice);
//			delete ag;
//			allAg_h[i] = ag_d;
//		}
//	}
//	void addAgentsToC2DOnHost(GModel *model_d, GAgent **allAgents_h){//add on host side
//		GModel *model_h = new GModel();
//		cudaMemcpy(model_h, model_d, sizeof(GModel), cudaMemcpyDeviceToHost);
//		Continuous2D *world_h = new Continuous2D();
//		cudaMemcpy(world_h, model_h->world, sizeof(Continuous2D), cudaMemcpyDeviceToHost);
//		cudaMemcpy(world_h->allAgents, allAgents_h, AGENT_NO*sizeof(GAgent*), cudaMemcpyHostToDevice);
//		delete model_h;
//		delete world_h;
//		//float *x_pos_h, *y_pos_h;
//		//x_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
//		//y_pos_h = (float*)malloc(AGENT_NO*sizeof(float));
//		//init(x_pos_h, y_pos_h);
//		//for(int i=0; i<AGENT_NO; i++){
//		//	GAgent *ag = new GAgent();
//		//	ag->ag_id = i;
//		//	ag->loc.x = x_pos_h[i];
//		//	ag->loc.y = y_pos_h[i];
//		//	GAgent *ag_d;
//		//	cudaMalloc((void**)&ag_d, sizeof(GAgent));
//		//	cudaMemcpy(ag_d, ag, sizeof(GAgent), cudaMemcpyHostToDevice);
//		//	cudaMemcpy(&world_h->allAgents[i], &ag_d, sizeof(GAgent*), cudaMemcpyHostToDevice);
//		//}
//	}
//	void addAgentsToSchOnHost(GModel *gm, GAgent **allAgents_h){//add on host side
//		GModel *model_h = new GModel();
//		GScheduler *scheduler_h = new GScheduler();
//		cudaMemcpy(model_h, gm, sizeof(GModel), cudaMemcpyDeviceToHost);
//		cudaMemcpy(scheduler_h, model_h->scheduler, sizeof(GScheduler), cudaMemcpyDeviceToHost);
//		cudaMemcpy(scheduler_h->allAgents, allAgents_h, AGENT_NO*sizeof(GAgent*), cudaMemcpyHostToDevice);
//		delete model_h;
//		delete scheduler_h;
//	}
//
//	void backup(){
//		//seg1
//		//GAgent **allAgents_h = (GAgent**)malloc(AGENT_NO*sizeof(GAgent*));
//		//init(allAgents_h);
//		//addAgentsToC2DOnHostSide(model, allAgents_h);
//		//addAgentsToSchOnHostSide(model, allAgents_h);
//
//		//seg2
//		//model_h = new GModel();
//		//cudaMemcpy(model_h, model, sizeof(GModel), cudaMemcpyDeviceToHost);
//		//c2dUtil::genNeighbor(model_h->world);
//		//c2dUtil::queryNeighbor<<<1, BLOCK_SIZE>>>(model_h->world);
//		//schUtil::sortWithKey(model);
//	}
//};