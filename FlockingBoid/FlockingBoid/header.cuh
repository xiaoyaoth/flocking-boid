#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ int AGENT_NO_D;	//copied from host
__constant__ int CELL_NO_D;		//copied from host
__constant__ int BOARDER_L_D;	//copied from host
__constant__ int BOARDER_R_D;	//copied from host
__constant__ int BOARDER_U_D;	//copied from host
__constant__ int BOARDER_D_D;	//copied from host
__constant__ int CNO_PER_DIM;	//(int)pow((float)2, DISCRETI)
__constant__ float CLEN_X;		//(float)(BOARDER_R-BOARDER_L)/CNO_PER_DIM;
__constant__ float CLEN_Y;		//(float)(BOARDER_D-BOARDER_U)/CNO_PER_DIM;

int CELL_NO;		//CNO_PER_DIM * CNO_PER_DIM;
int DISCRETI;		//read from config

int BOARDER_L_H;	//read from config
int BOARDER_R_H;	//read from config
int BOARDER_U_H;	//read from config
int BOARDER_D_H;	//read from config
int AGENT_NO;		//read from config
int STEPS;			//read from config

int SELECTION;		//read from config
bool VISUALIZE;		//read from config
int VERBOSE;		//read from config
int FILE_GEN;		//read from config

int BLOCK_SIZE;		//read from config
int GRID_SIZE;		//calc with BLOCK_SIZE and AGENT_NO

typedef struct int_2d
{
	int x;
	int y;
	__device__ __host__ int_2d():x(0),y(0){}
	__device__ __host__ int cell_id(){
		return y * CNO_PER_DIM + x;
	}
	__device__ __host__ int zcode(){
		int xt = x;
		int yt = y;
		xt &= 0x0000ffff;                 // x = ---- ---- ---- ---- fedc ba98 7654 3210
		xt = (xt ^ (xt << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		xt = (xt ^ (xt << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		xt = (xt ^ (xt << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		xt = (xt ^ (xt << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

		yt &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
		yt = (yt ^ (yt << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		yt = (yt ^ (yt << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		yt = (yt ^ (yt << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		yt = (yt ^ (yt << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

		return xt | (yt << 1);
	}
	__device__ __host__ void print(){
		printf("(%d, %d)", x, y);
	}
} int2d_t;
typedef struct float_2d
{
	float x;
	float y;
	//__device__ __host__ float_2d():x(0),y(0){}
	__device__ __host__ float_2d(){
		this->x = 0;
		this->y = 0;
	}
	__device__ __host__ float_2d(const float xx, const float yy){
		this->x = xx;
		this->y = yy;
	}
	__device__ __host__ float distance (float_2d p){
		return sqrt((p.x-x)*(p.x-x)+(p.y-y)*(p.y-y));
	}
	__device__ __host__ void print(){
		printf("(%f, %f)", x, y);
	}
} float2d_t;

namespace CONSTANT{
	// CONSTANTS FOR PREY BOIDS
	__device__ const float PREY_SENSE_RANGE_FOOD = 50;
	__device__ static const int PREY_HUNGER_LIMIT = 500;
	__device__ static const int PREY_STARVE_LIMIT = 1000;
	__device__ static const int PREY_MATING_TIME = 3;
	__device__ static const float PREY_STD_SPEED = 0.7;

	// CONSTANTS FOR PREDATORS
	__device__ static const int PRED_HUNGER_LIMIT = 600;
	__device__ static const float PRED_DEFAULT_SPEED = 0.4;
	__device__ static const float PRED_MAX_SPEED = 1.8;
	__device__ static const int PRED_ACC_DURATION = 150;
	__device__ static const int PRED_ACC_COOLDOWN = 250;
	__device__ static const int PRED_STARVE_LIMIT = 1500;
	__device__ static const float PRED_NEIGHBORHOOD_SENSE = 50;
	__device__ static const float PRED_SURROUNDING_SENSE = 40;
	__device__ static const float PRED_STD_JUMP = 0.5;
	__device__ static const float PRED_MATING_TIME = 3;

	// CONSTANTS FOR FOOD
	__device__ static const int FOOD_AMOUNT = 100;

	// CONSTANTS OF MODEL
	__device__ static const int NUM_PREY = 100; //default = 75
	__device__ static const int NUM_PRED = 5; //default = 10

	// Set to false if the GUI is not used
	// (mainly for ea runs)
	__device__ static const bool IS_VISUALIZED = true;

	// CONSTANTS OF EA
	__device__ static const int NUM_REPETITIONS = 50;
	__device__ static const int SIM_STEPS = 5000;
	__device__ static const int GENERATIONS = 50;
}
namespace SCHEDULE_CONSTANT{
	static const float EPOCH = 0.0;
	static const float BEFORE_SIMULATION = EPOCH - 1.0;
	static const float AFTER_SIMULTION = FLT_MAX;
	static const float EPSILON = 1.0;
}

#define CU_CHECK( call) {											\
	cudaError err = call;											\
	if( VERBOSE == 1)  \
	printf("CU_CHECK: %d-%s in line %i.\n", err, 					\
	cudaGetErrorString(err), __LINE__);								\
	if( cudaSuccess != err) {										\
	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	__FILE__, __LINE__, cudaGetErrorString( err) );					\
	exit(EXIT_FAILURE);												\
	} }

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
		if( VERBOSE == 1)  \
		printf("cudaCheckErrors: %s: \n\t%d-%s in line %i.\n", msg, \
		__err, cudaGetErrorString(__err), __LINE__);	\
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#endif