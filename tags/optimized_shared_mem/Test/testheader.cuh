#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//	Later to be specified in the XML file
#define CELL_RESO	100
#define BOARDER_L	0
#define BOARDER_R	1000
#define BOARDER_U	0
#define BOARDER_D	1000
#define CNO_PER_DIM		((int)(BOARDER_R-BOARDER_L)/CELL_RESO)
#define AGENT_NO	1000
#define CELL_NO		100 //10*10
#define BLOCK_SIZE	1024

typedef struct int_2d
{
	int x;
	int y;
	__device__ __host__ int_2d():x(0),y(0){}
	__device__ __host__ int cell_id(){
		return y * CNO_PER_DIM + x;
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
	__device__ __host__ float distance (float_2d p){
		return sqrt((p.x-x)*(p.x-x)+(p.y-y)*(p.y-y));
	}
	__device__ __host__ void print(){
		printf("(%f, %f)", x, y);
	}
} float2d_t;

namespace CONSTANT{
	// CONSTANTS FOR PREY BOIDS
	const float PREY_SENSE_RANGE_FOOD = 50;
	static const int PREY_HUNGER_LIMIT = 500;
	static const int PREY_STARVE_LIMIT = 1000;
	static const int PREY_MATING_TIME = 3;
	static const float PREY_STD_SPEED = 0.7;

	// CONSTANTS FOR PREDATORS
	static const int PRED_HUNGER_LIMIT = 600;
	static const float PRED_DEFAULT_SPEED = 0.4;
	static const float PRED_MAX_SPEED = 1.8;
	static const int PRED_ACC_DURATION = 150;
	static const int PRED_ACC_COOLDOWN = 250;
	static const int PRED_STARVE_LIMIT = 1500;
	static const float PRED_NEIGHBORHOOD_SENSE = 50;
	static const float PRED_SURROUNDING_SENSE = 40;
	static const float PRED_STD_JUMP = 0.5;
	static const float PRED_MATING_TIME = 3;

	// CONSTANTS FOR FOOD
	static const int FOOD_AMOUNT = 100;

	// CONSTANTS OF MODEL
	static const int NUM_PREY = 100; //default = 75
	static const int NUM_PRED = 5; //default = 10

	// Set to false if the GUI is not used
	// (mainly for ea runs)
	static const bool IS_VISUALIZED = true;

	// CONSTANTS OF EA
	static const int NUM_REPETITIONS = 50;
	static const int SIM_STEPS = 5000;
	static const int GENERATIONS = 50;
}

#ifdef __CUDACC__
#define _cudaHost __host__
#define _cudaDevice __device__
#define _cudaShared __shared__
#else
#define _cudaHost
#define _cudaDevice
#define _cudaShared
#endif

#define CU_CHECK( call) {											\
	cudaError err = call;											\
	printf("CU_CHECK: %d-%s in line %i.\n", err, 					\
	cudaGetErrorString(err), __LINE__);								\
	if( cudaSuccess != err) {										\
	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	__FILE__, __LINE__, cudaGetErrorString( err) );					\
	exit(EXIT_FAILURE);												\
	} }

#define getLastCudaError(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
		printf("getLastCudaError: %s: \n\t%d-%s in line %i.\n", msg, \
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