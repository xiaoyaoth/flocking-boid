#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>

//	Later to be specified in the XML file
#define BOARDER_L	0
#define BOARDER_R	1000
#define BOARDER_U	0
#define BOARDER_D	1000
#define AGENT_NO	1024
#define BLOCK_SIZE	1024
#define DISCRETI	7
int CNO_PER_DIM = (int)pow((float)2, DISCRETI);
int CELL_NO = CNO_PER_DIM * CNO_PER_DIM;
float CLEN_X = (float)(BOARDER_R-BOARDER_L)/CNO_PER_DIM;
float CLEN_Y = (float)(BOARDER_D-BOARDER_U)/CNO_PER_DIM;

typedef struct int_2d
{
	int x;
	int y;
	int_2d():x(0),y(0){}
	int cell_id(){
		return y * CNO_PER_DIM + x;
	}
	int zCode(){
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
	void print(){
		printf("(%d, %d)", x, y);
	}
} int2d_t;
typedef struct float_2d
{
	float x;
	float y;
	//  float_2d():x(0),y(0){}
	float_2d(){
		this->x = 0;
		this->y = 0;
	}
	float_2d(const float xx, const float yy){
		this->x = xx;
		this->y = yy;
	}
	float distance (float_2d p){
		return sqrt((p.x-x)*(p.x-x)+(p.y-y)*(p.y-y));
	}
	void print(){
		printf("(%f, %f)", x, y);
	}
} float2d_t;

typedef struct shared_info sinfo_t;

#endif