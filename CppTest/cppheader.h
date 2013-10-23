#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>

//	Later to be specified in the XML file
#define CELL_RESO	10
#define BOARDER_L	0
#define BOARDER_R	1000
#define BOARDER_U	0
#define BOARDER_D	1000
#define AGENT_NO	1024
#define BLOCK_SIZE	1024
int XLENGTH = ((int)(BOARDER_R-BOARDER_L)/CELL_RESO);
int CELL_NO = ((int)(BOARDER_R-BOARDER_L)/CELL_RESO)*
	((int)(BOARDER_D-BOARDER_U)/CELL_RESO);

typedef struct int_2d
{
	int x;
	int y;
	int_2d():x(0),y(0){}
	int cell_id(){
		return y * XLENGTH + x;
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



#endif