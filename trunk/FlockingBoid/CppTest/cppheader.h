#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>

//	Later to be specified in the XML file
#define CELL_RESO	100
#define BOARDER_L	0
#define BOARDER_R	1000
#define BOARDER_U	0
#define BOARDER_D	1000
#define XLENGTH		((int)(BOARDER_R-BOARDER_L)/CELL_RESO)
#define AGENT_NO	1024
#define CELL_NO		100 //10*10
#define BLOCK_SIZE	1024

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
	  float distance (float_2d p){
		return sqrt((p.x-x)*(p.x-x)+(p.y-y)*(p.y-y));
	}
	  void print(){
		printf("(%f, %f)", x, y);
	}
} float2d_t;



#endif