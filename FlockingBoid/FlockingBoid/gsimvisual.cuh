#ifndef GSIMVISUAL_H
#define GSIMVISUAL_H

#include "header.cuh"
#include "common\book.h"
#include "common\gl_helper.h"
#include "cuda_gl_interop.h"
#include "gsimcore.cuh"

namespace visUtil{
	__global__ void paint(uchar4 *devPtr, const Continuous2D *world);
};

class GSimVisual{
private:
	GLuint bufferObj;
	cudaGraphicsResource *resource;
	Continuous2D *world;
	int width;
	int height;

	PFNGLBINDBUFFERARBPROC    glBindBuffer;
	PFNGLDELETEBUFFERSARBPROC glDeleteBuffers;
	PFNGLGENBUFFERSARBPROC    glGenBuffers;
	PFNGLBUFFERDATAARBPROC    glBufferData;

	GSimVisual(){
		this->width = 256;
		this->height = 256;
		glBindBuffer     = NULL;
		glDeleteBuffers  = NULL;
		glGenBuffers     = NULL;
		glBufferData     = NULL;
		
		int c = 1;
		char *dummy = " ";
		glutInit( &c, &dummy );
		glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
		glutInitWindowSize( this->width, this->height );
		glutCreateWindow( "bitmap" );

		glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
		glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
		glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
		glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
		glGenBuffers( 1, &bufferObj );
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
		glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, this->height*this->height*sizeof(uchar4),
			NULL, GL_DYNAMIC_DRAW_ARB );
		cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
		cudaCheckErrors("cudaGraphicsGLRegisterBuffer");

		glutDisplayFunc(drawFunc);
		glutIdleFunc(idleFunc);
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	}

	static void idleFunc(){
		GSimVisual vis = GSimVisual::getInstance();
		uchar4* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &vis.resource, NULL);
		cudaCheckErrors("cudaGraphicsMapResources");
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, vis.resource);
		cudaCheckErrors("cudaGraphicsResourceGetMappedPointer");
		int gSize = GRID_SIZE;
		
		visUtil::paint<<<256, 256>>>(devPtr, vis.world);

		cudaGraphicsUnmapResources(1, &vis.resource, NULL);
		cudaCheckErrors("cudaGraphicsUnmapResources");

		glutPostRedisplay();
	}

	static void drawFunc(){
		glClearColor( 0.0, 0.0, 0.0, 1.0 );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		
		GSimVisual vis = GSimVisual::getInstance();

		uchar4* devPtr;
		size_t size;
		cudaGraphicsMapResources(1, &vis.resource, NULL);
		cudaCheckErrors("cudaGraphicsMapResources");
		cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, vis.resource);
		cudaCheckErrors("cudaGraphicsResourceGetMappedPointer");
		cudaMemset(devPtr, 0, size);
		cudaCheckErrors("cudaMemset");

		int gSize = GRID_SIZE;
		visUtil::paint<<<gSize, BLOCK_SIZE>>>(devPtr, vis.world);

		cudaGraphicsUnmapResources(1, &vis.resource, NULL);
		cudaCheckErrors("cudaGraphicsUnmapResources");

		glDrawPixels( vis.height, vis.height, GL_RGBA,GL_UNSIGNED_BYTE, 0 );

		glutSwapBuffers();
		glutPostRedisplay();
	}

public:
	static GSimVisual& getInstance(){
		static GSimVisual instance;
		return instance;
	}

	void setWorld(Continuous2D *world){
		GSimVisual::getInstance().world = world;
	}

	void animate(){
		glutMainLoopEvent();
	}
};

__global__ void visUtil::paint(uchar4 *devPtr, const Continuous2D *world){

	GAgent *ag = world->obtainAgentPerThread();
	int canvasX = (int)(ag->loc.x*256/1000);
	int canvasY = (int)(ag->loc.y*256/1000);
	int canvasIdx = canvasY*256 + canvasX;
	const int idx = canvasIdx;
	devPtr[idx].x = 0;
	devPtr[idx].y = 255;
	devPtr[idx].z = 0;
	devPtr[idx].w = 255;
}

#endif