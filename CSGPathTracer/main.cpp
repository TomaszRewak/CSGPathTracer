#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <GL/freeglut.h>
#include <GL/glew.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <iostream>

#include "kernel.h"

float globalTime = 0;

int width = 500;
int height = 500;

GLuint texture;
GLuint pixelBuffer;
cudaGraphicsResource *cudaBuffer;

size_t shapesNumber = 0;
Shape* shapes;

void renderImage()
{
	uchar4* imageArray;
	size_t imageArraySize;

	cudaGraphicsMapResources(1, &cudaBuffer);
	cudaGraphicsResourceGetMappedPointer((void **)&imageArray, &imageArraySize, cudaBuffer);

	renderRect(imageArray, width, height, shapes, shapesNumber);

	cudaGraphicsUnmapResources(1, &cudaBuffer);
}

void createTexture()
{
	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void createPixelBuffer()
{
	glGenBuffers(1, &pixelBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);

	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_STREAM_DRAW);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&cudaBuffer, pixelBuffer, cudaGraphicsRegisterFlagsNone);
}

void createDataBuffer()
{
	shapesNumber = 5;

	size_t size = shapesNumber * sizeof(Shape);
	cudaMalloc(&shapes, size);

	Shape data[] = {
		Shape(ShapeType::Difference, AffineTransformation().scale(200, 200, 200).translate(0, 0, 500), 1, 2), // 0
		Shape(ShapeType::Sphere, AffineTransformation()), // 1
		Shape(ShapeType::Union, AffineTransformation().translate(0, 0, -1). rotateX(1.4), 3, 4), // 2
		Shape(ShapeType::Sphere, AffineTransformation().translate(-0.3, 0, 0)), // 3
		Shape(ShapeType::Sphere, AffineTransformation().translate(0.3, 0, 0)), // 4
	};

	cudaMemcpy(shapes, data, size, cudaMemcpyHostToDevice);
}

void bindCuda()
{
	//cudaGraphicsGLRegisterImage()
}

void displayFunc() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	renderImage();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glutSwapBuffers();
}

void timerEvent(int value) {
	if (glutGetWindow())
		glutPostRedisplay();

	glutTimerFunc(30, timerEvent, 0);
}

void reshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-0.1, 1.1, -0.1, 1.1, -0.1, 1.1);
}

void initGlut(int *argc, char **argv)
{
	glutInit(argc, argv);

	glutInitWindowPosition(-1, -1);
	glutInitWindowSize(width, height);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutCreateWindow(argv[0]);

	glutDisplayFunc(displayFunc);
	glutReshapeFunc(reshapeFunc);

	glewInit();
}

int main(int argc, char **argv)
{
	initGlut(&argc, argv);
	findCudaGLDevice(argc, (const char**)argv);

	createTexture();
	createPixelBuffer();
	createDataBuffer();

	cudaDeviceSetLimit(cudaLimitStackSize, 5000);

	glutMainLoop();
	timerEvent(0);

	return 0;
}