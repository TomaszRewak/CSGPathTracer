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

int windowWidth = 300;
int windowHeight = 300;

int width = 300;
int height = 300;

GLuint texture;
GLuint pixelBuffer;
cudaGraphicsResource *cudaBuffer;

size_t shapesNumber = 0;
Shape* shapes;

float rotation = 0;
void createDataBuffer()
{
	rotation += 0.1;

	Shape data[] = {
		Shape(ShapeType::Difference, AffineTransformation().scale(100, 100, 100).rotateX(rotation * 0.2).rotateY(rotation * -0.4), 1, 8), // 0
		Shape(ShapeType::Difference, AffineTransformation(), 2, 3), // 1
		Shape(ShapeType::Sphere, AffineTransformation()), // 2
		Shape(ShapeType::Union, AffineTransformation().scale(0.3, 0.3, 0.3), 4, 5), // 3
		Shape(ShapeType::Cylinder, AffineTransformation()), // 4
		Shape(ShapeType::Union, AffineTransformation(), 6, 7), // 5
		Shape(ShapeType::Cylinder, AffineTransformation().rotateZ(1.57)), // 6
		Shape(ShapeType::Cylinder, AffineTransformation().rotateX(1.57)), // 7

		Shape(ShapeType::Union, AffineTransformation().scale(0.8, 0.8, 0.8), 9, 10), // 8
		Shape(ShapeType::Union, AffineTransformation(), 11, 12), // 9
		Shape(ShapeType::Union, AffineTransformation(), 13, 14), // 10
		Shape(ShapeType::Union, AffineTransformation(), 15, 16), // 11
		Shape(ShapeType::Union, AffineTransformation(), 17, 18), // 12
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -1, 0)), // 13
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -1, 0).rotateZ(1.57)), // 14
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -1, 0).rotateZ(-1.57)), // 15
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -1, 0).rotateX(1.57)), // 16
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -1, 0).rotateX(-1.57)), // 17
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -1, 0).rotateZ(-3.14)), // 18

		Shape(ShapeType::Sphere, AffineTransformation().scale(20, 20, 20)),

		Shape(ShapeType::Plane, AffineTransformation().translate(0, -200, 0)),
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -200, 0).rotateX(-1.57)),
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -200, 0).rotateX(3.14)),
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -200, 0).rotateZ(1.57)),
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -200, 0).rotateZ(-1.57)),
		Shape(ShapeType::Plane, AffineTransformation().translate(0, -800, 0).rotateX(1.57)),

		Shape(ShapeType::Sphere, AffineTransformation().scale(100, 100, 100).translate(200, 200, 200).rotateZ(rotation * 0.5f)),
	};

	size_t newShapesNumber = sizeof(data) / sizeof(*data);
	size_t size = newShapesNumber * sizeof(Shape);

	if (newShapesNumber != shapesNumber)
	{
		if (shapes)
			cudaFree(shapes);

		cudaMalloc(&shapes, size);
	}

	shapesNumber = newShapesNumber;
	cudaMemcpy(shapes, data, size, cudaMemcpyHostToDevice);
}

void renderImage()
{
	createDataBuffer();

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

void bindCuda()
{
	//cudaGraphicsGLRegisterImage()
}

void timerEvent(int value) {
	if (glutGetWindow())
		glutPostRedisplay();
}

void displayFunc() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	renderImage();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	float border = (float)(windowWidth - width) / 2 / windowWidth;

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(border, border);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f - border, border);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f - border, 1.0f - border);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(border, 1.0f - border);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glutSwapBuffers();

	glutTimerFunc(1, timerEvent, 0);
}

void reshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-0, 1, -0, 1, -0, 1);
}

void initGlut(int *argc, char **argv)
{
	glutInit(argc, argv);

	glutInitWindowPosition(-1, -1);
	glutInitWindowSize(windowWidth, windowHeight);

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

	glutMainLoop();
	timerEvent(0);

	return 0;
}