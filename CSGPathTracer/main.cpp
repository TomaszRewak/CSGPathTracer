#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <GL/freeglut.h>
#include <GL/glew.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <iostream>

#include "kernel.h"
#include "PathTracer\SceneDefinition\scene.hpp"
#include "PathTracer\Communication\component.hpp"

float globalTime = 0;

int windowWidth = 300;
int windowHeight = 300;

int width = 300;
int height = 300;

GLuint texture;
GLuint pixelBuffer;
cudaGraphicsResource *cudaBuffer;

size_t shapesNumber = 0;
PathTracer::SceneDefinition::Scene scene;
PathTracer::Communication::Component* zippedComponentsDevice = NULL;
PathTracer::Communication::Component* zippedComponentsHost = NULL;

size_t WangHash(size_t a) { // as described here : http://richiesams.blogspot.com/2015/03/creating-randomness-and-acummulating.html
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

float rotation = 0;
void createDataBuffer()
{
	rotation += 0.1;

	scene.components.clear();

	scene.components.push_back(
		std::make_shared<PathTracer::SceneDefinition::DifferenceComponent>(
			Math::AffineTransformation().scale(100, 100, 100).rotateX(rotation * 0.2).rotateY(rotation * -0.4),
			std::make_shared<PathTracer::SceneDefinition::DifferenceComponent>(
				std::make_shared<PathTracer::SceneDefinition::SphereComponent>(),
				std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
					Math::AffineTransformation().scale(0.3, 0.3, 0.3),
					std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
						std::make_shared<PathTracer::SceneDefinition::CylinderComponent>(Math::AffineTransformation().rotateZ(1.57)),
						std::make_shared<PathTracer::SceneDefinition::CylinderComponent>(Math::AffineTransformation().rotateX(1.57))
						),
					std::make_shared<PathTracer::SceneDefinition::CylinderComponent>()
					)
				),
			std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
				Math::AffineTransformation().scale(0.8, 0.8, 0.8),
				std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
					std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0)),
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateZ(1.57))
						),
					std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateZ(-1.57)),
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateX(1.57))
						)
					),
				std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
					std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateX(-1.57)),
					std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateZ(-3.14))
					)
				)
			)
	);

	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::SphereComponent>(Math::AffineTransformation().scale(20, 20, 20)));

	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0)));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateX(-1.57)));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateX(3.14)));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateZ(1.57)));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateZ(-1.57)));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -800, 0).rotateX(1.57)));

	//scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::SphereComponent>(Math::AffineTransformation().scale(50, 50, 50).translate(150, 150, 150).rotateZ(rotation * 0.5f)));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::SphereComponent>(Math::AffineTransformation().scale(50, 50, 50).translate(100, 0, 100).rotateY(rotation * 0.5f)));

	size_t newShapesNumber = scene.zipSize();
	size_t size = newShapesNumber * sizeof(PathTracer::Communication::Component);

	if (newShapesNumber != shapesNumber)
	{
		if (zippedComponentsHost)
			delete[] zippedComponentsHost;
		if (zippedComponentsDevice)
			cudaFree(zippedComponentsDevice);

		zippedComponentsHost = new PathTracer::Communication::Component[newShapesNumber];
		cudaMalloc(&zippedComponentsDevice, size);
	}

	scene.zip(zippedComponentsHost);

	shapesNumber = newShapesNumber;
	cudaMemcpy(zippedComponentsDevice, zippedComponentsHost, size, cudaMemcpyHostToDevice);
}

size_t frameNumber = 0;

void renderImage()
{
	createDataBuffer();

	uchar4* imageArray;
	size_t imageArraySize;

	cudaGraphicsMapResources(1, &cudaBuffer);
	cudaGraphicsResourceGetMappedPointer((void **)&imageArray, &imageArraySize, cudaBuffer);

	PathTracer::renderRect(imageArray, width, height, zippedComponentsDevice, shapesNumber, WangHash(frameNumber++));

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

	glutTimerFunc(30, timerEvent, 0);
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