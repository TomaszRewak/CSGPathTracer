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
#include "PathTracer\Shading\shader.hpp"
#include "PathTracer\Rendering\Camera.hpp"

float globalTime = 0;

int width = 600;
int height = 400;

GLuint texture;
GLuint pixelBuffer;
cudaGraphicsResource *cudaBuffer;

PathTracer::Rendering::Camera camera = PathTracer::Rendering::Camera(
	Math::AffineTransformation().translate(100, 0, -600),
	5.f
);

size_t zippedComponentsNumber = 0;
PathTracer::SceneDefinition::Scene scene;
PathTracer::Communication::Component* zippedComponentsDevice = NULL;
PathTracer::Communication::Component* zippedComponentsHost = NULL;

void createDataBuffer()
{
	auto whiteShader = PathTracer::Shading::Shader(PathTracer::Shading::ShaderType::Uniform, PathTracer::Shading::Shading(0, 0.7, 0.3, 0.5, 0, PathTracer::Shading::Color(0.9, 0.9, 0.9)));
	auto redShader = PathTracer::Shading::Shader(PathTracer::Shading::ShaderType::Uniform, PathTracer::Shading::Shading(0, 0.7, 0.3, 1., 0, PathTracer::Shading::Color(0.9, 0.4, 0.6)));
	auto greenShader = PathTracer::Shading::Shader(PathTracer::Shading::ShaderType::Uniform, PathTracer::Shading::Shading(0, 0.7, 0.3, 0.3, 0, PathTracer::Shading::Color(0.4, 0.9, 0.6)));
	auto blueShader = PathTracer::Shading::Shader(PathTracer::Shading::ShaderType::Uniform, PathTracer::Shading::Shading(0, 0.7, 0.3, 0.8, 0, PathTracer::Shading::Color(0.5, 0.6, 0.9)));
	auto emmisiveShaderA = PathTracer::Shading::Shader(PathTracer::Shading::ShaderType::Uniform, PathTracer::Shading::Shading(0.85, 0.7, 0.3, 0.5, 0, PathTracer::Shading::Color(1.f, 1.f, 1.f)));
	auto emmisiveShaderB = PathTracer::Shading::Shader(PathTracer::Shading::ShaderType::Uniform, PathTracer::Shading::Shading(0.85, 0.7, 0.3, 0.5, 0, PathTracer::Shading::Color(0.7f, 0.6f, 0.9f)));

	scene.components.clear();

	scene.components.push_back(
		std::make_shared<PathTracer::SceneDefinition::DifferenceComponent>(
			Math::AffineTransformation().scale(100, 100, 100).rotateX(-0.36).rotateY(0.52),
			std::make_shared<PathTracer::SceneDefinition::DifferenceComponent>(
				std::make_shared<PathTracer::SceneDefinition::SphereComponent>(redShader),
				std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
					Math::AffineTransformation().scale(0.5, 0.5, 0.5),
					std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
						std::make_shared<PathTracer::SceneDefinition::CylinderComponent>(Math::AffineTransformation().rotateZ(1.57), greenShader),
						std::make_shared<PathTracer::SceneDefinition::CylinderComponent>(Math::AffineTransformation().rotateX(1.57), greenShader)
						),
					std::make_shared<PathTracer::SceneDefinition::CylinderComponent>(greenShader)
					)
				),
			std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
				Math::AffineTransformation().scale(0.8, 0.8, 0.8),
				std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
					std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0), blueShader),
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateZ(1.57), blueShader)
						),
					std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateZ(-1.57), blueShader),
						std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateX(1.57), blueShader)
						)
					),
				std::make_shared<PathTracer::SceneDefinition::UnionComponent>(
					std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateX(-1.57), blueShader),
					std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -1, 0).rotateZ(-3.14), blueShader)
					)
				)
			)
	);

	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::SphereComponent>(Math::AffineTransformation().scale(20, 20, 20).translate(100, 0, 0), emmisiveShaderA));

	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0), whiteShader));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -800, 0).rotateX(-1.57), whiteShader));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateX(3.14), whiteShader));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateZ(1.57), whiteShader));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -200, 0).rotateZ(-1.57), whiteShader));
	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::PlaneComponent>(Math::AffineTransformation().translate(0, -800, 0).rotateX(1.57), whiteShader));

	scene.components.push_back(std::make_shared<PathTracer::SceneDefinition::SphereComponent>(Math::AffineTransformation().scale(50, 50, 50).translate(0, 175, -200), emmisiveShaderB));

	size_t newShapesNumber = scene.zipSize();
	size_t size = newShapesNumber * sizeof(PathTracer::Communication::Component);

	if (newShapesNumber != zippedComponentsNumber)
	{
		if (zippedComponentsHost)
			delete[] zippedComponentsHost;
		if (zippedComponentsDevice)
			cudaFree(zippedComponentsDevice);

		zippedComponentsHost = new PathTracer::Communication::Component[newShapesNumber];
		cudaMalloc(&zippedComponentsDevice, size);
	}

	scene.zip(zippedComponentsHost);

	zippedComponentsNumber = newShapesNumber;
	cudaMemcpy(zippedComponentsDevice, zippedComponentsHost, size, cudaMemcpyHostToDevice);
}

size_t frameNumber = 0;

void renderImage()
{
	if (frameNumber > 3)
	{
		frameNumber = 0;
		camera.transformation = camera.transformation.translate(0, 0, 25).rotateZ(0.15);
	}

	createDataBuffer();

	float4* imageArray;
	size_t imageArraySize;

	cudaGraphicsMapResources(1, &cudaBuffer);
	cudaGraphicsResourceGetMappedPointer((void **)&imageArray, &imageArraySize, cudaBuffer);

	PathTracer::renderRect(
		imageArray,
		width, height,
		camera,
		zippedComponentsDevice, zippedComponentsNumber,
		frameNumber++);

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

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void createPixelBuffer()
{
	glGenBuffers(1, &pixelBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);

	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float4), NULL, GL_STREAM_DRAW);

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

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);

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

	glutMainLoop();
	timerEvent(0);

	return 0;
}