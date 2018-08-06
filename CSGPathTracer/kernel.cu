#pragma once

#include "kernel.h"
#include <math.h>

#define N 128
#define Intersections 100

__device__
size_t sphereIntersections(Ray ray, Intersection* intersections)
{
	float a, b, c;

	a = ray.direction.dx * ray.direction.dx + ray.direction.dy * ray.direction.dy + ray.direction.dz * ray.direction.dz;
	b = 2 * (ray.begin.x * ray.direction.dx + ray.begin.y * ray.direction.dy + ray.begin.z * ray.direction.dz);
	c = ray.begin.x * ray.begin.x + ray.begin.y * ray.begin.y + ray.begin.z * ray.begin.z - 1;

	float delta = b * b - 4 * a * c;

	if (delta >= 0)
	{
		float deltaSqrt = sqrtf(delta);

		intersections[0].distance = -(b - deltaSqrt) / (2 * a);
		intersections[1].distance = -(b + deltaSqrt) / (2 * a);

		return 2;
	}
	else
		return 0;
}

bool sphereWithin(Point& point)
{
	return point.x * point.x + point.y * point.y + point.z * point.z <= 1;
}

__device__
size_t trace(Ray& ray, ShapeSpace* shapes, size_t shapesNumber)
{
	Intersection intersections[Intersections];
	size_t intersectionsNumber = 0;

	for (size_t i = 0; i < shapesNumber; i++)
	{
		TwoWayAffineTransformation transformation = shapes[i].transformation;

		Ray transformedRay = transformation.inverse(ray);

		intersectionsNumber += sphereIntersections(transformedRay, intersections + intersectionsNumber);
	}

	return intersectionsNumber;
}

__device__
void copyShapesToSharedMemory(ShapeSpace** shapeSpaces, Shape* shapes, size_t shapesNumber)
{
	if (threadIdx.x == 0) {
		size_t size = sizeof(ShapeSpace) * shapesNumber;
		(*shapeSpaces) = (ShapeSpace*)malloc(size);

		for (size_t i = 0; i < shapesNumber; i++)
		{
			(*shapeSpaces)[i] = shapes[i];
		}
	}
	__syncthreads();
}

__global__
void kernel(uchar4* image, size_t imageWidth, size_t imageHeight, Shape* shapes, size_t shapesNumber)
{
	size_t n = threadIdx.x;
	size_t blockSize = imageWidth * imageHeight / N;
	size_t blockStart = n * blockSize;
	size_t blockEnd = blockStart + blockSize;

	__shared__ ShapeSpace* shapeSpaces;
	copyShapesToSharedMemory(&shapeSpaces, shapes, shapesNumber);

	for (size_t index = blockStart; index < blockEnd; index++)
	{
		float x = (float)(index % imageWidth) - imageWidth / 2;
		float y = (float)(index / imageWidth) - imageHeight / 2;

		Ray ray(Point(x, y, 0), Vector(0, 0, 1));

		size_t intersectionsNumber = trace(ray, shapeSpaces, shapesNumber);

		if (intersectionsNumber)
		{
			image[index].x = 255;
			image[index].y = 255;
			image[index].z = 255;
			image[index].w = 255;
		}
		else
		{
			image[index].x = 100;
			image[index].y = 100;
			image[index].z = 255;
			image[index].w = 255;
		}
	}
}

void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Shape* shapes, size_t shapesNumber) {
	kernel << <1, N >> > (image, imageWidth, imageHeight, shapes, shapesNumber);
}