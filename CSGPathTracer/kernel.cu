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

	size_t intersectionsNumber = 0;

	if (delta >= 0.0f)
	{
		float deltaSqrt = sqrtf(delta);
		float
			d1 = -(b - deltaSqrt) / (2 * a),
			d2 = -(b + deltaSqrt) / (2 * a);

		if (d1 > 0.0f)
		{
			Point intersectionPoint = ray.begin + ray.direction * d1;

			intersections[intersectionsNumber] = Intersection(
				intersectionPoint,
				intersectionPoint - Point(0, 0, 0)
			);

			intersectionsNumber++;
		}
		if (d2 > 0.0f)
		{
			Point intersectionPoint = ray.begin + ray.direction * d2;

			intersections[intersectionsNumber] = Intersection(
				intersectionPoint,
				intersectionPoint - Point(0, 0, 0)
			);

			intersectionsNumber++;
		}
	}

	return intersectionsNumber;
}

bool sphereWithin(Point& point)
{
	return point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
}

__device__
float trace(Ray& ray, ShapeSpace* shapes, size_t shapesNumber)
{
	Intersection intersections[Intersections];
	size_t intersectionsNumber = 0;

	for (size_t i = 0; i < shapesNumber; i++)
	{
		TwoWayAffineTransformation& transformation = shapes[i].transformation;

		Ray transformedRay = transformation.inverse(ray);

		size_t newIntersectionsNumber = intersectionsNumber + sphereIntersections(transformedRay, intersections + intersectionsNumber);

		for (size_t intersectionNumber = intersectionsNumber; intersectionNumber < newIntersectionsNumber; intersectionNumber++)
		{
			intersections[intersectionNumber].position = transformation.transform(intersections[intersectionNumber].position);
			intersections[intersectionNumber].normalVector = transformation.transform(intersections[intersectionNumber].normalVector);
		}

		intersectionsNumber = newIntersectionsNumber;
	}

	if (intersectionsNumber)
	{
		size_t closestIntersection = -1;
		float closestIntersectionDistance = 99999999999.0f;

		for (size_t i = 0; i < intersectionsNumber; i++)
		{
			float distance = (intersections[i].position - ray.begin).norm2();

			if (distance < closestIntersectionDistance)
			{
				closestIntersectionDistance = distance;
				closestIntersection = i;
			}
		}

		float angle = Vector(0.5, 0.5, -1).unitVector().dotProduct(intersections[closestIntersection].normalVector.unitVector());

		return fmaxf(angle, 0.0f);
	}
	else
	{
		return 0.0f;
	}
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

		float light = trace(ray, shapeSpaces, shapesNumber);

		image[index].x = light * 255;
		image[index].y = light * 255;
		image[index].z = light * 255;
		image[index].w = light * 255;
	}
}

void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Shape* shapes, size_t shapesNumber) {
	kernel << <1, N >> > (image, imageWidth, imageHeight, shapes, shapesNumber);
}