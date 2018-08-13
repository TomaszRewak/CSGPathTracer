#pragma once

#include "kernel.h"
#include <math.h>
#include <new.h>

#include "PathTracer\component.hpp"

__device__
float trace(Ray& ray, Component* components, const size_t componentsNumber)
{
	float light = 0.f;
	float factor = 1.f;

	for (size_t iteration = 0; iteration < 3; iteration++)
	{
		Intersection closestIntersection;
		float closestIntersectionDistance = -1;

		for (size_t componentNumber = 0; componentNumber < componentsNumber; componentNumber++)
		{
			Component& component = components[componentNumber];

			Intersection intersection = component.intersect(ray);

			if (intersection.hit)
			{
				float intersectionDistance = (intersection.position - ray.begin).norm2();

				if (!closestIntersection.hit || intersectionDistance < closestIntersectionDistance)
				{
					closestIntersection = intersection;
					closestIntersectionDistance = intersectionDistance;
				}
			}
		}

		if (closestIntersection.hit)
		{
			float angle = Vector(0.5, 0.5, -1).unitVector().dotProduct(closestIntersection.normalVector.unitVector());

			bool shadow = false;
			for (size_t shadowComponentNumber = 0; !shadow && shadowComponentNumber < componentsNumber; shadowComponentNumber++)
			{
				if (components[shadowComponentNumber].intersect(Ray(
					closestIntersection.position + Vector(0.5, 0.5, -1) * 0.01,
					Vector(0.5, 0.5, -1)
				)).hit)
				{
					shadow = true;
				}
			}

			Vector normalVector = closestIntersection.normalVector.unitVector();

			ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
			ray.begin = closestIntersection.position + ray.direction * 0.01;

			light += factor * fmaxf(angle + 0.05, 0) * (shadow ? 0.5f : 1);
			factor *= 0.5;
		}
		else break;
	}

	return fminf(light, 1.f);
}

__device__
void copyShapesToSharedMemory(Component*& components, Shape* shapes, size_t shapesNumber)
{
	if (!threadIdx.x && !threadIdx.y) {
		components = (Component*)malloc(sizeof(Component) * shapesNumber);

		for (size_t i = 0; i < shapesNumber; i++)
		{
			Shape& shape = shapes[i];

			components[i].type = shape.type;
			components[i].normalDirection = 1.f;
			components[i].leftOperand = NULL;
			components[i].rightOperand = NULL;
			components[i].parent = NULL;
		}

		for (size_t i = 0; i < shapesNumber; i++)
		{
			Shape& shape = shapes[i];

			if (shape.leftOperand)
			{
				components[i].leftOperand = &components[shape.leftOperand];
				components[shape.leftOperand].parent = &components[i];
			}
			if (shape.rightOperand)
			{
				components[i].rightOperand = &components[shape.rightOperand];
				components[shape.rightOperand].parent = &components[i];
			}

			if (components[i].parent)
				components[i].globalTransformation = shape.localTransformation.combine(components[i].parent->globalTransformation);
			else
				components[i].globalTransformation = shape.localTransformation;

			if (components[i].parent)
				components[i].normalDirection = components[i].parent->normalDirection;
			if (components[i].parent && components[i].parent->type == ShapeType::Difference && components[i].parent->rightOperand == &components[i])
				components[i].normalDirection *= -1;
		}
	}
	__syncthreads();
}

__device__
void freeShapes(Component* components)
{
	__syncthreads();

	if (!threadIdx.x && !threadIdx.y) {
		free(components);
	}
}

__global__
void kernel(uchar4* image, size_t imageWidth, size_t imageHeight, Shape* shapes, size_t shapesNumber)
{
	__shared__ Component* components;
	copyShapesToSharedMemory(components, shapes, shapesNumber);

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < imageWidth && y < imageHeight)
	{
		float xi = (float)x - imageWidth * 0.5;
		float yi = (float)y - imageHeight * 0.5;

		float light = trace(Ray(Point(0, 0, -(float)imageWidth), Vector(xi, yi, imageWidth)), components, shapesNumber);

		size_t index = y * imageWidth + x;

		image[index].x = light * 255;
		image[index].y = light * 255;
		image[index].z = light * 255;
		image[index].w = light * 255;
	}

	freeShapes(components);
}

void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Shape* shapes, size_t shapesNumber) {
	dim3 block(8, 8, 1);
	dim3 grid(imageWidth / block.x + 1, imageHeight / block.y + 1, 1);

	kernel << <grid, block >> > (image, imageWidth, imageHeight, shapes, shapesNumber);
}