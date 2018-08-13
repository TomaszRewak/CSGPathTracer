#pragma once

#include "kernel.h"
#include <math.h>
#include <new.h>

#include "PathTracer/Scene/component.hpp"

namespace PathTracer
{
	__device__
		float trace(Math::Ray& ray, Scene::Component* components, const size_t componentsNumber)
	{
		float light = 0.f;
		float factor = 1.f;

		for (size_t iteration = 0; iteration < 3; iteration++)
		{
			Scene::Intersection closestIntersection;

			for (size_t componentNumber = 0; componentNumber < componentsNumber; componentNumber++)
				components[componentNumber].intersect(ray, closestIntersection);

			if (closestIntersection.hit)
			{
				float angle = Math::Vector(0.5, 0.5, -1).unitVector().dotProduct(closestIntersection.normalVector.unitVector());

				/*Scene::Intersection shadowIntersection;
				bool shadow = false;

				for (size_t shadowComponentNumber = 0; !shadow && shadowComponentNumber < componentsNumber; shadowComponentNumber++)
				{
					components[shadowComponentNumber].intersect(Math::Ray(
						closestIntersection.position + Math::Vector(0.5, 0.5, -1) * 0.01,
						Math::Vector(0.5, 0.5, -1)
					), shadowIntersection);

					if (shadowIntersection.hit)
						shadow = true;
				}*/

				Math::Vector normalVector = closestIntersection.normalVector.unitVector();

				ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
				ray.begin = closestIntersection.position + ray.direction * 0.01;

				light += factor * fmaxf(angle + 0.05, 0) /** (shadow ? 0.5f : 1)*/;
				factor *= 0.5;
			}
			else break;
		}

		return fminf(light, 1.f);
	}

	__device__
		void copyShapesToSharedMemory(Scene::Component*& components, Communication::Component* shapes, size_t shapesNumber)
	{
		if (!threadIdx.x && !threadIdx.y) {
			components = new Scene::Component[shapesNumber];

			for (size_t i = 0; i < shapesNumber; i++)
			{
				Communication::Component& shape = shapes[i];

				components[i].type = shape.type;
				components[i].globalTransformation = shape.globalTransformation;

				if (shape.leftOperandOffset)
				{
					components[i].leftOperand = &components[i + shape.leftOperandOffset];
					components[i + shape.leftOperandOffset].parent = &components[i];
					components[i + shape.leftOperandOffset].normalDirection = components[i].normalDirection;
				}
				if (shape.rightOperandOffset)
				{
					components[i].rightOperand = &components[i + shape.rightOperandOffset];
					components[i + shape.rightOperandOffset].parent = &components[i];
					components[i + shape.rightOperandOffset].normalDirection = components[i].normalDirection;

					if (shape.type == Common::ComponentType::Difference)
						components[i + shape.rightOperandOffset].normalDirection *= -1;
				}
			}
		}
		__syncthreads();
	}

	__device__
		void freeShapes(Scene::Component* components)
	{
		__syncthreads();

		if (!threadIdx.x && !threadIdx.y) {
			free(components);
		}
	}

	__global__
		void kernel(uchar4* image, size_t imageWidth, size_t imageHeight, Communication::Component* shapes, size_t shapesNumber)
	{
		__shared__ Scene::Component* components;
		copyShapesToSharedMemory(components, shapes, shapesNumber);

		size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		size_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < imageWidth && y < imageHeight)
		{
			float xi = (float)x - imageWidth * 0.5;
			float yi = (float)y - imageHeight * 0.5;

			float light = trace(Math::Ray(Math::Point(0, 0, -(float)imageWidth), Math::Vector(xi, yi, imageWidth)), components, shapesNumber);

			size_t index = y * imageWidth + x;

			image[index].x = light * 255;
			image[index].y = light * 255;
			image[index].z = light * 255;
			image[index].w = light * 255;
		}

		freeShapes(components);
	}

	void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Communication::Component* components, size_t shapesNumber) {
		dim3 block(8, 8, 1);
		dim3 grid(imageWidth / block.x + 1, imageHeight / block.y + 1, 1);

		kernel << <grid, block >> > (image, imageWidth, imageHeight, components, shapesNumber);
	}
}