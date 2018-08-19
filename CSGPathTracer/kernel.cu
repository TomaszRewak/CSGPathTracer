#pragma once

#include "kernel.h"
#include <math.h>
#include <new.h>
#include <curand.h>
#include <curand_kernel.h>

#include "PathTracer/Scene/intersection-finder.hpp"
#include "PathTracer/Scene/light-sources.hpp"

namespace PathTracer
{
	__device__ float probeLightSources(
		Math::Point& position,
		Scene::Component** shapeComponents, size_t shapeComponentsNumber,
		Scene::Component** lightComponents, size_t lightComponentsNumber,
		curandState& curandState)
	{
		float illumination = 0;

		for (size_t source = 0; source < lightComponentsNumber; source++)
		{
			Scene::Component* lightComponent = lightComponents[source];

			for (size_t iteration = 0; iteration < 1; iteration++)
			{
				Math::Ray lightRay = Scene::LightSources::generateLightRay(lightComponent, curandState);
				Math::Ray connectionRay = Math::Ray(position, lightRay.begin);

				Scene::Intersection closestIntersection(1.1f);
				for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
					Scene::IntersectionFinder::intersect(shapeComponents[componentNumber], connectionRay, closestIntersection);

				if (closestIntersection.component == lightComponent)
					illumination += 1.f;
			}
		}

		return fminf(illumination / lightComponentsNumber, 1.0f);
	}

	__device__ float trace(
		Math::Ray& ray,
		Scene::Component** shapeComponents, size_t shapeComponentsNumber,
		Scene::Component** lightComponents, size_t lightComponentsNumber,
		curandState& curandState)
	{
		float light = 0.f;
		float factor = 1.f;

		for (size_t iteration = 0; iteration < 2; iteration++)
		{
			Scene::Intersection closestIntersection;

			for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
				Scene::IntersectionFinder::intersect(shapeComponents[componentNumber], ray, closestIntersection);

			if (closestIntersection.distance != INFINITY)
			{
				float angle = Math::Vector(0.5, 0.5, -1).unitVector().dotProduct(closestIntersection.normalVector.unitVector());

				Math::Vector normalVector = closestIntersection.normalVector.unitVector();

				ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
				ray.begin = closestIntersection.position + ray.direction * 0.0001;

				float illumination = probeLightSources(ray.begin, shapeComponents, shapeComponentsNumber, lightComponents, lightComponentsNumber, curandState);

				illumination = 0.5f + illumination * 0.5f;

				light += (0.1 + fmaxf(angle, 0) * 0.2 + illumination * 0.5) * factor ;
				factor *= 0.25;
			}
			else break;
		}

		return fminf(light, 1.f);
	}

	__device__ void copyShapesToSharedMemory(
		Scene::Component*& components,
		Scene::Component**& shapeComponents,
		Scene::Component**& lightComponents,
		Communication::Component* zippedComponents,
		size_t componentsNumber,
		size_t& shapeComponentsNumber,
		size_t& lightComponentsNumber)
	{
		if (!threadIdx.x && !threadIdx.y) {
			shapeComponentsNumber = 0;
			lightComponentsNumber = 0;

			components = new Scene::Component[componentsNumber];
			shapeComponents = new Scene::Component*[componentsNumber];
			lightComponents = new Scene::Component*[componentsNumber];

			for (size_t i = 0; i < componentsNumber; i++)
			{
				Communication::Component& shape = zippedComponents[i];

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

				if (shape.type == Common::ComponentType::Sphere || shape.type == Common::ComponentType::Cylinder || shape.type == Common::ComponentType::Plane)
					shapeComponents[shapeComponentsNumber++] = &components[i];

				if (shape.type == Common::ComponentType::Sphere && components[i].parent == NULL) // just a temporary hack
					lightComponents[lightComponentsNumber++] = &components[i];
			}
		}
		__syncthreads();
	}

	__device__ void freeShapes(
		Scene::Component* components,
		Scene::Component**& shapeComponents,
		Scene::Component**& lightComponents)
	{
		__syncthreads();

		if (!threadIdx.x && !threadIdx.y) {
			free(components);
			free(shapeComponents);
			free(lightComponents);
		}
	}

	__global__ void kernel(uchar4* image, size_t imageWidth, size_t imageHeight, Communication::Component* zippedComponents, size_t componentsNumber, size_t hashedFrameNumber)
	{
		__shared__ Scene::Component* components;
		__shared__ Scene::Component** shapeComponents; __shared__ size_t shapeComponentsNumber;
		__shared__ Scene::Component** lightComponents; __shared__ size_t lightComponentsNumber;

		copyShapesToSharedMemory(components, shapeComponents, lightComponents, zippedComponents, componentsNumber, shapeComponentsNumber, lightComponentsNumber);

		size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		size_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < imageWidth && y < imageHeight)
		{
			size_t index = y * imageWidth + x;

			curandState randState;
			curand_init(index + hashedFrameNumber, 0, 0, &randState);

			float light = 0;

#define RaySamples 2
			for (size_t sample = 0; sample < RaySamples; sample++)
			{
				float xi = (float)x - imageWidth * 0.5 + curand_uniform(&randState) - 0.5;
				float yi = (float)y - imageHeight * 0.5 + curand_uniform(&randState) - 0.5;

				light += trace(
					Math::Ray(Math::Point(0, 0, -(float)imageWidth), Math::Vector(xi, yi, imageWidth)),
					shapeComponents, shapeComponentsNumber,
					lightComponents, lightComponentsNumber, 
					randState);
			}

			light /= RaySamples;

			image[index].x = light * 255;
			image[index].y = light * 255;
			image[index].z = light * 255;
			image[index].w = light * 255;
		}

		freeShapes(components, shapeComponents, lightComponents);
	}

	void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Communication::Component* components, size_t shapesNumber, size_t hashedFrameNumber) {
		dim3 block(8, 8, 1);
		dim3 grid(imageWidth / block.x + 1, imageHeight / block.y + 1, 1);

		kernel << <grid, block >> > (image, imageWidth, imageHeight, components, shapesNumber, hashedFrameNumber);
	}
}