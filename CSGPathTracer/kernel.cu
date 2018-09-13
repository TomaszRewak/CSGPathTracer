#pragma once

#include "kernel.h"
#include <math.h>
#include <new.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <limits>

#include "PathTracer/scene.h"
#include "PathTracer/operations.h"
#include "Communication/component-mapping.h"

namespace PathTracer
{
	__device__ Shading::Color probeLightSources(
		Math::Point& position,
		Component** shapeComponents, size_t shapeComponentsNumber,
		Component** lightComponents, size_t lightComponentsNumber,
		curandState& curandState)
	{
		Shading::Color illumination;

		for (size_t source = 0; source < lightComponentsNumber; source++)
		{
			Component* lightComponent = lightComponents[source];

			for (size_t iteration = 0; iteration < 1; iteration++)
			{
				Math::Ray lightRay = lightComponent->generateRay(curandState);
				Math::Ray connectionRay = Math::Ray(position, lightRay.begin);

				Intersection closestIntersection(1.1f);
				for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
				{
					Intersection intersection = shapeComponents[componentNumber]->intersect(connectionRay, closestIntersection.distance);

					if (intersection.distance != INFINITY)
						closestIntersection = intersection;
				}

				if (closestIntersection.component == lightComponent)
				{
					auto shading = lightComponent->shader.getShading(closestIntersection.position);
					float angle = fmaxf(0.f, connectionRay.direction.unitVector().dotProduct(-closestIntersection.normalVector.unitVector()));

					illumination = illumination + shading.color * shading.emission * angle;
				}
			}
		}

		return illumination;
	}

	__device__ Shading::Color trace(
		Math::Ray& ray,
		Component** shapeComponents, size_t shapeComponentsNumber,
		Component** lightComponents, size_t lightComponentsNumber,
		curandState& curandState)
	{
#define MAX_RAY_DEPTH 5
		Shading::Color light;
		Shading::Filter filter;

		for (size_t iteration = 0; ; iteration++)
		{
			Intersection closestIntersection;

			for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
			{
				Intersection intersection = shapeComponents[componentNumber]->intersect(ray, closestIntersection.distance);

				if (intersection.distance < closestIntersection.distance)
					closestIntersection = intersection;
			}

			if (closestIntersection.distance != INFINITY)
			{
				Shading::Shading shading = closestIntersection.component->shader.getShading(closestIntersection.position);
				float randomNumber = curand_uniform(&curandState);

				if (shading.emission > 0)
				{
					light = light + filter * shading.color * shading.emission;
				}

				filter = filter * shading.color;

				if (iteration >= MAX_RAY_DEPTH || randomNumber > shading.reflectionProbability + shading.refractionProbability)
				{
					Math::Vector normalVector = closestIntersection.normalVector.unitVector();
					ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
					ray.begin = closestIntersection.position + ray.direction * 0.0001;

					light = light + filter * probeLightSources(ray.begin, shapeComponents, shapeComponentsNumber, lightComponents, lightComponentsNumber, curandState);

					return light;
				}
				else if (randomNumber < shading.reflectionProbability)
				{
					Math::Vector normalVector = closestIntersection.normalVector.unitVector();
					ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
					ray.begin = closestIntersection.position + ray.direction * 0.0001;
				}
				else
				{
					ray.begin = closestIntersection.position + ray.direction * 0.0001;
				}
			}
			else return light;
		}
	}

	__device__ void copyShapesToSharedMemory(
		Communication::Component* zippedComponents, size_t componentsNumber,
		Component*& components,
		Component**& shapeComponents, size_t& shapeComponentsNumber,
		Component**& lightComponents, size_t& lightComponentsNumber)
	{
		if (!threadIdx.x && !threadIdx.y) {
			shapeComponentsNumber = 0;
			lightComponentsNumber = 0;

			Communication::mapComponents(
				zippedComponents, componentsNumber,
				components,
				shapeComponents, shapeComponentsNumber,
				lightComponents, lightComponentsNumber
			);
		}

		__syncthreads();
	}

	__device__ void freeShapes(
		Component* components,
		Component**& shapeComponents,
		Component**& lightComponents)
	{
		__syncthreads();

		if (!threadIdx.x && !threadIdx.y) {
			free(components);
			free(shapeComponents);
			free(lightComponents);
		}
	}

	__global__ void kernel(
		float4* image,
		const size_t imageWidth, const size_t imageHeight,
		Rendering::Camera camera,
		Communication::Component* zippedComponents, size_t zippedComponentsNumber,
		size_t frameNumber, unsigned long long seed)
	{
		__shared__ Component* components;
		__shared__ Component** shapeComponents; __shared__ size_t shapeComponentsNumber;
		__shared__ Component** lightComponents; __shared__ size_t lightComponentsNumber;

		copyShapesToSharedMemory(
			zippedComponents, zippedComponentsNumber,
			components,
			shapeComponents, shapeComponentsNumber,
			lightComponents, lightComponentsNumber);

		size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		size_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < imageWidth && y < imageHeight)
		{
			size_t index = y * imageWidth + x;

			curandState randState;
			curand_init(seed + index, 0, 0, &randState); // worth remembering: curand_init(seed + index, 0, 0, &randState);

			Math::Ray ray = camera.getRay(x, y, imageWidth, imageHeight, randState);

			Shading::Color light = trace(
				ray,
				shapeComponents, shapeComponentsNumber,
				lightComponents, lightComponentsNumber,
				randState);

			image[index].x = ((double)image[index].x * frameNumber + light.r) / (frameNumber + 1);
			image[index].y = ((double)image[index].y * frameNumber + light.g) / (frameNumber + 1);
			image[index].z = ((double)image[index].z * frameNumber + light.b) / (frameNumber + 1);
			image[index].w = 1.f;
		}

		freeShapes(components, shapeComponents, lightComponents);
	}

	std::default_random_engine generator;

	void renderRect(
		float4* image,
		const size_t imageWidth, const size_t imageHeight,
		Rendering::Camera camera,
		Communication::Component* zippedComponents, size_t zippedComponentsNumber,
		size_t frameNumber)
	{
		dim3 block(16, 16, 1);
		dim3 grid(imageWidth / block.x + 1, imageHeight / block.y + 1, 1);

		std::uniform_int_distribution<unsigned long long> distribution(0, std::numeric_limits<unsigned long long>::max());
		unsigned long long seed = distribution(generator);

		kernel << <grid, block >> > (image, imageWidth, imageHeight, camera, zippedComponents, zippedComponentsNumber, frameNumber, seed);
	}
}