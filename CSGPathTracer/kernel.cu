#pragma once

#include "kernel.h"

#include "PathTracer/rendering.h"
#include "PathTracer/rendering.h"
#include "PathTracer/scene.h"
#include "Communication/component-mapping.h"

namespace PathTracer
{
	__device__ void copyShapesToSharedMemory(
		Communication::Component* zippedComponents, size_t componentsNumber,
		Component*& components,
		Component**& rootComponents, size_t& rootComponentsNumber)
	{
		if (!threadIdx.x && !threadIdx.y) {
			Communication::mapComponents(
				zippedComponents, componentsNumber,
				components,
				rootComponents, rootComponentsNumber
			);
		}

		__syncthreads();
	}

	__device__ void freeShapes(
		Component* components,
		Component**& rootComponents)
	{
		__syncthreads();

		if (!threadIdx.x && !threadIdx.y) {
			free(components);
			free(rootComponents);
		}
	}

	__global__ void kernel(
		float4* image,
		const size_t imageWidth, const size_t imageHeight,
		Camera camera,
		Communication::Component* zippedComponents, size_t zippedComponentsNumber,
		size_t frameNumber, unsigned long long seed)
	{
		__shared__ Component* components;
		__shared__ Component** rootComponents; __shared__ size_t rootComponentsNumber;

		copyShapesToSharedMemory(
			zippedComponents, zippedComponentsNumber,
			components,
			rootComponents, rootComponentsNumber);

		PathTracer::Scene scene(rootComponents, rootComponentsNumber);

		size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		size_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < imageWidth && y < imageHeight)
		{
			size_t index = y * imageWidth + x;

			curandState randState;
			curand_init(seed + index, 0, 0, &randState); // worth remembering: curand_init(seed + index, 0, 0, &randState);

			Math::Ray ray = camera.getRay(x, y, imageWidth, imageHeight, randState);

			Shading::Color light = Rendering::shootRay<5, 2>(ray, scene, randState);

			image[index].x = light.r;
			image[index].y = light.g;
			image[index].z = light.b;
			image[index].w = 1.f / frameNumber;
		}

		freeShapes(components, rootComponents);
	}

	std::default_random_engine generator;

	void renderRect(
		float4* image,
		const size_t imageWidth, const size_t imageHeight,
		Camera camera,
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