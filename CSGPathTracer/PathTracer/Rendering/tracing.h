#pragma once

#include "lightProbing.h"

namespace PathTracer
{
	namespace Rendering {
		__device__ Shading::Color trace(
			Math::Ray& ray,
			Component** shapeComponents, size_t shapeComponentsNumber,
			Component** lightComponents, size_t lightComponentsNumber,
			curandState& curandState)
		{
			const int maxRayDepth = 2;
			const float rayEpsylon = 0.001;

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

					if (iteration >= maxRayDepth || randomNumber > shading.reflectionProbability + shading.refractionProbability)
					{
						Math::Vector normalVector = closestIntersection.normalVector.unitVector();
						ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
						ray.begin = closestIntersection.position + ray.direction * rayEpsylon;

						light = light + filter * probeLightSources(ray.begin, shapeComponents, shapeComponentsNumber, lightComponents, lightComponentsNumber, curandState);

						return light;
					}
					else if (randomNumber < shading.reflectionProbability)
					{
						Math::Vector normalVector = closestIntersection.normalVector.unitVector();
						ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
						ray.begin = closestIntersection.position + ray.direction * rayEpsylon;
					}
					else
					{
						ray.begin = closestIntersection.position + ray.direction * rayEpsylon;
					}
				}
				else return light;
			}
		}
	}
}