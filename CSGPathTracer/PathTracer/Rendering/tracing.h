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

					light = light + filter * shading.color * shading.emission;
					filter = filter * shading.color;

					if (iteration >= maxRayDepth)
					{
						Math::Vector normalVector = closestIntersection.normalVector.unitVector();
						ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
						ray.begin = closestIntersection.position + ray.direction * rayEpsylon;

						light = light + filter * probeLightSources(ray.begin, shapeComponents, shapeComponentsNumber, lightComponents, lightComponentsNumber, curandState);

						return light;
					}
					else if (curand_uniform(&curandState) < shading.translucency)
					{
						ray.begin = closestIntersection.position + ray.direction * rayEpsylon;
					}
					else if (curand_uniform(&curandState) < shading.reflectance)
					{
						Math::Vector normalVector = closestIntersection.normalVector.unitVector();

						Math::Vector roughnessVector = Math::Vector(
							curand_uniform(&curandState),
							curand_uniform(&curandState),
							curand_uniform(&curandState)
						);

						if (roughnessVector.dotProduct(normalVector) < 0)
							roughnessVector = -roughnessVector;

						ray.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
						ray.direction = (
							ray.direction.unitVector() * (1.f - shading.roughness) +
							roughnessVector.unitVector() * shading.roughness
						).unitVector();
						ray.begin = closestIntersection.position + ray.direction * rayEpsylon;
					}
					else
					{
						return light;
					}
				}
				else return light;
			}
		}
	}
}