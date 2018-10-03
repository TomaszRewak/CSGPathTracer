#pragma once

#include "path-step.h"
#include "../Scene/component-intersection.h"

namespace PathTracer
{
	namespace Tracing {
		template<size_t MAX_DEPTH>
		__device__ size_t trace(
			PathStep* steps,
			const Scene& scene,
			curandState& curandState)
		{
			steps[0].ray.begin = steps[0].ray.begin + steps[0].ray.direction.unitVector() * RAY_EPSYLON;
			Math::Ray ray = steps[0].ray;

			for (size_t depth = 1; depth < MAX_DEPTH; depth++)
			{
				ComponentIntersection closestIntersection = scene.intersect(ray);

				if (closestIntersection.component == NULL) // SYNC
				{
					if (false)
						continue;
					else
						return depth;
				}

				Math::Point intersectionPoint = ray.point(closestIntersection.distance);
				Math::Vector direction = ray.direction.unitVector();
				Math::Vector normalVector = closestIntersection.component->normalVector(intersectionPoint).unitVector();
				Shading::Shading shading = closestIntersection.component->shader.getShading(intersectionPoint);

				Math::Vector baseDirection = direction;

				float densityFactor = 1 / shading.density;
				float dotProduct = direction.dotProduct(normalVector);

				if (dotProduct > 0)
				{
					normalVector = -normalVector;
					dotProduct = direction.dotProduct(normalVector);
					densityFactor = 1 / densityFactor;
				}

				double refractionFactor = 1 - densityFactor * densityFactor * (1 - dotProduct * dotProduct);

				if (refractionFactor < 0 || curand_uniform(&curandState) < shading.reflectance)
				{
					baseDirection = direction - normalVector * 2 * dotProduct;
				}
				else
				{
					baseDirection = direction * densityFactor - normalVector * densityFactor * dotProduct - normalVector * sqrt(refractionFactor);
				}
				baseDirection = baseDirection.unitVector();

				Math::Vector roughnessVector = Math::Vector(
					curand_uniform(&curandState) - 0.5,
					curand_uniform(&curandState) - 0.5,
					curand_uniform(&curandState) - 0.5
				).unitVector();

				if (roughnessVector.dotProduct(normalVector) < 0)
					roughnessVector = -roughnessVector;

				ray.direction = (
					baseDirection * (1.f - shading.roughness) +
					roughnessVector * shading.roughness
					).unitVector();
				ray.begin = intersectionPoint + ray.direction * RAY_EPSYLON;

				PathStep& step = steps[depth];
				step.ray.direction = baseDirection;
				step.ray.begin = intersectionPoint + step.ray.direction * RAY_EPSYLON;
				step.color = Shading::Filter(steps[depth - 1].color) * shading.color;
				step.roughness = shading.roughness;
			}

			return MAX_DEPTH;
		}
	}
}