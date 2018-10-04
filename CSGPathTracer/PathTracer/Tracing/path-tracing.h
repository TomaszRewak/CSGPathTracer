#pragma once

#include "path-step.h"
#include "../Scene/component-intersection.h"

namespace PathTracer
{
	namespace Tracing {
		__device__ Math::Ray getNextRay(const PathStep& step, curandState& curandState)
		{
			float roughness = step.shading.roughness;

			Math::Vector roughnessVector = Math::Vector(
				curand_uniform(&curandState) - 0.5,
				curand_uniform(&curandState) - 0.5,
				curand_uniform(&curandState) - 0.5
			).unitVector();

			if (roughnessVector.dotProduct(step.normalVector) < 0)
				roughnessVector = -roughnessVector;

			return Math::Ray(
				step.ray.begin,
				(step.ray.direction * (1.f - roughness) + roughnessVector * roughness).unitVector()
			);
		}

		__device__ PathStep trace(
			const PathStep& previousStep,
			const Scene& scene,
			curandState& curandState)
		{
			Math::Ray ray = getNextRay(previousStep, curandState);

			ComponentIntersection closestIntersection = scene.intersect(ray);

			if (closestIntersection.component == NULL)
				return PathStep();

			Math::Point intersectionPoint = ray.point(closestIntersection.distance);
			Math::Vector direction = ray.direction.unitVector();
			Math::Vector normalVector = closestIntersection.component->normalVector(intersectionPoint).unitVector();
			Shading::Shading shading = closestIntersection.component->shader.getShading(intersectionPoint);

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
				direction = direction - normalVector * 2 * dotProduct;
			}
			else
			{
				direction = direction * densityFactor - normalVector * densityFactor * dotProduct - normalVector * sqrt(refractionFactor);
				// normalVector = -normalVector
			}

			return PathStep(
				closestIntersection.component,
				Math::Ray(
					intersectionPoint + direction.unitVector() * RAY_EPSYLON,
					direction
				),
				normalVector,
				shading
			);
		}
	}
}