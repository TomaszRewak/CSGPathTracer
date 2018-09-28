#pragma once

#include "path-step.h"
#include "../Scene/component-intersection.h"

namespace PathTracer
{
	namespace Tracing {
		template<size_t MAX_DEPTH>
		__device__ size_t trace(
			PathStep* steps,
			Math::Ray ray,
			Component** shapeComponents, size_t shapeComponentsNumber,
			curandState& curandState)
		{
			const float rayEpsylon = 0.001;

			for (size_t depth = 0; depth < MAX_DEPTH; depth++)
			{
				ComponentIntersection closestIntersection;

				for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
				{
					ComponentIntersection intersection = shapeComponents[componentNumber]->intersect(ray, closestIntersection.distance);

					if (intersection.distance < closestIntersection.distance)
						closestIntersection = intersection;
				}

				if (closestIntersection.component == NULL)
					return depth;

				PathStep& step = steps[depth];
				step.shading = closestIntersection.component->shader.getShading(closestIntersection.position);

				Math::Vector normalVector = closestIntersection.normalVector.unitVector();

				if (curand_uniform(&curandState) < step.shading.translucency)
				{
					step.baseRay.direction = ray.direction;
					normalVector = -normalVector;
				}
				else if (curand_uniform(&curandState) < step.shading.reflectance)
				{
					step.baseRay.direction = ray.direction - normalVector * 2 * (ray.direction.dotProduct(normalVector));
				}
				else
					return depth;

				Math::Vector roughnessVector = Math::Vector(
					curand_uniform(&curandState) - 0.5,
					curand_uniform(&curandState) - 0.5,
					curand_uniform(&curandState) - 0.5
				);

				if (roughnessVector.dotProduct(normalVector) < 0)
					roughnessVector = -roughnessVector;

				ray.direction = (
					step.baseRay.direction.unitVector() * (1.f - step.shading.roughness) +
					roughnessVector.unitVector() * step.shading.roughness
					).unitVector();

				step.baseRay.begin = closestIntersection.position + step.baseRay.direction * rayEpsylon;
				ray.begin = closestIntersection.position + ray.direction * rayEpsylon;
			}

			return MAX_DEPTH;
		}
	}
}