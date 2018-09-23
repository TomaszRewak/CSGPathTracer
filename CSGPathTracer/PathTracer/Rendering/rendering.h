#pragma once

#include "../Tracing/path-tracing.h"
#include "../Scene/component.h"

namespace PathTracer
{
	namespace Rendering {
		template<size_t MaxViewDepth, size_t MaxLightDepth>
		__device__ Shading::Color shootRay(
			const Math::Ray& ray,
			Component** shapeComponents, size_t shapeComponentsNumber,
			Component** lightComponents, size_t lightComponentsNumber,
			curandState& curandState)
		{
			Tracing::PathStep viewRaySteps[MaxViewDepth];
			size_t viewDepth = Tracing::trace<MaxViewDepth>(viewRaySteps, ray, shapeComponents, shapeComponentsNumber, curandState);

			for (size_t lightComponentIndex; lightComponentIndex < lightComponentsNumber; lightComponentIndex++)
			{
				Component* lightComponent = lightComponents[lightComponentIndex];
				Math::Ray lightRay = lightComponent->generateRay(curandState);

				Tracing::PathStep lightRaySteps[MaxLightDepth + 1];

				lightRaySteps[0] = Tracing::PathStep(
					lightComponent->shader.getShading(lightRay.begin),
					lightRay
				);
				size_t lightDepth = Tracing::trace<MaxLightDepth>(lightRaySteps + 1, lightRay, shapeComponents, shapeComponentsNumber, curandState);

				//merge paths
			}






			const int maxRayDepth = 2;
			const float rayEpsylon = 0.001;

			Shading::Color light;
			Shading::Filter filter;

			for (size_t iteration = 0; ; iteration++)
			{
				ComponentIntersection closestIntersection;

				for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
				{
					ComponentIntersection intersection = shapeComponents[componentNumber]->intersect(ray, closestIntersection.distance);

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