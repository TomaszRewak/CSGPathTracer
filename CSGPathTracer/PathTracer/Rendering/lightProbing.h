#pragma once

#include <math.h>
#include <new.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <limits>

#include "../Scene/component.h"

namespace PathTracer
{
	namespace Rendering
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
	}
}