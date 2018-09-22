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
	namespace Tracing
	{
		__device__ float lightRayVisibility(Math::Vector viewRay, Math::Vector lightRay, float roughness)
		{
			float r = roughness;
			float xp = lightRay.dx;
			float yp = lightRay.dy;

			float dx = viewRay.dx;
			float dy = viewRay.dy;

			float a = dy;
			float b = dx;
			float c = 0;

			float d = std::abs(a * xp + b * yp + c) / std::sqrt(a * a + b * b) / roughness;

			if (d < 1)
				return std::sqrt(1 - d * d);
			else
				return 0;
		}

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

					ComponentIntersection closestIntersection(1.1f);
					for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
					{
						ComponentIntersection intersection = shapeComponents[componentNumber]->intersect(connectionRay, closestIntersection.distance);

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