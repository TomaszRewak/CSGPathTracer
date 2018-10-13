#pragma once

#include <math.h>
#include <new.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <limits>

#include "../Tracing/path-tracing.h"
#include "../Scene/component.h"

namespace PathTracer
{
	namespace Rendering
	{
		__device__ float lightRayVisibilityFactor(
			const Math::Vector& viewRay,
			const Math::Vector& lightRay,
			float roughness)
		{
			float d = (lightRay.unitVector() * (1 - roughness)).crossProduct(viewRay.unitVector()).norm() / roughness;

			if (d < 1)
				return std::sqrt(1 - d * d);
			else
				return 0;
		}

		__device__ Shading::Color probeLight(
			const Tracing::PathStep& viewRayStep,
			const Tracing::PathStep* lightRaySteps, size_t lightRayStepsNumber,
			const Scene &scene)
		{
			Shading::Color filter(lightRaySteps[0].shading.emission);
			Shading::Color illumination;

			for (size_t lightRayStepIndex = 0; lightRayStepIndex < lightRayStepsNumber; lightRayStepIndex++)
			{
				const Tracing::PathStep& lightRayStep = lightRaySteps[lightRayStepIndex];

				filter = filter * lightRayStep.shading.color;

				ComponentIntersection intersection = scene.intersect(
					Math::Ray(viewRayStep.ray.begin, lightRayStep.ray.begin)
				);

				if (intersection.distance > 1 || intersection.component == lightRayStep.component)
				{
					float visibilityFactor = 0.5;
						lightRayVisibilityFactor(viewRayStep.ray.direction, lightRayStep.ray.direction, lightRayStep.shading.roughness) *
						lightRayVisibilityFactor(lightRayStep.ray.direction, viewRayStep.ray.direction, viewRayStep.shading.roughness);
					
					illumination = illumination + filter * viewRayStep.shading.color * visibilityFactor * (1 / scene.totalPhotons);
				}
			}

			return illumination;
		}
	}
}