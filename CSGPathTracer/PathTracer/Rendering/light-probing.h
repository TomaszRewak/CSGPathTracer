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
			const Tracing::PathStep* viewRaySteps, size_t viewRayStepsNumber,
			const Tracing::PathStep* lightRaySteps, size_t lightRayStepsNumber,
			const Scene &scene)
		{
			Shading::Color illumination;

			for (size_t viewRayStepIndex = 0; viewRayStepIndex < viewRayStepsNumber; viewRayStepIndex++)
			{
				for (size_t lightRayStepIndex = 0; lightRayStepIndex < lightRayStepsNumber; lightRayStepIndex++)
				{
					const Tracing::PathStep& viewRayStep = viewRaySteps[viewRayStepIndex];
					const Tracing::PathStep& lightRayStep = lightRaySteps[lightRayStepIndex];

					bool visible = scene.hitsObstacle(
						Math::Ray(viewRayStep.ray.begin, lightRayStep.ray.begin)
					);

					if (visible)
					{
						float visibilityFactor =
							lightRayVisibilityFactor(viewRayStep.ray.direction, lightRayStep.ray.direction, lightRayStep.roughness) *
							lightRayVisibilityFactor(lightRayStep.ray.direction, viewRayStep.ray.direction, viewRayStep.roughness);

						illumination = illumination + Shading::Filter(viewRayStep.color) * lightRayStep.color * visibilityFactor;
					}

				}
			}

			return illumination;
		}
	}
}