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
	namespace Tracing
	{
		__device__ float lightRayVisibilityFactor(
			Math::Vector viewRay,
			Math::Vector lightRay,
			float roughness)
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

		__device__ bool lightRayVisibility(
			Component** shapeComponents, size_t shapeComponentsNumber,
			Math::Point viewPoint,
			Math::Point lightPoint,
			float roughness)
		{
			Math::Ray connectionRay = Math::Ray(viewPoint, lightPoint);

			ComponentIntersection closestIntersection(0.99f); // TODO: wprowadziæ sta³¹
			for (size_t componentNumber = 0; componentNumber < shapeComponentsNumber; componentNumber++)
			{
				ComponentIntersection intersection = shapeComponents[componentNumber]->intersect(connectionRay, closestIntersection.distance);

				if (intersection.distance != INFINITY)
					return false;
			}

			return true;
		}

		template<size_t MaxViewDepth, size_t MaxLightDepth>
		__device__ Shading::Color probeLight(
			const Tracing::PathStep* viewRaySteps, size_t viewRayStepsNumber,
			const Tracing::PathStep* lightRaySteps, size_t lightRayStepsNumber,
			Component** shapeComponents, size_t shapeComponentsNumber)
		{
			Shading::Color lightRayStepsIllumination[MaxLightDepth];

			lightRayStepsIllumination[0] = lightRaySteps[0].shading.color * lightRaySteps[0].shading.emission;

			for (size_t lightRayStepIndex = 1; lightRayStepIndex < lightRayStepsNumber; lightRayStepIndex++)
			{
				const Tracing::PathStep& lightRayStep = lightRaySteps[lightRayStepIndex];

				lightRayStepsIllumination[lightRayStepIndex] =
					Shading::Filter(lightRayStep.shading.color) * lightRayStepsIllumination[lightRayStepIndex - 1] +
					lightRayStep.shading.color * lightRayStep.shading.emission;
			}

			Shading::Color illumination;

			for (size_t viewRayStepIndex = 0; viewRayStepIndex < viewRayStepsNumber; viewRayStepIndex++)
			{
				const Tracing::PathStep& viewRayStep = viewRaySteps[viewRayStepsNumber - viewRayStepIndex - 1];

				for (size_t lightRayStepIndex = 0; lightRayStepIndex < lightRayStepsNumber; lightRayStepIndex++)
				{
					const Tracing::PathStep& lightRayStep = lightRaySteps[lightRayStepIndex];

					float visibility = 0;

					if (lightRayVisibility())
					{
						visibility =
							lightRayVisibilityFactor(viewRayStep.baseRay, lightRayStep.baseRay, viewRayStep.shading.roughness) *
							lightRayVisibilityFactor(viewRayStep.baseRay, lightRayStep.baseRay, viewRayStep.shading.roughness);
					}

					illumination = illumination + lightRayStepsIllumination[lightRayStepIndex] * visibility;
				}

				illumination = Shading::Filter(viewRayStep.shading.color) * illumination;
			}

			return illumination;
		}
	}
}