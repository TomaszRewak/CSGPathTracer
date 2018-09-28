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

		__device__ bool lightRayVisibility(
			Component** shapeComponents, size_t shapeComponentsNumber,
			Math::Point viewPoint,
			Math::Point lightPoint)
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

					bool visible = lightRayVisibility(
						shapeComponents, shapeComponentsNumber,
						viewRayStep.baseRay.begin,
						lightRayStep.baseRay.begin
					);

					if (visible)
					{
						float visibilityFactor =
							lightRayVisibilityFactor(viewRayStep.baseRay.direction, lightRayStep.baseRay.direction, viewRayStep.shading.roughness) *
							lightRayVisibilityFactor(viewRayStep.baseRay.direction, lightRayStep.baseRay.direction, viewRayStep.shading.roughness);

						illumination = illumination + lightRayStepsIllumination[lightRayStepIndex] * visibilityFactor;
					}

				}

				illumination = Shading::Filter(viewRayStep.shading.color) * illumination + viewRayStep.shading.color * viewRayStep.shading.emission;
			}

			return illumination;
		}
	}
}