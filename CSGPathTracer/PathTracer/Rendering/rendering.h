#pragma once

#include "../Tracing/path-tracing.h"
#include "../Scene/component.h"
#include "light-probing.h"

namespace PathTracer
{
	namespace Rendering 
	{
		template<size_t MaxViewDepth, size_t MaxLightDepth>
		__device__ Shading::Color shootRay(
			const Math::Ray& ray,
			Component** shapeComponents, size_t shapeComponentsNumber,
			Component** lightComponents, size_t lightComponentsNumber,
			curandState& curandState)
		{
			Shading::Color color;

			Tracing::PathStep viewRaySteps[MaxViewDepth];
			size_t viewDepth = Tracing::trace<MaxViewDepth>(viewRaySteps, ray, shapeComponents, shapeComponentsNumber, curandState);

			for (size_t lightComponentIndex = 0; lightComponentIndex < lightComponentsNumber; lightComponentIndex++)
			{
				Component* lightComponent = lightComponents[lightComponentIndex];
				Math::Ray lightRay = lightComponent->generateRay(curandState);

				Tracing::PathStep lightRaySteps[MaxLightDepth];

				lightRaySteps[0] = Tracing::PathStep(
					lightComponent->shader.getShading(lightRay.begin),
					lightRay
				);
				size_t lightDepth = 1 + Tracing::trace<MaxLightDepth - 1>(lightRaySteps + 1, lightRay, shapeComponents, shapeComponentsNumber, curandState);

				color = color + probeLight<MaxViewDepth, MaxLightDepth>(
					viewRaySteps, viewDepth,
					lightRaySteps, lightDepth,
					shapeComponents, shapeComponentsNumber
				);
			}

			return color;
		}
	}
}