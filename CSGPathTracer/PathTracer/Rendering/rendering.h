#pragma once

#include "../Tracing/path-tracing.h"
#include "../Scene/scene.h"
#include "light-probing.h"

namespace PathTracer
{
	namespace Rendering
	{
		template<size_t MaxViewDepth, size_t MaxLightDepth>
		__device__ Shading::Color shootRay(
			const Math::Ray& ray,
			const Scene& scene,
			curandState& curandState)
		{
			ComponentPhoton photon = scene.generatePhoton(curandState);
			Shading::Shading lightSourceShading = photon.component->shader.getShading(photon.ray.begin);

			Tracing::PathStep lightRaySteps[MaxLightDepth];
			lightRaySteps[0] = Tracing::PathStep(photon.component, photon.ray, lightSourceShading);
			for (size_t i = 1; i < MaxLightDepth; i++)
				lightRaySteps[i] = Tracing::trace(
					lightRaySteps[i - 1],
					scene,
					curandState);

			Tracing::PathStep viewStep = Tracing::PathStep(NULL, ray, Shading::Shading(Shading::Color(1.f, 1.f, 1.f), 0.001f));

			Shading::Color color;
			Shading::Color filter(1.f);

			for (size_t i = 0; i < MaxViewDepth; i++)
			{
				viewStep = Tracing::trace(
					viewStep,
					scene,
					curandState);

				Shading::Color stepColor = probeLight(
					viewStep,
					lightRaySteps, MaxLightDepth,
					scene
				);

				color = color + filter * stepColor + filter * (viewStep.shading.color * viewStep.shading.emission);
				filter = filter * viewStep.shading.color;
			}

			return color;
		}
	}
}