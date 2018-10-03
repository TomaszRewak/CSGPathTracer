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
			Tracing::PathStep viewRaySteps[MaxViewDepth];
			viewRaySteps[0] = Tracing::PathStep(
				Shading::Color(1, 1, 1),
				ray,
				0.1);
			size_t viewDepth = Tracing::trace<MaxViewDepth>(
				viewRaySteps,
				scene,
				curandState);

			ComponentPhoton photon = scene.generatePhoton(curandState);
			Shading::Shading lightSourceShading = photon.component->shader.getShading(photon.ray.begin);

			Tracing::PathStep lightRaySteps[MaxLightDepth];
			lightRaySteps[0] = Tracing::PathStep(
				lightSourceShading.color * photon.strength * lightSourceShading.emission,
				ray,
				lightSourceShading.emission);
			size_t lightDepth = Tracing::trace<MaxLightDepth>(
				lightRaySteps,
				scene,
				curandState);

			return probeLight(
				viewRaySteps, viewDepth,
				lightRaySteps, lightDepth,
				scene
			);
		}
	}
}