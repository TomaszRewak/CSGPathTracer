#pragma once

#include "component.h"

namespace PathTracer
{
	struct Scene
	{
	private:
		Component** rootComponents;
		size_t rootComponentsNumber;

	public:
		float totalPhotons;

		__device__ Scene(
			Component** rootComponents, size_t rootComponentsNumber
		) :
			rootComponents(rootComponents),
			rootComponentsNumber(rootComponentsNumber)
		{ 
			totalPhotons = 0;

			for (size_t i = 0; i < rootComponentsNumber; i++)
			{
				totalPhotons += rootComponents[i]->totalPhotons;
			}
		}

		__device__ ComponentIntersection intersect(const Math::Ray &ray) const
		{
			ComponentIntersection intersection;

			for (size_t i = 0; i < rootComponentsNumber; i++)
			{
				rootComponents[i]->intersect(ray, intersection);
			}
			
			return intersection;
		}

		__device__ ComponentPhoton generatePhoton(curandState& rand) const
		{
			float photonsLeft = totalPhotons;
			ComponentPhoton photon;

			for (size_t i = 0; i < rootComponentsNumber; i++)
			{
				rootComponents[i]->generatePhoton(photon, photonsLeft, rand);
			}

			return photon;
		}

		__device__ bool hitsObstacle(const Math::Ray &ray) const
		{
			return intersect(ray).distance < 1;
		}
	};
}