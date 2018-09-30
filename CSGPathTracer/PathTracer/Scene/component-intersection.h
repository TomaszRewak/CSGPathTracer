#pragma once

#include "../../Math/geometry.h"

namespace PathTracer
{
	struct Component;

	struct ComponentIntersection
	{
		const Component* component;
		float distance;

		__device__ ComponentIntersection():
			component(NULL),
			distance(INFINITY)
		{ }

		__device__ ComponentIntersection(float maxDisntace) :
			component(NULL),
			distance(maxDisntace)
		{ }

		__device__ ComponentIntersection(Component* component, float maxDisntace) :
			component(component),
			distance(maxDisntace)
		{ }
	};
}