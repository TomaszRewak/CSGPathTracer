#pragma once

#include "../../Math/geometry.h"

namespace PathTracer
{
	struct Component;

	struct ComponentIntersection : public Math::Intersection
	{
		Math::Intersection intersection;
		const Component* component;

		__device__ ComponentIntersection() :
			intersection(),
			component(NULL)
		{ }

		__device__ explicit ComponentIntersection(float maxDistance) :
			Math::Intersection(maxDistance),
			component(NULL)
		{ }

		__device__ explicit ComponentIntersection(const Math::Intersection& intersection, const Component* component) :
			Math::Intersection(intersection),
			component(component)
		{ }
	};
}