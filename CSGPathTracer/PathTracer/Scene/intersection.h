#pragma once

#include "../../Math/geometry.h"

namespace PathTracer
{
	struct Component;

	struct Intersection : public Math::Intersection
	{
		Math::Intersection intersection;
		const Component* component;

		__device__ Intersection() :
			intersection(),
			component(NULL)
		{ }

		__device__ explicit Intersection(float maxDistance) :
			Math::Intersection(maxDistance),
			component(NULL)
		{ }

		__device__ Intersection(const Math::Intersection& intersection, const Component* component) :
			Math::Intersection(intersection),
			component(component)
		{ }
	};
}