#pragma once

#include "component.hpp"
#include "../../Math/shapes.hpp"

namespace PathTracer
{
	namespace Scene
	{
		struct Intersection : public Math::Intersection
		{
			const Component* component;

			__device__ Intersection() :
				Math::Intersection(),
				component(NULL)
			{ }

			__device__ Intersection(const Math::Intersection& intersection, const Component* component) :
				Math::Intersection(intersection),
				component(component)
			{ }
		};
	}
}