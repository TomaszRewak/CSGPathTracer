#pragma once

#include "../../Utils/math.hpp"

namespace PathTracer
{
	namespace Scene
	{
		struct Component;

		struct Intersection
		{
			bool hit;

			Math::Point position;
			Math::Vector normalVector;
			Component* component;

			__device__ Intersection() :
				hit(false),
				component(NULL)
			{ }

			__device__ Intersection(Math::Point position, Math::Vector normalVector, Component* component) :
				hit(true),
				position(position),
				normalVector(normalVector),
				component(component)
			{ }
		};
	}
}