#pragma once

#include "../../Utils/math.hpp"

namespace PathTracer
{
	namespace Scene
	{
		struct Component;

		struct Intersection
		{
			Math::Point position;
			Math::Vector normalVector;

			float distance2;

			Component* component;

			__device__ Intersection() :
				component(NULL),
				distance2(INFINITY)
			{ }

			__device__ Intersection(Math::Point rayBegin, Math::Point position, Math::Vector normalVector, Component* component) :
				position(position),
				normalVector(normalVector),
				distance2((position - rayBegin).norm2()),
				component(component)
			{ }
		};
	}
}