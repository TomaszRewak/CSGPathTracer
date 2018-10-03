#pragma once

#include "../../Math/geometry.h"

namespace PathTracer
{
	struct Component;

	struct ComponentPhoton
	{
		float strength;
		const Component* component;
		Math::Ray ray;

		__device__ ComponentPhoton():
			strength(0.),
			component(NULL)
		{ }
	};
}