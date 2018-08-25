#pragma once

#include <curand.h>

#include "../Communication/component.hpp"
#include "../../Math/shapes.hpp"

namespace PathTracer
{
	namespace Scene
	{
		struct Component
		{
			Common::ComponentType type;
			Math::TwoWayAffineTransformation globalTransformation;

			Shading::Shader shader;

			Component* parent;
			Component* leftOperand;
			Component* rightOperand;

			float normalDirection;

			__device__ Component() :
				parent(NULL),
				leftOperand(NULL),
				rightOperand(NULL),
				normalDirection(1.f)
			{ }
		};
	}
}