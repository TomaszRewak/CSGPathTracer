#pragma once

#include "component-type.h"
#include "../Math/geometry.h"
#include "../PathTracer/Shading/shader.hpp"

namespace Communication
{
	struct Component
	{
		ComponentType type;
		Math::AffineTransformation globalTransformation;

		PathTracer::Shading::Shader shader;

		size_t leftOperandOffset;
		size_t rightOperandOffset;

		__host__ Component() :
			type(ComponentType::None)
		{ }

		__host__ Component(
			ComponentType type,
			Math::AffineTransformation globalTransformation,
			size_t leftOperandOffset,
			size_t rightOperandOffset,
			PathTracer::Shading::Shader shader
		) :
			type(type),
			globalTransformation(globalTransformation),
			leftOperandOffset(leftOperandOffset),
			rightOperandOffset(rightOperandOffset),
			shader(shader)
		{}
	};
}