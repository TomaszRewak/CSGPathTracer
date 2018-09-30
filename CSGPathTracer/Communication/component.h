#pragma once

#include "../Math/geometry.h"
#include "../PathTracer/Shading/shader.h"
#include "../PathTracer/Scene/component-type.h"

namespace Communication
{
	struct Component
	{
		PathTracer::ComponentType type;
		Math::AffineTransformation globalTransformation;

		PathTracer::Shading::Shader shader;

		size_t leftOperandOffset;
		size_t rightOperandOffset;

		__host__ Component() :
			type(PathTracer::ComponentType::Shape)
		{ }

		__host__ Component(
			PathTracer::ComponentType type,
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