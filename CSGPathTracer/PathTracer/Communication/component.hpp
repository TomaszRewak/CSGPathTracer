#pragma once

#include "../Common/component-type.hpp"
#include "../../Math/math.hpp"
#include "../Shading/shader.hpp"

namespace PathTracer
{
	namespace Communication
	{
		struct Component
		{
			Common::ComponentType type;
			Math::AffineTransformation globalTransformation;

			Shading::Shader shader;

			size_t leftOperandOffset;
			size_t rightOperandOffset;

			__host__ Component() :
				type(Common::ComponentType::None)
			{ }

			__host__ Component(
				Common::ComponentType type,
				Math::AffineTransformation globalTransformation,
				size_t leftOperandOffset,
				size_t rightOperandOffset,
				Shading::Shader shader
			) :
				type(type),
				globalTransformation(globalTransformation),
				leftOperandOffset(leftOperandOffset),
				rightOperandOffset(rightOperandOffset),
				shader(shader)
			{}
		};
	}
}