#pragma once

#include "operation-helpers.h"

namespace PathTracer
{
	namespace Operations
	{
		namespace Shape
		{
			__device__ void validate(const Math::Point& point, ComponentIterator<Component>& iterator)
			{
				bool result = iterator.currentComponent->configuration.localShapeValidationFunction(
					iterator.currentComponent->globalTransformation.inverse(point)
				);

				return iterator.goUp(result);
			}
		}
	}
}