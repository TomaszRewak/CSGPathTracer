#pragma once

#include "operation-helpers.h"

namespace PathTracer
{
	namespace Operations
	{
		namespace Shape
		{
			__device__ bool validate(const Math::Point& point, const Component*& rootComponent, const Component*& previousComponent, const Component*& currentComponent, bool stackedResult)
			{
				bool result = currentComponent->configuration.localShapeValidationFunction(
					currentComponent->globalTransformation.inverse(point)
				);

				return Helpers::goUp<false>(currentComponent, previousComponent, result);
			}
		}
	}
}