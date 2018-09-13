#pragma once

#include "operation-helpers.h"

namespace PathTracer
{
	namespace Operations
	{
		namespace Union
		{
			__device__ bool validate(const Math::Point& point, const Component*& rootComponent, const Component*& previousComponent, const Component*& currentComponent, bool stackedResult)
			{
				return Helpers::validateTwoOperandOperation<true, true, true, false, false>(point, rootComponent, previousComponent, currentComponent, stackedResult);
			}
		}

		namespace Difference
		{
			__device__ bool validate(const Math::Point& point, const Component*& rootComponent, const Component*& previousComponent, const Component*& currentComponent, bool stackedResult)
			{
				return Helpers::validateTwoOperandOperation<false, true, false, false, true>(point, rootComponent, previousComponent, currentComponent, stackedResult);
			}
		}

		namespace Intersection
		{
			__device__ bool validate(const Math::Point& point, const Component*& rootComponent, const Component*& previousComponent, const Component*& currentComponent, bool stackedResult)
			{
				return Helpers::validateTwoOperandOperation<false, false, false, false, false>(point, rootComponent, previousComponent, currentComponent, stackedResult);
			}
		}
	}
}