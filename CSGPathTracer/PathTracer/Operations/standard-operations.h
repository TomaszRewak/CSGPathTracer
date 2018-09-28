#pragma once

#include "operation-helpers.h"

namespace PathTracer
{
	namespace Operations
	{
		namespace Union
		{
			__device__ void validate(const Math::Point& point, ComponentIterator<Component>& iterator)
			{
				return Helpers::validateTwoOperandOperation<true, true, true, false, false>(point, iterator);
			}
		}

		namespace Difference
		{
			__device__ void validate(const Math::Point& point, ComponentIterator<Component>& iterator)
			{
				return Helpers::validateTwoOperandOperation<false, true, false, false, true>(point, iterator);
			}
		}

		namespace Intersection
		{
			__device__ void validate(const Math::Point& point, ComponentIterator<Component>& iterator)
			{
				return Helpers::validateTwoOperandOperation<false, false, false, false, false>(point, iterator);
			}
		}
	}
}