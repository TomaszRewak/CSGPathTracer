#pragma once

#include "../Scene/component.h"

namespace PathTracer
{
	namespace Operations
	{
		namespace Helpers
		{
			template<bool LeftOutsideRoot, bool RightOutsideRoot, bool LeftBreaksOn, bool LeftReverse, bool RightReverse>
			__device__ void validateTwoOperandOperation(const Math::Point& point, ComponentIterator<Component>& iterator)
			{
				if (iterator.currentComponent->leftOperand == iterator.rootComponent)
				{
					if (iterator.currentComponent->leftOperand == iterator.previousComponent && !iterator.stackedResult)
						return iterator.goUpRoot<false>(false);
					if (iterator.currentComponent->rightOperand == iterator.previousComponent)
						return iterator.goUpRoot<RightOutsideRoot>();
					return iterator.goRight();
				}
				else if (iterator.currentComponent->rightOperand == iterator.rootComponent)
				{
					if (iterator.currentComponent->rightOperand == iterator.previousComponent && !iterator.stackedResult)
						return iterator.goUpRoot<false>(false);
					if (iterator.currentComponent->leftOperand == iterator.previousComponent)
						return iterator.goUpRoot<LeftOutsideRoot>();
					return iterator.goLeft();
				}
				else
				{
					if (iterator.previousComponent == iterator.currentComponent->parent)
						return iterator.goLeft();
					if (iterator.previousComponent == iterator.currentComponent->rightOperand)
						return iterator.goUp<RightReverse>();
					if (iterator.stackedResult == LeftBreaksOn)
						return iterator.goUp<LeftReverse>();
					return iterator.goRight();
				}
			}
		}
	}
}