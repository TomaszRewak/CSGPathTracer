#pragma once

#include "../Scene/component.hpp"

namespace PathTracer
{
	namespace Operations
	{
		namespace Helpers
		{
			const bool NONE = false;

			__device__ inline bool goLeft(const Component*& currentComponent, const Component*& previousComponent)
			{
				previousComponent = currentComponent;
				currentComponent = currentComponent->leftOperand;

				return NONE;
			}

			__device__ inline bool goRight(const Component*& currentComponent, const Component*& previousComponent)
			{
				previousComponent = currentComponent;
				currentComponent = currentComponent->rightOperand;

				return NONE;
			}

			template<bool Reverse>
			__device__ inline bool goUp(const Component*& currentComponent, const Component*& previousComponent, bool stackedResult)
			{
				previousComponent = currentComponent;
				currentComponent = currentComponent->parent;

				return stackedResult != Reverse;
			}

			template<bool Reverse>
			__device__ inline bool goUpRoot(const Component*& currentComponent, const Component*& previousComponent, const Component*& rootComponent, bool stackedResult)
			{
				rootComponent = currentComponent;
				previousComponent = currentComponent;
				currentComponent = currentComponent->parent;

				return stackedResult != Reverse;
			}

			template<bool LeftOutsideRoot, bool RightOutsideRoot, bool LeftBreaksOn, bool LeftReverse, bool RightReverse>
			__device__ bool validateTwoOperandOperation(const Math::Point& point, const Component*& rootComponent, const Component*& previousComponent, const Component*& currentComponent, bool stackedResult)
			{
				if (currentComponent->leftOperand == rootComponent)
				{
					if (currentComponent->leftOperand == previousComponent && !stackedResult)
						return goUpRoot<false>(currentComponent, previousComponent, rootComponent, false);
					if (currentComponent->rightOperand == previousComponent)
						return goUpRoot<RightOutsideRoot>(currentComponent, previousComponent, rootComponent, stackedResult);
					return goRight(currentComponent, previousComponent);
				}
				else if (currentComponent->rightOperand == rootComponent)
				{
					if (currentComponent->rightOperand == previousComponent && !stackedResult)
						return goUpRoot<false>(currentComponent, previousComponent, rootComponent, false);
					if (currentComponent->leftOperand == previousComponent)
						return goUpRoot<LeftOutsideRoot>(currentComponent, previousComponent, rootComponent, stackedResult);
					return goLeft(currentComponent, previousComponent);
				}
				else
				{
					if (previousComponent == currentComponent->parent)
						return goLeft(currentComponent, previousComponent);
					if (previousComponent == currentComponent->rightOperand)
						return goUp<RightReverse>(currentComponent, previousComponent, stackedResult);
					if (stackedResult == LeftBreaksOn)
						return goUp<LeftReverse>(currentComponent, previousComponent, stackedResult);
					return goRight(currentComponent, previousComponent);
				}
			}
		}
	}
}