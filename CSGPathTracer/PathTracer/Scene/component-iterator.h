#pragma once

namespace PathTracer
{
	template<class Component>
	struct ComponentIterator {
		const Component* currentComponent;
		const Component* previousComponent;
		const Component* rootComponent;

		bool stackedResult;

		__device__ ComponentIterator(const Component* currentComponent, const Component* previousComponent) :
			currentComponent(currentComponent),
			previousComponent(previousComponent),
			rootComponent(previousComponent),
			stackedResult(true)
		{ }

		__device__ void goLeft()
		{
			previousComponent = currentComponent;
			currentComponent = currentComponent->leftOperand;
		}

		__device__ void goRight()
		{
			previousComponent = currentComponent;
			currentComponent = currentComponent->rightOperand;
		}

		template<bool Reverse>
		__device__ void goUp()
		{
			previousComponent = currentComponent;
			currentComponent = currentComponent->parent;

			stackedResult = stackedResult != Reverse;
		}

		__device__ void goUp(bool newResult)
		{
			previousComponent = currentComponent;
			currentComponent = currentComponent->parent;

			stackedResult = newResult;
		}

		template<bool Reverse>
		__device__ void goUpRoot()
		{
			rootComponent = currentComponent;
			previousComponent = currentComponent;
			currentComponent = currentComponent->parent;

			stackedResult = stackedResult != Reverse;
		}

		template<bool Reverse>
		__device__ void goUpRoot(bool newResult)
		{
			rootComponent = currentComponent;
			previousComponent = currentComponent;
			currentComponent = currentComponent->parent;

			stackedResult = newResult != Reverse;
		}
	};
}