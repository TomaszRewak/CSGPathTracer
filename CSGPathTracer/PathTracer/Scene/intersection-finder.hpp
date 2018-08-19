#pragma once

#include <curand.h>

#include "intersection.hpp"
#include "../../Math/shapes.hpp"

namespace PathTracer
{
	namespace Scene
	{
		class IntersectionFinder
		{
		public:
			__device__ static void intersect(const Component* component, const Math::Ray& ray, Intersection& closestIntersection)
			{
				for (size_t intersectionNumber = 0; ; intersectionNumber++)
				{
					Math::Intersection intersection;

					if (component->type == Common::ComponentType::Sphere)
						intersection = Math::Sphere::intersect(ray, component->globalTransformation, closestIntersection.distance, intersectionNumber);
					else if (component->type == Common::ComponentType::Cylinder)
						intersection = Math::Cylinder::intersect(ray, component->globalTransformation, closestIntersection.distance, intersectionNumber);
					else if (component->type == Common::ComponentType::Plane)
						intersection = Math::Plane::intersect(ray, component->globalTransformation, closestIntersection.distance, intersectionNumber);

					if (intersection.distance == INFINITY)
						break;

					if (validateDown(component, intersection.position))
						closestIntersection = Intersection(intersection, component);
				}
			}

		private:
			__device__ static bool validateUp(const Component* requestingComponent, Component* currentComponent, const Math::Point& position)
			{
				const Component* previousComponent = requestingComponent;

				bool stackedResult;

				while (currentComponent != requestingComponent)
				{
					if (currentComponent->type == Common::ComponentType::Sphere)
					{
						stackedResult = Math::Sphere::validateIntersection(position, currentComponent->globalTransformation);
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else if (currentComponent->type == Common::ComponentType::Cylinder)
					{
						stackedResult = Math::Cylinder::validateIntersection(position, currentComponent->globalTransformation);
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else if (currentComponent->type == Common::ComponentType::Plane)
					{
						stackedResult = Math::Plane::validateIntersection(position, currentComponent->globalTransformation);
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else
					{
						if (previousComponent == currentComponent->parent)
						{
							previousComponent = currentComponent;
							currentComponent = currentComponent->leftOperand;
						}
						else if (previousComponent == currentComponent->leftOperand)
						{
							if (
								currentComponent->type == Common::ComponentType::Union && stackedResult ||
								currentComponent->type == Common::ComponentType::Difference && !stackedResult
								)
							{
								previousComponent = currentComponent;
								currentComponent = currentComponent->parent;
							}
							else
							{
								previousComponent = currentComponent;
								currentComponent = currentComponent->rightOperand;
							}
						}
						else
						{
							if (currentComponent->type == Common::ComponentType::Difference)
								stackedResult = !stackedResult;

							previousComponent = currentComponent;
							currentComponent = currentComponent->parent;
						}
					}
				}

				return stackedResult;
			}

			__device__ static bool validateDown(const Component* component, const Math::Point& position)
			{
				const Component* previousComponent = component;
				const Component* currentComponent = component->parent;

				while (currentComponent)
				{
					if (currentComponent->type == Common::ComponentType::Union)
					{
						if (currentComponent->leftOperand == previousComponent)
						{
							if (validateUp(currentComponent, currentComponent->rightOperand, position))
								return false;
						}
						else
						{
							if (validateUp(currentComponent, currentComponent->leftOperand, position))
								return false;
						}
					}
					else if (currentComponent->type == Common::ComponentType::Difference)
					{
						if (currentComponent->leftOperand == previousComponent)
						{
							if (validateUp(currentComponent, currentComponent->rightOperand, position))
								return false;
						}
						else
						{
							if (!validateUp(currentComponent, currentComponent->leftOperand, position))
								return false;
						}
					}

					previousComponent = currentComponent;
					currentComponent = currentComponent->parent;
				}

				return true;
			}
		};
	}
}