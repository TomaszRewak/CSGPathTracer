#pragma once

#include "../Communication/component.hpp"
#include "intersection.hpp"

namespace PathTracer
{
	namespace Scene
	{
		struct Component
		{
			Common::ComponentType type;
			Math::TwoWayAffineTransformation globalTransformation;

			Component* parent;
			Component* leftOperand;
			Component* rightOperand;

			float normalDirection;

			__device__ Component() :
				parent(NULL),
				leftOperand(NULL),
				rightOperand(NULL),
				normalDirection(1.f)
			{ }

			__device__ void intersect(const Math::Ray& ray, Intersection& closestIntersection)
			{
				for (size_t intersectionNumber = 0; ; intersectionNumber++)
				{
					Intersection intersection;

					switch (type)
					{
					case Common::ComponentType::Sphere:
						intersection = intersectSphere(ray, intersectionNumber);
						break;
					case Common::ComponentType::Cylinder:
						intersection = intersectCylinder(ray, intersectionNumber);
						break;
					case Common::ComponentType::Plane:
						intersection = intersectPlane(ray, intersectionNumber);
						break;
					}

					if (!intersection.hit)
						break;
					
					if ((!closestIntersection.hit || (intersection.position - ray.begin).norm2() < (closestIntersection.position - ray.begin).norm2()) &&
						validateDown(intersection.position))
					{
						closestIntersection = intersection;
					}
				}
			}

			__device__ bool validateUp(Component* requestingComponent, Component* currentComponent, const Math::Point& position)
			{
				Component* previousComponent = requestingComponent;

				bool stackedResult;

				while (currentComponent != requestingComponent)
				{
					if (currentComponent->type == Common::ComponentType::Sphere)
					{
						Math::Point point = currentComponent->globalTransformation.inverse(position);
						stackedResult = point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else if (currentComponent->type == Common::ComponentType::Cylinder)
					{
						Math::Point point = currentComponent->globalTransformation.inverse(position);
						stackedResult = point.x * point.x + point.z * point.z <= 1.0f;
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else if (currentComponent->type == Common::ComponentType::Plane)
					{
						Math::Point point = currentComponent->globalTransformation.inverse(position);
						stackedResult = point.y <= 0.0f;
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

			__device__ bool validateDown(const Math::Point& position)
			{
				Component* previousComponent = this;
				Component* currentComponent = parent;

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

		protected:
			__device__ Intersection globalIntersection(const Math::Point& position, const Math::Vector& normalVector)
			{
				return Intersection(
					globalTransformation.transform(position),
					globalTransformation.transform(normalVector) * normalDirection,
					this
				);
			}

			__device__ Intersection intersectSphere(const Math::Ray& globalRay, size_t intersectionNumber)
			{
				if (intersectionNumber > 1)
					return Intersection();

				Math::Ray ray = globalTransformation.inverse(globalRay);

				float
					a = ray.direction.dx * ray.direction.dx + ray.direction.dy * ray.direction.dy + ray.direction.dz * ray.direction.dz,
					b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.y * ray.direction.dy + ray.begin.z * ray.direction.dz),
					c = ray.begin.x * ray.begin.x + ray.begin.y * ray.begin.y + ray.begin.z * ray.begin.z - 1.f;

				float delta = b * b - 4 * a * c;

				if (delta >= 0.0f)
				{
					float deltaSqrt = sqrtf(delta);

					float d1 = (-b - deltaSqrt) / (2 * a);
					float d2 = (-b + deltaSqrt) / (2 * a);

					float d = !intersectionNumber && d1 < d2 && d1 > 0 || intersectionNumber && d1 > d2 && d2 > 0 ? d1 : d2;

					if (d > 0)
					{
						Math::Point intersectionPoint = ray.begin + ray.direction * d;
						return globalIntersection(intersectionPoint, intersectionPoint - Math::Point(0, 0, 0));
					}
				}

				return Intersection();
			}

			__device__ Intersection intersectCylinder(const Math::Ray& globalRay, size_t intersectionNumber)
			{
				if (intersectionNumber > 1)
					return Intersection();

				Math::Ray ray = globalTransformation.inverse(globalRay);

				float
					a = ray.direction.dx * ray.direction.dx + ray.direction.dz * ray.direction.dz,
					b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.z * ray.direction.dz),
					c = ray.begin.x * ray.begin.x + ray.begin.z * ray.begin.z - 1.f;

				float delta = b * b - 4 * a * c;

				if (delta >= 0.0f)
				{
					float deltaSqrt = sqrtf(delta);

					float d1 = (-b - deltaSqrt) / (2 * a);
					float d2 = (-b + deltaSqrt) / (2 * a);

					float d = !intersectionNumber && d1 < d2 && d1 > 0 || intersectionNumber && d1 > d2 && d2 > 0 ? d1 : d2;

					if (d > 0)
					{
						Math::Point intersectionPoint = ray.begin + ray.direction * d;
						return globalIntersection(intersectionPoint, intersectionPoint - Math::Point(0, intersectionPoint.y, 0));
					}
				}

				return Intersection();
			}

			__device__ Intersection intersectPlane(const Math::Ray& globalRay, size_t intersectionNumber)
			{
				if (intersectionNumber > 0)
					return Intersection();

				Math::Ray ray = globalTransformation.inverse(globalRay);

				float d = -ray.begin.y / ray.direction.dy;

				if (d > 0)
				{
					Math::Point intersectionPoint = ray.begin + ray.direction * d;
					return globalIntersection(intersectionPoint, Math::Vector(0., 1., 0.));
				}

				return Intersection();
			}
		};
	}
}