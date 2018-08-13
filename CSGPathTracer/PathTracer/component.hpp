#pragma once

#include "shape.hpp"
#include "intersection.hpp"

struct Component
{
	ShapeType type;
	TwoWayAffineTransformation globalTransformation;

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

	__device__ Intersection intersect(const Ray& ray)
	{
		for (size_t intersectionNumber = 0; ; intersectionNumber++)
		{
			Intersection intersection;

			switch (type)
			{
			case ShapeType::Sphere:
				intersection = intersectSphere(ray, intersectionNumber);
				break;
			case ShapeType::Cylinder:
				intersection = intersectCylinder(ray, intersectionNumber);
				break;
			case ShapeType::Plane:
				intersection = intersectPlane(ray, intersectionNumber);
				break;
			default:
				return Intersection();
			}

			if (!intersection.hit)
				break;

			if (validate(intersection.position))
				return intersection;
		}

		return Intersection();
	}

	__device__ bool validate(const Point& position)
	{
		Component* requestingComponent = this;
		Component* previousComponent = this;
		Component* currentComponent = parent;

		bool stackedResult = true;

		while (currentComponent)
		{
			switch (currentComponent->type)
			{
			case ShapeType::Sphere:
			{
				Point point = currentComponent->globalTransformation.inverse(position);
				stackedResult = point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
				previousComponent = currentComponent;
				currentComponent = currentComponent->parent;
				break;
			}
			case ShapeType::Cylinder:
			{
				Point point = currentComponent->globalTransformation.inverse(position);
				stackedResult = point.x * point.x + point.z * point.z <= 1.0f;
				previousComponent = currentComponent;
				currentComponent = currentComponent->parent;
				break;
			}
			case ShapeType::Plane:
			{
				Point point = currentComponent->globalTransformation.inverse(position);
				stackedResult = point.y <= 0.0f;
				previousComponent = currentComponent;
				currentComponent = currentComponent->parent;
				break;
			}
			case ShapeType::Union:
			{
				if (currentComponent->leftOperand == requestingComponent)
				{
					if (currentComponent->rightOperand == previousComponent)
					{
						if (stackedResult) return false;
						requestingComponent = currentComponent;
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else
					{
						previousComponent = currentComponent;
						currentComponent = currentComponent->rightOperand;
					}
				}
				else if (currentComponent->rightOperand == requestingComponent)
				{
					if (currentComponent->leftOperand == previousComponent)
					{
						if (stackedResult) return false;
						requestingComponent = currentComponent;
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else
					{
						previousComponent = currentComponent;
						currentComponent = currentComponent->leftOperand;
					}
				}
				else if (previousComponent == currentComponent->parent)
				{
					previousComponent = currentComponent;
					currentComponent = currentComponent->leftOperand;
				}
				else if (previousComponent == currentComponent->leftOperand)
				{
					if (stackedResult)
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
					previousComponent = currentComponent;
					currentComponent = currentComponent->parent;
				}
				break;
			}
			case ShapeType::Difference:
			{
				if (currentComponent->leftOperand == requestingComponent)
				{
					if (currentComponent->rightOperand == previousComponent)
					{
						if (stackedResult) return false;
						requestingComponent = currentComponent;
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else
					{
						previousComponent = currentComponent;
						currentComponent = currentComponent->rightOperand;
					}
				}
				else if (currentComponent->rightOperand == requestingComponent)
				{
					if (currentComponent->leftOperand == previousComponent)
					{
						if (!stackedResult) return false;
						requestingComponent = currentComponent;
						previousComponent = currentComponent;
						currentComponent = currentComponent->parent;
					}
					else
					{
						previousComponent = currentComponent;
						currentComponent = currentComponent->leftOperand;
					}
				}
				else if (previousComponent == currentComponent->parent)
				{
					previousComponent = currentComponent;
					currentComponent = currentComponent->leftOperand;
				}
				else if (previousComponent == currentComponent->leftOperand)
				{
					if (!stackedResult)
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
					stackedResult = !stackedResult;

					previousComponent = currentComponent;
					currentComponent = currentComponent->parent;
				}
				break;
			}
			}
		}

		return true;
	}

protected:
	__device__ Intersection globalIntersection(const Point& position, const Vector& normalVector)
	{
		return Intersection(
			globalTransformation.transform(position),
			globalTransformation.transform(normalVector) * normalDirection,
			this
		);
	}

	__device__ Intersection intersectSphere(const Ray& globalRay, size_t intersectionNumber)
	{
		if (intersectionNumber > 1)
			return Intersection();

		Ray ray = globalTransformation.inverse(globalRay);

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
				Point intersectionPoint = ray.begin + ray.direction * d;
				return globalIntersection(intersectionPoint, intersectionPoint - Point(0, 0, 0));
			}
		}

		return Intersection();
	}

	__device__ Intersection intersectCylinder(const Ray& globalRay, size_t intersectionNumber)
	{
		if (intersectionNumber > 1)
			return Intersection();

		Ray ray = globalTransformation.inverse(globalRay);

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
				Point intersectionPoint = ray.begin + ray.direction * d;
				return globalIntersection(intersectionPoint, intersectionPoint - Point(0, intersectionPoint.y, 0));
			}
		}

		return Intersection();
	}

	__device__ Intersection intersectPlane(const Ray& globalRay, size_t intersectionNumber)
	{
		if (intersectionNumber > 0)
			return Intersection();

		Ray ray = globalTransformation.inverse(globalRay);

		float d = -ray.begin.y / ray.direction.dy;

		if (d > 0)
		{
			Point intersectionPoint = ray.begin + ray.direction * d;
			return globalIntersection(intersectionPoint, Vector(0., 1., 0.));
		}

		return Intersection();
	}
};