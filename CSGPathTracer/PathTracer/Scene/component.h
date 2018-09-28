#pragma once

#include "component-configuration.h"

namespace PathTracer
{
	struct Component
	{
		ComponentConfiguration<Component> configuration;

		Math::TwoWayAffineTransformation globalTransformation;

		Shading::Shader shader;

		Component* parent;
		Component* leftOperand;
		Component* rightOperand;

		float normalDirection;

		__device__ Component() :
			parent(NULL), leftOperand(NULL), rightOperand(NULL),
			normalDirection(1.f)
		{ }

		__device__ ComponentIntersection intersect(Math::Ray ray, float maxDistance)
		{
			const int maxIntersections = 2;

			ray = globalTransformation.inverse(ray);

			Math::Intersection intersections[maxIntersections];
			Math::Intersection* intersectionsEnd = configuration.localIntersectionFunction(
				intersections,
				ray,
				maxDistance
			);

			for (Math::Intersection* intersection = intersections; intersection < intersectionsEnd; intersection++)
			{
				Math::Intersection globalIntersection = globalTransformation.transform(*intersection);

				if (validateIntersection(globalIntersection.position))
				{
					globalIntersection.normalVector = globalIntersection.normalVector * normalDirection;
					return ComponentIntersection(globalIntersection, this);
				}
			}

			return ComponentIntersection();
		}

		__device__ bool validateIntersection(const Math::Point& point)
		{
			ComponentIterator<Component> iterator(this->parent, this);

			while (iterator.currentComponent)
			{
				iterator.currentComponent->configuration.localValidationFunction(point, iterator);
			}

			return iterator.stackedResult;
		}

		__device__ Math::Ray generateRay(curandState& state)
		{
			return globalTransformation.transform(
				this->configuration.localRandomSurfaceRayFunction(state)
			);
		}
	};
}