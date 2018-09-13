#pragma once

#include "component-configuration.h"

namespace PathTracer
{
	struct Component
	{
		ComponentConfiguration configuration;

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

		__device__ Intersection intersect(Math::Ray ray, float maxDistance)
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
					return Intersection(globalIntersection, this);
				}
			}

			return Intersection();
		}

		__device__ bool validateIntersection(const Math::Point& point)
		{
			bool stackedResult = true;

			const Component* rootComponent = this;
			const Component* previousComponent = this;
			const Component* currentComponent = this->parent;

			while (currentComponent)
			{
				stackedResult = currentComponent->configuration.localValidationFunction(
					point,
					rootComponent,
					previousComponent,
					currentComponent,
					stackedResult
				);
			}

			return stackedResult;
		}

		__device__ Math::Ray generateRay(curandState& state)
		{
			return globalTransformation.transform(
				this->configuration.localRandomSurfaceRayFunction(state)
			);
		}
	};
}