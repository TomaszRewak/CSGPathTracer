#pragma once

#include <curand.h>

#include "../../Math/shapes.h"
#include "intersection.h"
#include "../Shading/shader.h"

namespace PathTracer 
{
	struct ComponentConfiguration
	{
		using LocalIntersectionFunction = Math::Intersection*(*)(Math::Intersection*, const Math::Ray& ray, float maxDistance);
		using LocalShapeValidationFunction = bool(*)(const Math::Point& point);
		using LocalValidationFunction = bool(*)(const Math::Point& point, const Component*& rootComponent, const Component*& previousComponent, const Component*& currentComponent, bool stackedResult);
		using LocalRandomSurfaceRayFunction = Math::Ray(*)(curandState&);

		LocalIntersectionFunction localIntersectionFunction;
		LocalShapeValidationFunction localShapeValidationFunction;
		LocalValidationFunction localValidationFunction;
		LocalRandomSurfaceRayFunction localRandomSurfaceRayFunction;

		float leftChildNormalDirection;
		float rightChildNormalDirection;

		__device__ ComponentConfiguration(
			LocalIntersectionFunction localIntersectionFunction,
			LocalShapeValidationFunction localShapeValidationFunction,
			LocalValidationFunction localValidationFunction,
			LocalRandomSurfaceRayFunction localRandomSurfaceRayFunction,
			float leftChildNormalDirection,
			float rightChildNormalDirection) :
			localIntersectionFunction(localIntersectionFunction),
			localShapeValidationFunction(localShapeValidationFunction),
			localValidationFunction(localValidationFunction),
			localRandomSurfaceRayFunction(localRandomSurfaceRayFunction),
			leftChildNormalDirection(leftChildNormalDirection),
			rightChildNormalDirection(rightChildNormalDirection)
		{ }

		__device__ ComponentConfiguration() :
			ComponentConfiguration(NULL, NULL, NULL, NULL, 0.f, 0.f)
		{ }
	};
}