#pragma once

#include "../PathTracer/operations.h"
#include "component.h"

namespace Communication
{
	namespace Configurations
	{
		__device__
			PathTracer::ComponentConfiguration sphereConfiguration()
		{
			return PathTracer::ComponentConfiguration(
				Math::Sphere::intersect,
				Math::Sphere::validateIntersection,
				PathTracer::Operations::Shape::validate,
				Math::Sphere::randomSurfaceRay,
				0.f, 0.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration cylinderConfiguration()
		{
			return PathTracer::ComponentConfiguration(
				Math::Cylinder::intersect,
				Math::Cylinder::validateIntersection,
				PathTracer::Operations::Shape::validate,
				Math::Cylinder::randomSurfaceRay,
				0.f, 0.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration planeConfiguration()
		{
			return PathTracer::ComponentConfiguration(
				Math::Plane::intersect,
				Math::Plane::validateIntersection,
				PathTracer::Operations::Shape::validate,
				Math::Plane::randomSurfaceRay,
				0.f, 0.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration unionConfiguration()
		{
			return PathTracer::ComponentConfiguration(
				NULL,
				NULL,
				PathTracer::Operations::Union::validate,
				NULL,
				1.f, 1.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration differenceConfiguration()
		{
			return PathTracer::ComponentConfiguration(
				NULL,
				NULL,
				PathTracer::Operations::Difference::validate,
				NULL,
				1.f, -1.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration intersectionConfiguration()
		{
			return PathTracer::ComponentConfiguration(
				NULL,
				NULL,
				PathTracer::Operations::Intersection::validate,
				NULL,
				-1.f, -1.f
			);
		}
	}
}