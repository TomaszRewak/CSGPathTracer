#pragma once

#include "../PathTracer/operations.h"
#include "component.h"

namespace Communication
{
	namespace Configurations
	{
		__device__
			PathTracer::ComponentConfiguration<PathTracer::Component> sphereConfiguration()
		{
			return PathTracer::ComponentConfiguration<PathTracer::Component>(
				Math::Sphere::intersect,
				Math::Sphere::validateIntersection,
				PathTracer::Operations::Shape::validate,
				Math::Sphere::randomSurfaceRay,
				0.f, 0.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration<PathTracer::Component> cylinderConfiguration()
		{
			return PathTracer::ComponentConfiguration<PathTracer::Component>(
				Math::Cylinder::intersect,
				Math::Cylinder::validateIntersection,
				PathTracer::Operations::Shape::validate,
				Math::Cylinder::randomSurfaceRay,
				0.f, 0.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration<PathTracer::Component> planeConfiguration()
		{
			return PathTracer::ComponentConfiguration<PathTracer::Component>(
				Math::Plane::intersect,
				Math::Plane::validateIntersection,
				PathTracer::Operations::Shape::validate,
				Math::Plane::randomSurfaceRay,
				0.f, 0.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration<PathTracer::Component> unionConfiguration()
		{
			return PathTracer::ComponentConfiguration<PathTracer::Component>(
				NULL,
				NULL,
				PathTracer::Operations::Union::validate,
				NULL,
				1.f, 1.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration<PathTracer::Component> differenceConfiguration()
		{
			return PathTracer::ComponentConfiguration<PathTracer::Component>(
				NULL,
				NULL,
				PathTracer::Operations::Difference::validate,
				NULL,
				1.f, -1.f
			);
		}

		__device__
			PathTracer::ComponentConfiguration<PathTracer::Component> intersectionConfiguration()
		{
			return PathTracer::ComponentConfiguration<PathTracer::Component>(
				NULL,
				NULL,
				PathTracer::Operations::Intersection::validate,
				NULL,
				-1.f, -1.f
			);
		}
	}
}