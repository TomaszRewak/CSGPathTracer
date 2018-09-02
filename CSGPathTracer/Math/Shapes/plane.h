#pragma once

#include "..\geometry.h"

#include <curand_kernel.h>

namespace Math
{
	namespace Plane
	{
		__device__ Intersection intersect(const Ray& globalRay, const TwoWayAffineTransformation& transformation, float maxDistance, size_t intersectionNumber)
		{
			if (intersectionNumber > 0)
				return Intersection();

			Ray ray = transformation.inverse(globalRay);

			float d = -ray.begin.y / ray.direction.dy;

			if (0 < d && d < maxDistance)
			{
				Point intersectionPoint = ray.begin + ray.direction * d;
				return Intersection(intersectionPoint, Vector(0., 1., 0.), d, transformation);
			}

			return Intersection();
		}

		__device__ bool validateIntersection(const Point& position, const TwoWayAffineTransformation& transformation)
		{
			Point point = transformation.inverse(position);
			return point.y <= 0.0f;
		}

		__device__ Ray randomSurfaceRay(const AffineTransformation& transformation, curandState& randomNumberGenerator)
		{
			float x = 1 - 2 * curand_uniform(&randomNumberGenerator);
			float y = 0;
			float z = 1 - 2 * curand_uniform(&randomNumberGenerator);


			return transformation.transform(Ray(
				Point(x, y, z),
				Vector(0, 1, 0)
			));
		}
	}
}