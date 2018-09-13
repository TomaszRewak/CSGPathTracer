#pragma once

#include "../geometry.h"

#include <curand_kernel.h>

namespace Math
{
	namespace Plane
	{
		__device__ Intersection* intersect(Intersection* intersections, const Ray& ray, float maxDistance)
		{
			float d = -ray.begin.y / ray.direction.dy;

			if (0 < d && d < maxDistance)
			{
				Point point = ray.begin + ray.direction * d;
				*intersections = Intersection(point, Vector(0., 1., 0.), d);
				intersections++;
			}

			return intersections;
		}

		__device__ bool validateIntersection(const Point& point)
		{
			return point.y <= 0.0f;
		}

		__device__ Ray randomSurfaceRay(curandState& randomNumberGenerator)
		{
			float x = 1 - 2 * curand_uniform(&randomNumberGenerator);
			float y = 0;
			float z = 1 - 2 * curand_uniform(&randomNumberGenerator);


			return Ray(
				Point(x, y, z),
				Vector(0, 1, 0)
			);
		}
	}
}