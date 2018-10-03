#pragma once

#include "../geometry.h"

#include <curand_kernel.h>

namespace Math
{
	struct Plane
	{
		__device__ static float intersect(const Ray& ray, float minDistance)
		{
			float d = -ray.begin.y / ray.direction.dy;

			if (d > minDistance)
				return d;
			else
				return INFINITY;
		}

		__device__ static bool pointInside(const Point& point)
		{
			return point.y <= 0.0f;
		}

		__device__ static Vector getNormalVector(const Point& point)
		{
			return Vector(0.f, 1.f, 0.f);
		}

		__device__ static Ray generateRandomSurfaceRay(curandState& randomNumberGenerator)
		{
			float x = 1 - 2 * curand_uniform(&randomNumberGenerator);
			float y = 0;
			float z = 1 - 2 * curand_uniform(&randomNumberGenerator);

			return Ray(
				Point(x, y, z),
				Vector(0, 1, 0)
			);
		}
	};
}