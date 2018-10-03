#pragma once

#include "../geometry.h"

#include <curand_kernel.h>

namespace Math
{
	struct Cylinder
	{
		__device__ static float intersect(const Ray& ray, float minDistance)
		{
			float
				a = ray.direction.dx * ray.direction.dx + ray.direction.dz * ray.direction.dz,
				b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.z * ray.direction.dz),
				c = ray.begin.x * ray.begin.x + ray.begin.z * ray.begin.z - 1.f;

			float delta = b * b - 4 * a * c;

			if (delta >= 0.0f)
			{
				float deltaSqrt = sqrtf(delta);

				if (a < 0)
					deltaSqrt = -deltaSqrt;

				float d1 = (-b - deltaSqrt) / (2 * a);
				float d2 = (-b + deltaSqrt) / (2 * a);

				if (d1 > minDistance)
					return d1;
				if (d2 > minDistance)
					return d2;
			}

			return INFINITY;
		}

		__device__ static bool pointInside(const Point& point)
		{
			return point.x * point.x + point.z * point.z <= 1.0f;
		}

		__device__ static Vector getNormalVector(const Point& point)
		{
			return Vector(point.x, 0.f, point.z);
		}

		__device__ static Ray generateRandomSurfaceRay(curandState& curandState)
		{
			float t = 2 * 3.1415 * curand_uniform(&curandState);
			float h = 1 - 2 * curand_uniform(&curandState);

			float x = sinf(t);
			float y = h;
			float z = cosf(t);

			return Ray(
				Point(x, y, z),
				Vector(x, 0, z)
			);
		}
	};
}