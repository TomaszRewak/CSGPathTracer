#pragma once

#include "../geometry.h"

#include <curand_kernel.h>

namespace Math
{
	struct Sphere
	{
		__device__ static float intersect(const Ray& ray, float minDistance)
		{
			float
				a = ray.direction.dx * ray.direction.dx + ray.direction.dy * ray.direction.dy + ray.direction.dz * ray.direction.dz,
				b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.y * ray.direction.dy + ray.begin.z * ray.direction.dz),
				c = ray.begin.x * ray.begin.x + ray.begin.y * ray.begin.y + ray.begin.z * ray.begin.z - 1.f;

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
			return point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
		}

		__device__ static Vector getNormalVector(const Point& point)
		{
			return Vector(point.x, point.y, point.z);
		}

		__device__ static Ray generateRandomSurfaceRay(curandState& curandState)
		{
			float t = 2.f * 3.1415f * curand_uniform(&curandState);
			float p = acosf(1.f - 2.f * curand_uniform(&curandState));

			float x = sinf(p) * cosf(t);
			float y = sinf(p) * sinf(t);
			float z = cosf(p);

			return Ray(
				Point(x, y, z),
				Vector(x, y, z)
			);
		}
	};
}