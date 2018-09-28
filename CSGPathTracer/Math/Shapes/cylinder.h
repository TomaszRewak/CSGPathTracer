#pragma once

#include "../geometry.h"

#include <curand_kernel.h>

namespace Math
{
	struct Cylinder
	{
		__device__ static Intersection* intersect(Intersection* intersections, const Ray& ray, float maxDistance)
		{
			float
				a = ray.direction.dx * ray.direction.dx + ray.direction.dz * ray.direction.dz,
				b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.z * ray.direction.dz),
				c = ray.begin.x * ray.begin.x + ray.begin.z * ray.begin.z - 1.f;

			float delta = b * b - 4 * a * c;

			if (delta >= 0.0f)
			{
				float deltaSqrt = sqrtf(delta);

				float d1;
				float d2;

				if (a > 0)
				{
					d1 = (-b - deltaSqrt) / (2 * a);
					d2 = (-b + deltaSqrt) / (2 * a);
				}
				else
				{
					d1 = (-b + deltaSqrt) / (2 * a);
					d2 = (-b - deltaSqrt) / (2 * a);
				}

				if (0 < d1 && d1 < maxDistance)
				{
					Point point = ray.begin + ray.direction * d1;
					*intersections = Intersection(point, point - Point(0, point.y, 0), d1);
					intersections++;
				}

				if (0 < d2 && d2 < maxDistance)
				{
					Point point = ray.begin + ray.direction * d2;
					*intersections = Intersection(point, point - Point(0, point.y, 0), d2);
					intersections++;
				}
			}

			return intersections;
		}

		__device__ static bool validateIntersection(const Point& point)
		{
			return point.x * point.x + point.z * point.z <= 1.0f;
		}

		__device__ static Ray randomSurfaceRay(curandState& curandState)
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