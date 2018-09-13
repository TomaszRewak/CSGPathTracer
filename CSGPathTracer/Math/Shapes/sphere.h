#pragma once

#include "..\geometry.h"

#include <curand_kernel.h>

namespace Math
{
	namespace Sphere
	{
		__device__ Intersection* intersect(Intersection* intersections, const Ray& ray, float maxDistance)
		{
			float
				a = ray.direction.dx * ray.direction.dx + ray.direction.dy * ray.direction.dy + ray.direction.dz * ray.direction.dz,
				b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.y * ray.direction.dy + ray.begin.z * ray.direction.dz),
				c = ray.begin.x * ray.begin.x + ray.begin.y * ray.begin.y + ray.begin.z * ray.begin.z - 1.f;

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
					*intersections = Intersection(point, point - Point(0, 0, 0), d1);
					intersections++;
				}

				if (0 < d2 && d2 < maxDistance)
				{
					Point point = ray.begin + ray.direction * d2;
					*intersections = Intersection(point, point - Point(0, 0, 0), d2);
					intersections++;
				}
			}

			return intersections;
		}

		__device__ bool validateIntersection(const Point& point)
		{
			return point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
		}

		__device__ Ray randomSurfaceRay(curandState& curandState)
		{
			float t = 2 * 3.1415 * curand_uniform(&curandState);
			float p = acosf(1 - 2 * curand_uniform(&curandState));

			float x = sinf(p) * cosf(t);
			float y = sinf(p) * sinf(t);
			float z = cosf(p);

			return Ray(
				Point(x, y, z),
				Vector(x, y, z)
			);
		}
	}
}