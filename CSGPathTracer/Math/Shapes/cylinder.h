#pragma once

#include "..\geometry.h"

#include <curand_kernel.h>

namespace Math
{
	namespace Cylinder
	{
		__device__ Intersection intersect(const Ray& globalRay, const TwoWayAffineTransformation& transformation, float maxDistance, size_t intersectionNumber)
		{
			if (intersectionNumber > 1)
				return Intersection();

			Ray ray = transformation.inverse(globalRay);

			float
				a = ray.direction.dx * ray.direction.dx + ray.direction.dz * ray.direction.dz,
				b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.z * ray.direction.dz),
				c = ray.begin.x * ray.begin.x + ray.begin.z * ray.begin.z - 1.f;

			float delta = b * b - 4 * a * c;

			if (delta >= 0.0f)
			{
				float deltaSqrt = sqrtf(delta);

				float d1 = (-b - deltaSqrt) / (2 * a);
				float d2 = (-b + deltaSqrt) / (2 * a);

				float d = !intersectionNumber && d1 < d2 && d1 > 0 || intersectionNumber && d1 > d2 && d2 > 0 ? d1 : d2;

				if (0 < d && d < maxDistance)
				{
					Point intersectionPoint = ray.begin + ray.direction * d;
					return Intersection(intersectionPoint, intersectionPoint - Point(0, intersectionPoint.y, 0), d, transformation);
				}
			}

			return Intersection();
		}

		__device__ bool validateIntersection(const Point& position, const TwoWayAffineTransformation& transformation)
		{
			Point point = transformation.inverse(position);
			return point.x * point.x + point.z * point.z <= 1.0f;
		}

		__device__ Ray randomSurfaceRay(const AffineTransformation& transformation, curandState& curandState)
		{
			float t = 2 * 3.1415 * curand_uniform(&curandState);
			float h = 1 - 2 * curand_uniform(&curandState);

			float x = sinf(t);
			float y = h;
			float z = cosf(t);

			return transformation.transform(Ray(
				Point(x, y, z),
				Vector(x, 0, z)
			));
		}
	}
}