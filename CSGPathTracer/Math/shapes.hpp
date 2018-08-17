#pragma once

#include "math.hpp"

#include <curand_kernel.h>

namespace Math
{
	class Sphere
	{
	public:
		__device__ static Intersection intersect(const Math::Ray& globalRay, const TwoWayAffineTransformation& transformation, size_t intersectionNumber)
		{
			if (intersectionNumber > 1)
				return Intersection();

			Math::Ray ray = transformation.inverse(globalRay);

			float
				a = ray.direction.dx * ray.direction.dx + ray.direction.dy * ray.direction.dy + ray.direction.dz * ray.direction.dz,
				b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.y * ray.direction.dy + ray.begin.z * ray.direction.dz),
				c = ray.begin.x * ray.begin.x + ray.begin.y * ray.begin.y + ray.begin.z * ray.begin.z - 1.f;

			float delta = b * b - 4 * a * c;

			if (delta >= 0.0f)
			{
				float deltaSqrt = sqrtf(delta);

				float d1 = (-b - deltaSqrt) / (2 * a);
				float d2 = (-b + deltaSqrt) / (2 * a);

				float d = !intersectionNumber && d1 < d2 && d1 > 0 || intersectionNumber && d1 > d2 && d2 > 0 ? d1 : d2;

				if (d > 0)
				{
					Math::Point intersectionPoint = ray.begin + ray.direction * d;
					return Intersection(ray.begin, intersectionPoint, intersectionPoint - Math::Point(0, 0, 0), transformation);
				}
			}

			return Intersection();
		}

		__device__ static bool validateIntersection(const Point& position, const TwoWayAffineTransformation& transformation)
		{
			Math::Point point = transformation.inverse(position);
			return point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
		}

		__device__ static Ray randomSurfaceRay(const AffineTransformation& transformation, curandState& curandState)
		{
			float t = 2 * 3.1415 * curand_uniform(&curandState);
			float p = acosf(1 - 2 * curand_uniform(&curandState));

			float x = sinf(p) * cosf(t);
			float y = sinf(p) * sinf(t);
			float z = cosf(p);

			return transformation.transform(Ray(
				Point(x, y, z),
				Vector(x, y, z)
			));
		}
	};

	class Cylinder
	{
	public:
		__device__ static Intersection intersect(const Math::Ray& globalRay, const TwoWayAffineTransformation& transformation, size_t intersectionNumber)
		{
			if (intersectionNumber > 1)
				return Intersection();

			Math::Ray ray = transformation.inverse(globalRay);

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

				if (d > 0)
				{
					Math::Point intersectionPoint = ray.begin + ray.direction * d;
					return Intersection(ray.begin, intersectionPoint, intersectionPoint - Math::Point(0, intersectionPoint.y, 0), transformation);
				}
			}

			return Intersection();
		}

		__device__ static bool validateIntersection(const Point& position, const TwoWayAffineTransformation& transformation)
		{
			Math::Point point = transformation.inverse(position);
			return point.x * point.x + point.z * point.z <= 1.0f;
		}

		__device__ static Ray randomSurfaceRay(const AffineTransformation& transformation, curandState& curandState)
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
	};

	class Plane
	{
	public:
		__device__ static Intersection intersect(const Math::Ray& globalRay, const TwoWayAffineTransformation& transformation, size_t intersectionNumber)
		{
			if (intersectionNumber > 0)
				return Intersection();

			Math::Ray ray = transformation.inverse(globalRay);

			float d = -ray.begin.y / ray.direction.dy;

			if (d > 0)
			{
				Math::Point intersectionPoint = ray.begin + ray.direction * d;
				return Intersection(ray.begin, intersectionPoint, Math::Vector(0., 1., 0.), transformation);
			}

			return Intersection();
		}

		__device__ static bool validateIntersection(const Point& position, const TwoWayAffineTransformation& transformation)
		{
			Math::Point point = transformation.inverse(position);
			return point.y <= 0.0f;
		}

		__device__ static Ray randomSurfaceRay(const AffineTransformation& transformation, curandState& randomNumberGenerator)
		{
			float x = 1 - 2 * curand_uniform(&randomNumberGenerator);
			float y = 0;
			float z = 1 - 2 * curand_uniform(&randomNumberGenerator);


			return transformation.transform(Ray(
				Point(x, y, z),
				Vector(0, 1, 0)
			));
		}
	};
}