#pragma once

#include "point.h"

namespace Math
{
	struct Ray
	{
		Point begin;
		Vector direction;

		__host__ __device__ Ray()
		{ }

		__host__ __device__ Ray(const Point& begin, const Vector& direction) :
			begin(begin), direction(direction)
		{}
		__host__ __device__ Ray(const Point& begin, const Point& through) :
			begin(begin), direction(through - begin)
		{}

		__host__ __device__ Math::Point point(float length) const
		{
			return begin + direction * length;
		}

		__host__ __device__ Math::Ray unitRay() const
		{
			return Ray(begin, direction.unitVector());
		}
	};
}