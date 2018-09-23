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

		__host__ __device__ Ray(Point& begin, Vector& direction) :
			begin(begin), direction(direction)
		{}
		__host__ __device__ Ray(Point& begin, Point& through) :
			begin(begin), direction(through - begin)
		{}
	};
}