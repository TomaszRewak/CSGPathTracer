#pragma once

#include "vector.h"

namespace Math
{
	struct Point
	{
		float x;
		float y;
		float z;

		__host__ __device__ Point() :
			x(0), y(0), z(0) {};
		__host__ __device__ Point(float x, float y, float z) :
			x(x), y(y), z(z) {}
		__host__ __device__ Point(float x, float y, float z, float w) :
			x(x / w), y(y / w), z(z / w) {};

		__host__ __device__ Vector operator-(const Point& begin) const
		{
			return Vector(
				x - begin.x,
				y - begin.y,
				z - begin.z
			);
		}

		__host__ __device__ Point operator+(const Vector &v) const
		{
			return Point(
				x + v.dx,
				y + v.dy,
				z + v.dz
			);
		}
	};
}