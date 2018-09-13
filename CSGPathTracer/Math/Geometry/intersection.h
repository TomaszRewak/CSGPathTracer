#pragma once

#include "point.h"

namespace Math
{
	struct Intersection
	{
		Math::Point position;
		Math::Vector normalVector;

		float distance;

		__device__ __host__ Intersection() :
			distance(INFINITY)
		{ }

		__device__ __host__ explicit Intersection(float maxDistance) :
			distance(maxDistance)
		{ }

		__device__ __host__ Intersection(const Math::Point& position, const Math::Vector& normalVector, float distance) :
			position(position),
			normalVector(normalVector),
			distance(distance)
		{ }
	};
}