#pragma once

#include "affine-transformation.h"

namespace Math
{
	struct Intersection
	{
		Math::Point position;
		Math::Vector normalVector;

		float distance;

		__device__ Intersection() :
			distance(INFINITY)
		{ }

		__device__ Intersection(const Math::Point& position, const Math::Vector& normalVector, float distance) :
			position(position),
			normalVector(normalVector),
			distance(distance)
		{ }

		__device__ Intersection(const Math::Point& position, const Math::Vector& normalVector, float distance, const AffineTransformation& transformation) :
			Intersection(
				transformation.transform(position),
				transformation.transform(normalVector),
				distance
			)
		{ }
	};
}