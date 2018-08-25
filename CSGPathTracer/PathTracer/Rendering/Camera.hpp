#pragma once

#include "../../Math/math.hpp"

#include <curand_kernel.h>

namespace PathTracer
{
	namespace Rendering
	{
		struct Camera
		{
			Math::AffineTransformation transformation;
			float depth;

			__host__ Camera() :
				transformation(Math::AffineTransformation()),
				depth(1.f)
			{ }

			__host__ Camera(Math::AffineTransformation transformation, float depth) :
				transformation(transformation),
				depth(depth)
			{ }

			__device__ Math::Ray getRay(float x, float y)
			{
				return transformation.transform(
					Math::Ray(
						Math::Point(0, 0, 0),
						Math::Point(x, y, 1)
					)
				);
			}

			__device__ Math::Ray getRay(size_t x, size_t y, size_t width, size_t height, curandState& randState)
			{
				float boxSize = fminf(width, height);

				return getRay(
					(x - width * 0.5 + curand_uniform(&randState) - 0.5) / boxSize,
					(y - height * 0.5 + curand_uniform(&randState) - 0.5) / boxSize
				);
			}
		};
	}
}