#pragma once

#include "../../Math/geometry.h"

namespace PathTracer
{
	namespace Shading
	{
		struct Color
		{
			float b;
			float g;
			float r;
			float a;

			__host__ __device__ Color() :
				Color(0, 0, 0, 1.f)
			{}
			__host__ __device__ Color(float r, float g, float b) :
				b(b), g(g), r(r), a(1.f) {}
			__host__ __device__ Color(float r, float g, float b, float a) :
				b(b), g(g), r(r), a(a) {}
			__host__ __device__ Color(float g) :
				b(g), g(g), r(g), a(1.f)
			{ }

			__device__ Color operator+(const Color& second) const
			{
				return Color(
					fminf(1., r + second.r),
					fminf(1., g + second.g),
					fminf(1., b + second.b)
				);
			}

			__device__ Color operator*(float by) const
			{
				return Color(r * by, g * by, b * by, a);
			}

			__device__ Color operator*(const Color& second) const
			{
				return Color(r * second.r, g * second.g, b * second.b, a * second.a);
			}
		};
	}
}