#pragma once

#include "../../Math/math.hpp"

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
				return Color(r * by, g * by, b * by);
			}
		};

		struct Filter
		{
			float r;
			float g;
			float b;

			__device__ Filter(float r, float g, float b) :
				r(fminf(1., r)), g(fminf(1., g)), b(fminf(1., b))
			{ }
			__device__ Filter() : Filter(1, 1, 1)
			{ }
			__device__ Filter(const Color& color) :
				Filter(color.r, color.g, color.b)
			{ }

			__device__ Filter operator*(float by) const
			{
				return Filter(r * by, g * by, b * by);
			}

			__device__ Filter operator+(const Filter& second) const
			{
				return Filter(r + second.r, g + second.g, b + second.b);
			}

			__device__ Filter operator*(const Filter& second) const
			{
				return Filter(r * second.r, g * second.g, b * second.b);
			}

			__device__ Color operator*(const Color& second) const
			{
				return Color(r * second.r, g * second.g, b * second.b);
			}
		};
	}
}