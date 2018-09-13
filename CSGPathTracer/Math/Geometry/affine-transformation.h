#pragma once

#include "ray.h"
#include "intersection.h"

namespace Math
{
	struct AffineTransformation
	{
	protected:
		float matrix[4][4];

		__host__ __device__ void resetTransformation()
		{
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					matrix[i][j] = i == j;
		}

	public:
		__host__ __device__ AffineTransformation()
		{
			resetTransformation();
		}

		__host__ __device__ AffineTransformation combine(const AffineTransformation& second) const
		{
			return combine(second.matrix);
		}

		__host__ __device__ AffineTransformation rotateX(float angle) const
		{
			return rotateX(cosf(angle), sinf(angle));
		}

		__host__ __device__ AffineTransformation rotateY(float angle) const
		{
			return rotateY(cosf(angle), sinf(angle));
		}

		__host__ __device__ AffineTransformation rotateZ(float angle) const
		{
			return rotateZ(cosf(angle), sinf(angle));
		}

		__host__ __device__ AffineTransformation translate(float x, float y, float z) const
		{
			float translation[4][4] = {
				{ 1, 0, 0, x },
				{ 0, 1, 0, y },
				{ 0, 0, 1, z },
				{ 0, 0, 0, 1 }
			};
			return combine(translation);
		}

		__host__ __device__ AffineTransformation scale(float x, float y, float z) const
		{
			float scale[4][4] = {
				{ x, 0, 0, 0 },
				{ 0, y, 0, 0 },
				{ 0, 0, z, 0 },
				{ 0, 0, 0, 1 }
			};
			return combine(scale);
		}

		__host__ __device__ Point transform(const Point& point) const
		{
			return transformByMatrix(point, matrix);
		}

		__host__ __device__ Vector transform(const Vector& vector) const
		{
			return transform(Point(vector.dx, vector.dy, vector.dz)) - transform(Point());
		}

		__host__ __device__ Ray transform(const Ray& vector) const
		{
			return Ray(
				transform(vector.begin),
				transform(vector.direction)
			);
		}

		__host__ __device__ Intersection transform(const Intersection& intersection) const
		{
			return Intersection(
				transform(intersection.position),
				transform(intersection.normalVector),
				intersection.distance
			);
		}

	protected:
		__host__ __device__ AffineTransformation rotateX(float c, float s) const
		{
			float rotation[4][4] = {
				{ 1, 0, 0, 0 },
				{ 0, c, -s, 0 },
				{ 0, s, c, 0 },
				{ 0, 0, 0, 1 }
			};
			return combine(rotation);
		}

		__host__ __device__ AffineTransformation rotateY(float c, float s) const
		{
			float rotation[4][4] = {
				{ c, 0, s, 0 },
				{ 0, 1, 0, 0 },
				{ -s, 0, c, 0 },
				{ 0, 0, 0, 1 }
			};
			return combine(rotation);
		}

		__host__ __device__ AffineTransformation rotateZ(float c, float s) const
		{
			float rotation[4][4] = {
				{ c, -s, 0, 0 },
				{ s, c, 0, 0 },
				{ 0, 0, 1, 0 },
				{ 0, 0, 0, 1 }
			};
			return combine(rotation);
		}

		__host__ __device__ AffineTransformation combine(const float second[4][4]) const
		{
			AffineTransformation result;

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					result.matrix[i][j] = 0;

					for (int k = 0; k < 4; k++)
					{
						result.matrix[i][j] += second[i][k] * matrix[k][j];
					}
				}
			}

			return result;
		}

		__host__ __device__ Point transformByMatrix(const Point& point, const float matrix[4][4]) const
		{
			return Point(
				point.x * matrix[0][0] + point.y * matrix[0][1] + point.z * matrix[0][2] + /*point._w **/ matrix[0][3],
				point.x * matrix[1][0] + point.y * matrix[1][1] + point.z * matrix[1][2] + /*point._w **/ matrix[1][3],
				point.x * matrix[2][0] + point.y * matrix[2][1] + point.z * matrix[2][2] + /*point._w **/ matrix[2][3],
				point.x * matrix[3][0] + point.y * matrix[3][1] + point.z * matrix[3][2] + /*point._w **/ matrix[3][3]
			);
		}
	};
}