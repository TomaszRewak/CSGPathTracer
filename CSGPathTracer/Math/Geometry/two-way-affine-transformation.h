#pragma once

#include "affine-transformation.h"

namespace Math
{
	struct TwoWayAffineTransformation : public AffineTransformation
	{
	protected:
		float inversedMatrix[4][4];

		__device__ void resetTransformation()
		{
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					matrix[i][j] = inversedMatrix[i][j] = i == j;
		}

	public:
		__device__ TwoWayAffineTransformation()
		{
			resetTransformation();
		}

		__device__ explicit TwoWayAffineTransformation(const AffineTransformation& affineTransformation) :
			AffineTransformation(affineTransformation)
		{
			inverse();
		}

		__device__ TwoWayAffineTransformation rotateX(float angle) const
		{
			return TwoWayAffineTransformation(AffineTransformation::rotateX(angle));
		}

		__device__ TwoWayAffineTransformation rotateY(float angle) const
		{
			return TwoWayAffineTransformation(AffineTransformation::rotateY(angle));
		}

		__device__ TwoWayAffineTransformation rotateZ(float angle) const
		{
			return TwoWayAffineTransformation(AffineTransformation::rotateZ(angle));
		}

		__device__ TwoWayAffineTransformation translate(float x, float y, float z) const
		{
			return TwoWayAffineTransformation(AffineTransformation::translate(x, y, z));
		}

		__device__ TwoWayAffineTransformation scale(float x, float y, float z) const
		{
			return TwoWayAffineTransformation(AffineTransformation::scale(x, y, z));
		}

		__device__ Point inverse(const Point& point) const
		{
			return transformByMatrix(point, inversedMatrix);
		}

		__device__ Vector inverse(const Vector& vector) const
		{
			return inverse(Point(vector.dx, vector.dy, vector.dz)) - inverse(Point());
		}

		__device__ Ray inverse(const Ray& vector) const
		{
			return Ray(
				inverse(vector.begin),
				inverse(vector.direction)
			);
		}

		__device__ Intersection inverse(const Intersection& intersection)
		{
			return Intersection(
				inverse(intersection.position),
				inverse(intersection.normalVector),
				intersection.distance
			);
		}

	protected:
		__device__ void inverse()
		{
			float *m = (float *)matrix;
			float *im = (float *)inversedMatrix;

			float inv[16], det;
			int i;

			inv[0] = m[5] * m[10] * m[15] -
				m[5] * m[11] * m[14] -
				m[9] * m[6] * m[15] +
				m[9] * m[7] * m[14] +
				m[13] * m[6] * m[11] -
				m[13] * m[7] * m[10];

			inv[4] = -m[4] * m[10] * m[15] +
				m[4] * m[11] * m[14] +
				m[8] * m[6] * m[15] -
				m[8] * m[7] * m[14] -
				m[12] * m[6] * m[11] +
				m[12] * m[7] * m[10];

			inv[8] = m[4] * m[9] * m[15] -
				m[4] * m[11] * m[13] -
				m[8] * m[5] * m[15] +
				m[8] * m[7] * m[13] +
				m[12] * m[5] * m[11] -
				m[12] * m[7] * m[9];

			inv[12] = -m[4] * m[9] * m[14] +
				m[4] * m[10] * m[13] +
				m[8] * m[5] * m[14] -
				m[8] * m[6] * m[13] -
				m[12] * m[5] * m[10] +
				m[12] * m[6] * m[9];

			inv[1] = -m[1] * m[10] * m[15] +
				m[1] * m[11] * m[14] +
				m[9] * m[2] * m[15] -
				m[9] * m[3] * m[14] -
				m[13] * m[2] * m[11] +
				m[13] * m[3] * m[10];

			inv[5] = m[0] * m[10] * m[15] -
				m[0] * m[11] * m[14] -
				m[8] * m[2] * m[15] +
				m[8] * m[3] * m[14] +
				m[12] * m[2] * m[11] -
				m[12] * m[3] * m[10];

			inv[9] = -m[0] * m[9] * m[15] +
				m[0] * m[11] * m[13] +
				m[8] * m[1] * m[15] -
				m[8] * m[3] * m[13] -
				m[12] * m[1] * m[11] +
				m[12] * m[3] * m[9];

			inv[13] = m[0] * m[9] * m[14] -
				m[0] * m[10] * m[13] -
				m[8] * m[1] * m[14] +
				m[8] * m[2] * m[13] +
				m[12] * m[1] * m[10] -
				m[12] * m[2] * m[9];

			inv[2] = m[1] * m[6] * m[15] -
				m[1] * m[7] * m[14] -
				m[5] * m[2] * m[15] +
				m[5] * m[3] * m[14] +
				m[13] * m[2] * m[7] -
				m[13] * m[3] * m[6];

			inv[6] = -m[0] * m[6] * m[15] +
				m[0] * m[7] * m[14] +
				m[4] * m[2] * m[15] -
				m[4] * m[3] * m[14] -
				m[12] * m[2] * m[7] +
				m[12] * m[3] * m[6];

			inv[10] = m[0] * m[5] * m[15] -
				m[0] * m[7] * m[13] -
				m[4] * m[1] * m[15] +
				m[4] * m[3] * m[13] +
				m[12] * m[1] * m[7] -
				m[12] * m[3] * m[5];

			inv[14] = -m[0] * m[5] * m[14] +
				m[0] * m[6] * m[13] +
				m[4] * m[1] * m[14] -
				m[4] * m[2] * m[13] -
				m[12] * m[1] * m[6] +
				m[12] * m[2] * m[5];

			inv[3] = -m[1] * m[6] * m[11] +
				m[1] * m[7] * m[10] +
				m[5] * m[2] * m[11] -
				m[5] * m[3] * m[10] -
				m[9] * m[2] * m[7] +
				m[9] * m[3] * m[6];

			inv[7] = m[0] * m[6] * m[11] -
				m[0] * m[7] * m[10] -
				m[4] * m[2] * m[11] +
				m[4] * m[3] * m[10] +
				m[8] * m[2] * m[7] -
				m[8] * m[3] * m[6];

			inv[11] = -m[0] * m[5] * m[11] +
				m[0] * m[7] * m[9] +
				m[4] * m[1] * m[11] -
				m[4] * m[3] * m[9] -
				m[8] * m[1] * m[7] +
				m[8] * m[3] * m[5];

			inv[15] = m[0] * m[5] * m[10] -
				m[0] * m[6] * m[9] -
				m[4] * m[1] * m[10] +
				m[4] * m[2] * m[9] +
				m[8] * m[1] * m[6] -
				m[8] * m[2] * m[5];

			det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

			det = 1.0f / det;

			for (i = 0; i < 16; i++)
				im[i] = inv[i] * det;
		}
	};
}