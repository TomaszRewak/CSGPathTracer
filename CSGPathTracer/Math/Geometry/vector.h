#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace Math
{
	struct Vector
	{
		float dx;
		float dy;
		float dz;

		__host__ __device__ Vector() :
			dx(0), dy(0), dz(0) {};
		__host__ __device__ Vector(float dx, float dy, float dz) :
			dx(dx), dy(dy), dz(dz) {}
		__host__ __device__ Vector(float dx, float dy, float dz, float w) :
			dx(dx / w), dy(dy / w), dz(dz / w) {};

		__host__ __device__ float dotProduct(const Vector& b) const
		{
			return dx * b.dx + dy * b.dy + dz * b.dz;
		}

		__host__ __device__ Vector crossProduct(const Vector& b) const
		{
			return Vector(
				dy * b.dz - dz * b.dy,
				dz * b.dx - dx * b.dz,
				dx * b.dy - dy * b.dx
			);
		}

		__host__ __device__ float norm2() const
		{
			return dotProduct(*this);
		}

		__host__ __device__ float norm() const
		{
			return sqrtf(norm2());
		}

		__host__ __device__ float angle(const Vector& b) const
		{
			return acosf(unitVector().dotProduct(b.unitVector()));
		}

		__host__ __device__ float angleX() const
		{
			return Vector(0, dy, dx).angle(Vector(1, 0, 0));
		}

		__host__ __device__ float angleY() const
		{
			return Vector(dx, 0, dz).angle(Vector(0, 1, 0));
		}

		__host__ __device__ float angleZ() const
		{
			return Vector(dx, dy, 0).angle(Vector(0, 0, 1));
		}

		__host__ __device__ Vector unitVector() const
		{
			float rd = 1 / norm();

			return Vector(
				dx * rd,
				dy * rd,
				dz * rd
			);
		}

		__host__ __device__ Vector operator-() const
		{
			return Vector(
				-dx,
				-dy,
				-dz
			);
		}

		__host__ __device__ Vector operator*(float by) const
		{
			return Vector(
				dx * by,
				dy * by,
				dz * by
			);
		}

		__host__ __device__ Vector operator+(const Vector &v) const
		{
			return Vector(
				dx + v.dx,
				dy + v.dy,
				dz + v.dz
			);
		}

		__host__ __device__ Vector operator-(const Vector &v) const
		{
			return Vector(
				dx - v.dx,
				dy - v.dy,
				dz - v.dz
			);
		}
	};
}