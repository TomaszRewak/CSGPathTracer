#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <math.h>

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

struct Ray
{
	Point begin;
	Vector direction;

	__host__ __device__ Ray(Point& begin, Vector& direction) :
		begin(begin), direction(direction)
	{}
	__host__ __device__ Ray(Point& begin, Point& through) :
		begin(begin), direction(through - begin)
	{}
};

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

	__host__ AffineTransformation(const Point& p, const Vector& v)
	{
		Vector castedVector(v.dx, 0, v.dz);

		float yAngle = castedVector.angle(Vector(0, 0, 1));
		float xAngle = castedVector.angle(v);

		if (v.dy == 0)
			xAngle = 0;
		if (v.dx == 0)
			yAngle = 0;

		if (v.dy > 0)
			xAngle = -xAngle;
		if (v.dx < 0)
			yAngle = -yAngle;

		float d = v.norm();

		*this = AffineTransformation().translate(0, 0, d).rotateX(xAngle).rotateY(yAngle).translate(p.x, p.y, p.z);
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

	__device__ TwoWayAffineTransformation(const AffineTransformation& affineTransformation) :
		AffineTransformation(affineTransformation)
	{
		inverse();
	}

	__device__ TwoWayAffineTransformation rotateX(float angle) const
	{
		return AffineTransformation::rotateX(angle);
	}

	__device__ TwoWayAffineTransformation rotateY(float angle) const
	{
		return AffineTransformation::rotateY(angle);
	}

	__device__ TwoWayAffineTransformation rotateZ(float angle) const
	{
		return AffineTransformation::rotateZ(angle);
	}

	__device__ TwoWayAffineTransformation translate(float x, float y, float z) const
	{
		return AffineTransformation::translate(x, y, z);
	}

	__device__ TwoWayAffineTransformation scale(float x, float y, float z) const
	{
		return AffineTransformation::scale(x, y, z);
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