#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "math.hpp"

enum struct ShapeType : int
{
	Sphere
};

struct Shape
{
	ShapeType type;
	AffineTransformation transformation;

	__host__ Shape(ShapeType type, AffineTransformation transformation) :
		type(type),
		transformation(transformation)
	{ }
};

struct ShapeSpace
{
	ShapeType type;
	TwoWayAffineTransformation transformation;

	__device__ ShapeSpace(const Shape& shape) :
		type(shape.type),
		transformation(shape.transformation)
	{ }
};

struct Intersection
{
	bool valid;
	Point position;
	Vector normalVector;

	__device__ Intersection() :
		valid(false)
	{ }

	__device__ Intersection(Point position, Vector normalVector) :
		valid(true),
		position(position),
		normalVector(normalVector)
	{ }
};

void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Shape* shapes, size_t shapesNumber);