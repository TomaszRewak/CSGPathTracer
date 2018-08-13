#pragma once

#include "math.hpp"

enum struct ShapeType : int
{
	None,
	Union,
	Difference,
	Intersection,
	Sphere,
	Cylinder,
	Plane
};

struct Shape
{
	ShapeType type;
	AffineTransformation localTransformation;

	size_t leftOperand;
	size_t rightOperand;

	__host__ Shape(
		ShapeType type,
		AffineTransformation transformation,
		size_t leftOperand = 0,
		size_t rightOperand = 0
	) :
		type(type),
		localTransformation(transformation),
		leftOperand(leftOperand),
		rightOperand(rightOperand)
	{}
};