#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

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
	AffineTransformation transformation;

	size_t leftOperand;
	size_t rightOperand;

	__host__ Shape(
		ShapeType type,
		AffineTransformation transformation,
		size_t leftOperand = 0,
		size_t rightOperand = 0
	) :
		type(type),
		transformation(transformation),
		leftOperand(leftOperand),
		rightOperand(rightOperand)
	{}
};

struct Intersection
{
	bool hit;

	Point position;
	Vector normalVector;

	__device__ Intersection() :
		hit(false)
	{ }

	__device__ Intersection(Point position, Vector normalVector) :
		hit(true),
		position(position),
		normalVector(normalVector)
	{ }
};

struct Component
{
	TwoWayAffineTransformation transformation;
	Component* parent;

	__device__ Component(const AffineTransformation& transformation) :
		transformation(transformation),
		parent(NULL)
	{ }

	__device__ virtual ~Component() = default;

	__device__ Intersection intersect(Ray ray)
	{
		Intersection intersection = intersectLocally(transformation.inverse(ray));

		if (intersection.hit)
			return Intersection(
				transformation.transform(intersection.position),
				transformation.transform(intersection.normalVector)
			);
		else
			return Intersection();
	}

	__device__ bool validate(Point point, Component* currentComponent)
	{
		return validateLocally(transformation.inverse(point), currentComponent);
	}

protected:
	__device__ virtual Intersection intersectLocally(Ray ray) = 0;
	__device__ virtual bool validateLocally(Point point, Component* currentComponent) = 0;

	__device__ bool validateUp(Point point, Component* currentComponent)
	{
		return (
			(currentComponent == this || validateLocally(point, currentComponent))
			&&
			(parent == NULL || parent->validateUp(transformation.transform(point), this))
			);
	}
};

void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Shape* shapes, size_t shapesNumber);