#pragma once

#include "kernel.h"
#include <math.h>
#include <new.h>

#define N 128

struct NoneShapeComponent : public Component
{
	__device__ NoneShapeComponent(const AffineTransformation& transformation) :
		Component(transformation)
	{ }

	__device__ virtual Intersection intersectLocally(Ray ray)
	{
		return Intersection();
	}

	__device__ virtual bool validateLocally(Point point, Component* currentComponent)
	{
		return false;
	}
};

struct SphereShapeComponent : public Component
{
	__device__ SphereShapeComponent(const AffineTransformation& transformation) :
		Component(transformation)
	{ }

	__device__ virtual Intersection intersectLocally(Ray ray)
	{
		float a, b, c;

		a = ray.direction.dx * ray.direction.dx + ray.direction.dy * ray.direction.dy + ray.direction.dz * ray.direction.dz;
		b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.y * ray.direction.dy + ray.begin.z * ray.direction.dz);
		c = ray.begin.x * ray.begin.x + ray.begin.y * ray.begin.y + ray.begin.z * ray.begin.z - 1.f;

		float delta = b * b - 4 * a * c;

		if (delta >= 0.0f)
		{
			float deltaSqrt = sqrtf(delta);

			float d1 = (-b - deltaSqrt) / (2 * a);
			float d2 = (-b + deltaSqrt) / (2 * a);

			if (d1 > d2)
			{
				float d3 = d1;
				d1 = d2;
				d2 = d3;
			}

			if (d1 > 0.0f)
			{
				Point intersectionPoint = ray.begin + ray.direction * d1;

				if (validateUp(intersectionPoint, this))
					return Intersection(intersectionPoint, intersectionPoint - Point(0, 0, 0));
			}

			if (d2 > 0.0f)
			{
				Point intersectionPoint = ray.begin + ray.direction * d2;

				if (validateUp(intersectionPoint, this))
					return Intersection(intersectionPoint, intersectionPoint - Point(0, 0, 0));
			}
		}

		return Intersection();
	}

	__device__ virtual bool validateLocally(Point point, Component* component)
	{
		return point.x * point.x + point.y * point.y + point.z * point.z <= 1.0f;
	}
};

struct CylinderShapeComponent : public Component
{
	__device__ CylinderShapeComponent(const AffineTransformation& transformation) :
		Component(transformation)
	{ }

	__device__ virtual Intersection intersectLocally(Ray ray)
	{
		if (!ray.direction.dx && !ray.direction.dz)
			return Intersection();

		float a, b, c;

		a = ray.direction.dx * ray.direction.dx + ray.direction.dz * ray.direction.dz;
		b = 2.f * (ray.begin.x * ray.direction.dx + ray.begin.z * ray.direction.dz);
		c = ray.begin.x * ray.begin.x + ray.begin.z * ray.begin.z - 1.f;

		float delta = b * b - 4 * a * c;

		if (delta >= 0.0f)
		{
			float deltaSqrt = sqrtf(delta);

			float d1 = (-b - deltaSqrt) / (2 * a);
			float d2 = (-b + deltaSqrt) / (2 * a);

			if (d1 > d2)
			{
				float d3 = d1;
				d1 = d2;
				d2 = d3;
			}

			if (d1 > 0.0f)
			{
				Point intersectionPoint = ray.begin + ray.direction * d1;

				if (validateUp(intersectionPoint, this))
					return Intersection(intersectionPoint, intersectionPoint - Point(0, intersectionPoint.y, 0));
			}

			if (d2 > 0.0f)
			{
				Point intersectionPoint = ray.begin + ray.direction * d2;

				if (validateUp(intersectionPoint, this))
					return Intersection(intersectionPoint, intersectionPoint - Point(0, intersectionPoint.y, 0));
			}
		}

		return Intersection();
	}

	__device__ virtual bool validateLocally(Point point, Component* component)
	{
		return point.x * point.x + point.z * point.z <= 1.0f;
	}
};

struct PlaneShapeComponent : public Component
{
	__device__ PlaneShapeComponent(const AffineTransformation& transformation) :
		Component(transformation)
	{ }

	__device__ virtual Intersection intersectLocally(Ray ray)
	{
		if (!ray.direction.dx)
			return Intersection();

		float d = -ray.begin.x / ray.direction.dx;

		if (d > 0.0f)
		{
			Point intersectionPoint = ray.begin + ray.direction * d;

			if (validateUp(intersectionPoint, this))
				return Intersection(intersectionPoint, Vector(0, 1, 0));
		}

		return Intersection();
	}

	__device__ virtual bool validateLocally(Point point, Component* component)
	{
		return point.x <= 1.0f;
	}
};

struct OperationComponent : public Component
{
	Component* leftOperand;
	Component* rightOperand;

	__device__ OperationComponent(const AffineTransformation& transformation) :
		Component(transformation)
	{ }

	__device__ virtual Intersection intersectLocally(Ray ray)
	{
		Intersection leftIntersection = leftOperand->intersect(ray);
		Intersection rightIntersection = rightOperand->intersect(ray);

		if (!rightIntersection.hit)
			return leftIntersection;
		if (!leftIntersection.hit)
			return rightIntersection;

		float dLeft = (leftIntersection.position - ray.begin).norm2();
		float dRight = (rightIntersection.position - ray.begin).norm2();

		if (dLeft < dRight)
			return leftIntersection;
		else
			return rightIntersection;
	}
};

struct UnionOperationComponent : public OperationComponent
{
	__device__ UnionOperationComponent(const AffineTransformation& transformation) :
		OperationComponent(transformation)
	{ }

	__device__ virtual bool validateLocally(Point point, Component* currentComponent)
	{
		if (leftOperand == currentComponent)
			return !rightOperand->validate(point, this);

		if (rightOperand == currentComponent)
			return !leftOperand->validate(point, this);

		return leftOperand->validate(point, this) || rightOperand->validate(point, this);
	}
};

struct IntersectionOperationComponent : public OperationComponent
{
	__device__ IntersectionOperationComponent(const AffineTransformation& transformation) :
		OperationComponent(transformation)
	{ }

	__device__ virtual bool validateLocally(Point point, Component* currentComponent)
	{
		if (leftOperand == currentComponent)
			return rightOperand->validate(point, this);

		if (rightOperand == currentComponent)
			return leftOperand->validate(point, this);

		return leftOperand->validate(point, this) && rightOperand->validate(point, this);
	}
};

struct DifferenceOperationComponent : public OperationComponent
{
	__device__ DifferenceOperationComponent(const AffineTransformation& transformation) :
		OperationComponent(transformation)
	{ }

	__device__ virtual bool validateLocally(Point point, Component* currentComponent)
	{
		if (leftOperand == currentComponent)
			return !rightOperand->validate(point, this);

		if (rightOperand == currentComponent)
			return leftOperand->validate(point, this);

		return leftOperand->validate(point, this) && !rightOperand->validate(point, this);
	}
};

__device__
float trace(Ray& ray, Component* currentComponent)
{
	Intersection intersection = currentComponent->intersect(ray);

	if (intersection.hit)
	{
		float angle = Vector(0.5, 0.5, -1).unitVector().dotProduct(intersection.normalVector.unitVector());

		return (angle + 1.0f) * 0.5f;
	}
	else
	{
		return 0.0f;
	}
}

__device__
void copyShapesToSharedMemory(Component**& spaceComponents, Shape* shapes, size_t shapesNumber)
{
	if (threadIdx.x == 0) {
		size_t size = sizeof(Component*) * shapesNumber;
		spaceComponents = (Component**)malloc(size);

		for (size_t i = 0; i < shapesNumber; i++)
		{
			Shape& shape = shapes[i];

			switch (shape.type)
			{
			case ShapeType::None:
				spaceComponents[i] = new NoneShapeComponent(shape.transformation);
				break;
			case ShapeType::Union:
				spaceComponents[i] = new UnionOperationComponent(shape.transformation);
				break;
			case ShapeType::Difference:
				spaceComponents[i] = new DifferenceOperationComponent(shape.transformation);
				break;
			case ShapeType::Intersection:
				spaceComponents[i] = new IntersectionOperationComponent(shape.transformation);
				break;
			case ShapeType::Sphere:
				spaceComponents[i] = new SphereShapeComponent(shape.transformation);
				break;
			case ShapeType::Cylinder:
				spaceComponents[i] = new CylinderShapeComponent(shape.transformation);
				break;
			case ShapeType::Plane:
				spaceComponents[i] = new PlaneShapeComponent(shape.transformation);
				break;
			}
		}

		for (size_t i = 0; i < shapesNumber; i++)
		{
			Shape& shape = shapes[i];

			if (shape.leftOperand)
			{
				static_cast<OperationComponent*>(spaceComponents[i])->leftOperand = spaceComponents[shape.leftOperand];
				spaceComponents[shape.leftOperand]->parent = spaceComponents[i];
			}
			if (shape.rightOperand)
			{
				static_cast<OperationComponent*>(spaceComponents[i])->rightOperand = spaceComponents[shape.rightOperand];
				spaceComponents[shape.rightOperand]->parent = spaceComponents[i];
			}
		}
	}
	__syncthreads();
}

__device__
void freeShapes(Component** spaceComponents, size_t shapesNumber)
{
	__syncthreads();

	if (threadIdx.x == 0) {
		for (size_t i = 0; i < shapesNumber; i++)
			delete spaceComponents[i];
		delete[] spaceComponents;
	}
}

__global__
void kernel(uchar4* image, size_t imageWidth, size_t imageHeight, Shape* shapes, size_t shapesNumber)
{
	size_t n = threadIdx.x;
	size_t blockSize = imageWidth * imageHeight / N;
	size_t blockStart = n * blockSize;
	size_t blockEnd = blockStart + blockSize;

	__shared__ Component** spaceComponents;
	copyShapesToSharedMemory(spaceComponents, shapes, shapesNumber);

	for (size_t index = blockStart; index < blockEnd; index++)
	{
		float x = (float)(index % imageWidth) - imageWidth / 2;
		float y = (float)(index / imageWidth) - imageHeight / 2;

		Ray ray(Point(x, y, 0), Vector(0, 0, 1));

		float light = trace(ray, spaceComponents[0]);

		image[index].x = light * 255;
		image[index].y = light * 255;
		image[index].z = light * 255;
		image[index].w = light * 255;
	}

	freeShapes(spaceComponents, shapesNumber);
}

void renderRect(uchar4* image, const size_t imageWidth, const size_t imageHeight, Shape* shapes, size_t shapesNumber) {
	kernel << <1, N >> > (image, imageWidth, imageHeight, shapes, shapesNumber);
}