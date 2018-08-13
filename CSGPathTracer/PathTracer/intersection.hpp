#pragma once

#include "math.hpp"

struct Component;

struct Intersection
{
	bool hit;

	Point position;
	Vector normalVector;
	Component* component;

	__device__ Intersection() :
		hit(false),
		component(NULL)
	{ }

	__device__ Intersection(Point position, Vector normalVector, Component* component) :
		hit(true),
		position(position),
		normalVector(normalVector),
		component(component)
	{ }
};