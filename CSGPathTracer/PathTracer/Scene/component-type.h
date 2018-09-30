#pragma once

namespace PathTracer
{
	enum struct ComponentType : int
	{
		Shape =			0b0001,
		Sphere =		0b0101,
		Cylinder =		0b1001,
		Plane =			0b1101,

		Operation =		0b0010,
		Union =			0b0110,
		Difference =	0b1010,
		Intersection =	0b1110
	};
}