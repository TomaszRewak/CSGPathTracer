#pragma once

namespace Communication
{
	enum struct ComponentType : int
	{
		None,
		Union,
		Difference,
		Intersection,
		Sphere,
		Cylinder,
		Plane
	};
}