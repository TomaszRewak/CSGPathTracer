#pragma once

namespace PathTracer 
{
	namespace Common
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
}