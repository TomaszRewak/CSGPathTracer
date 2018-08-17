#pragma once

#include "component.hpp"

namespace PathTracer
{
	namespace Scene
	{
		class LightSources
		{
		public:
			__device__ static Math::Ray generateLightRay(const Component* component, curandState& curandState)
			{
				if (component->type == Common::ComponentType::Sphere)
					return Math::Sphere::randomSurfaceRay(component->globalTransformation, curandState);
				else if (component->type == Common::ComponentType::Cylinder)
					return Math::Cylinder::randomSurfaceRay(component->globalTransformation, curandState);
				else if (component->type == Common::ComponentType::Plane)
					return Math::Plane::randomSurfaceRay(component->globalTransformation, curandState);
			}
		};
	}
}