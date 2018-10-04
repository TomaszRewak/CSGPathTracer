#include "../Shading/shader.h"

namespace PathTracer
{
	struct Component;

	namespace Tracing
	{
		struct PathStep
		{
			const Component* component;
			Math::Ray ray;
			Math::Vector normalVector;
			Shading::Shading shading;

			__device__ PathStep()
			{ }

			__device__ PathStep(const Component* component, const Math::Ray& ray, const Shading::Shading& shading) :
				component(component),
				ray(ray),
				normalVector(ray.direction),
				shading(shading)
			{ }

			__device__ PathStep(const Component* component, const Math::Ray& ray, const Math::Vector& normalVector, const Shading::Shading& shading) :
				component(component),
				ray(ray),
				normalVector(normalVector),
				shading(shading)
			{ }
		};
	}
}