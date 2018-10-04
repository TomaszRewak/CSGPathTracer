#include "../Shading/shader.h"

namespace PathTracer
{
	namespace Tracing
	{
		struct PathStep
		{
			Math::Ray ray;
			Math::Vector normalVector;
			Shading::Shading shading;

			__device__ PathStep()
			{ }

			__device__ PathStep(const Math::Ray& ray, const Shading::Shading& shading) :
				ray(ray),
				normalVector(ray.direction),
				shading(shading)
			{ }

			__device__ PathStep(const Math::Ray& ray, const Math::Vector& normalVector, const Shading::Shading& shading) :
				ray(ray),
				normalVector(normalVector),
				shading(shading)
			{ }
		};
	}
}