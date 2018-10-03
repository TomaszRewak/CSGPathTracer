#include "../Shading/shader.h"

namespace PathTracer
{
	namespace Tracing
	{
		struct PathStep
		{
			Shading::Color color;
			Math::Ray ray;
			float roughness;

			__device__ PathStep() :
				roughness(0)
			{ }

			__device__ PathStep(Shading::Color color, const Math::Ray& ray, float roughness) :
				color(color),
				ray(ray),
				roughness(roughness)
			{ }
		};
	}
}