#include "../Shading/shader.h"

namespace PathTracer
{
	namespace Tracing
	{
		struct PathStep
		{
			Shading::Shading shading;
			Math::Ray baseRay;

			PathStep()
			{ }

			PathStep(const Shading::Shading& shading, const Math::Ray& baseRay) :
				shading(shading),
				baseRay(baseRay)
			{ }
		};
	}
}