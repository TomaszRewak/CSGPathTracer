#include "../Shading/shader.h"

namespace PathTracer
{
	namespace Tracing
	{
		struct PathStep
		{
			Shading::Shader shader;
			Math::Ray baseRay;
		};
	}
}