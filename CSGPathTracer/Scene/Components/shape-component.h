#pragma once

#include "component-base.h"

namespace Scene
{
	template<PathTracer::ComponentType ShapeType>
	class ShapeComponent : public Component
	{
	protected:
		PathTracer::Shading::Shader shader;

	public:
		ShapeComponent(Math::AffineTransformation localTransformation, PathTracer::Shading::Shader shader) :
			Component(localTransformation),
			shader(shader)
		{ }

		virtual size_t zip(Communication::Component* dst, const Math::AffineTransformation& globalTransformation) const
		{
			*dst = Communication::Component(
				ShapeType,
				localTransformation.combine(globalTransformation),
				0, 0,
				shader
			);

			return 1;
		}

		virtual size_t zipSize() const
		{
			return 1;
		}
	};

	class SphereComponent : public ShapeComponent<PathTracer::ComponentType::Sphere> {
	public:
		SphereComponent(Math::AffineTransformation localTransformation, PathTracer::Shading::Shader shader) :
			ShapeComponent(localTransformation, shader)
		{ }

		SphereComponent(PathTracer::Shading::Shader shader) :
			SphereComponent(Math::AffineTransformation(), shader)
		{ }
	};

	class PlaneComponent : public ShapeComponent<PathTracer::ComponentType::Plane> {
	public:
		PlaneComponent(Math::AffineTransformation localTransformation, PathTracer::Shading::Shader shader) :
			ShapeComponent(localTransformation, shader)
		{ }

		PlaneComponent(PathTracer::Shading::Shader shader) :
			PlaneComponent(Math::AffineTransformation(), shader)
		{ }
	};

	class CylinderComponent : public ShapeComponent<PathTracer::ComponentType::Cylinder> {
	public:
		CylinderComponent(Math::AffineTransformation localTransformation, PathTracer::Shading::Shader shader) :
			ShapeComponent(localTransformation, shader)
		{ }

		CylinderComponent(PathTracer::Shading::Shader shader) :
			CylinderComponent(Math::AffineTransformation(), shader)
		{ }
	};
}