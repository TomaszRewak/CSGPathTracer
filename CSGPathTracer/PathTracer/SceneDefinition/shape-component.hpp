#pragma once

#include "component.hpp"

namespace PathTracer
{
	namespace SceneDefinition
	{
		template<Common::ComponentType ShapeType>
		class ShapeComponent : public Component
		{
		protected:
			Shading::Shader shader;

		public:
			ShapeComponent(Math::AffineTransformation localTransformation, Shading::Shader shader) :
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

		class SphereComponent : public ShapeComponent<Common::ComponentType::Sphere> {
		public:
			SphereComponent(Math::AffineTransformation localTransformation, Shading::Shader shader) :
				ShapeComponent(localTransformation, shader)
			{ }

			SphereComponent(Shading::Shader shader) :
				SphereComponent(Math::AffineTransformation(), shader)
			{ }
		};

		class PlaneComponent : public ShapeComponent<Common::ComponentType::Plane> {
		public:
			PlaneComponent(Math::AffineTransformation localTransformation, Shading::Shader shader) :
				ShapeComponent(localTransformation, shader)
			{ }

			PlaneComponent(Shading::Shader shader) :
				PlaneComponent(Math::AffineTransformation(), shader)
			{ }
		};

		class CylinderComponent : public ShapeComponent<Common::ComponentType::Cylinder> {
		public:
			CylinderComponent(Math::AffineTransformation localTransformation, Shading::Shader shader) :
				ShapeComponent(localTransformation, shader)
			{ }

			CylinderComponent(Shading::Shader shader) :
				CylinderComponent(Math::AffineTransformation(), shader)
			{ }
		};
	}
}