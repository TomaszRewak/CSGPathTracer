#pragma once

#include "component.hpp"

namespace PathTracer
{
	namespace SceneDefinition
	{
		template<Common::ComponentType ShapeType>
		class ShapeComponent : public Component
		{
		public:
			ShapeComponent(Math::AffineTransformation localTransformation) :
				Component(localTransformation)
			{ }

			virtual size_t zip(Communication::Component* dst, const Math::AffineTransformation& globalTransformation) const
			{
				*dst = Communication::Component(
					ShapeType,
					localTransformation.combine(globalTransformation)
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
			SphereComponent(Math::AffineTransformation localTransformation) :
				ShapeComponent(localTransformation)
			{ }
		};

		class PlaneComponent : public ShapeComponent<Common::ComponentType::Plane> {
		public:
			PlaneComponent(Math::AffineTransformation localTransformation) :
				ShapeComponent(localTransformation)
			{ }
		};

		class CylinderComponent : public ShapeComponent<Common::ComponentType::Cylinder> {
		public:
			CylinderComponent(Math::AffineTransformation localTransformation) :
				ShapeComponent(localTransformation)
			{ }
		};
	}
}