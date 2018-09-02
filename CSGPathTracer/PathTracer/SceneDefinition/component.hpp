#pragma once

#include "../Communication/component.hpp"
#include "../../Math/geometry.h"

namespace PathTracer
{
	namespace SceneDefinition
	{
		class Component {
		protected:
			Math::AffineTransformation localTransformation;

		public:
			Component(Math::AffineTransformation localTransformation) :
				localTransformation(localTransformation)
			{ }

			virtual size_t zip(Communication::Component* dst, const Math::AffineTransformation& globalTransformation) const = 0;
			virtual size_t zipSize() const = 0;
		};
	}
}