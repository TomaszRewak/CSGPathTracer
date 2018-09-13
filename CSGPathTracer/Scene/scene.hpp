#pragma once

#include <vector>

#include "Components\operation-component.hpp"
#include "Components\shape-component.hpp"

namespace Scene
{
	class Scene
	{
	public:
		std::vector<std::shared_ptr<Component>> components;
		Math::AffineTransformation globalTransformation;

		size_t zip(Communication::Component* dst) const
		{
			size_t size = 0;

			for (const auto& component : components)
				size += component->zip(dst + size, globalTransformation);

			return size;
		}

		size_t zipSize() const
		{
			size_t size = 0;

			for (const auto& component : components)
				size += component->zipSize();

			return size;
		}
	};
}