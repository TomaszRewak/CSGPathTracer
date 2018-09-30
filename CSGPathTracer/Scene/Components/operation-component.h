#pragma once

#include "component-base.h"
#include <memory>

namespace Scene
{
	template<PathTracer::ComponentType OperationType>
	class OperationComponent : public Component
	{
	protected:
		std::shared_ptr<Component> leftOperand;
		std::shared_ptr<Component> rightOperand;

	public:
		OperationComponent(
			Math::AffineTransformation localTransformation,
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			Component(localTransformation),
			leftOperand(leftOperand),
			rightOperand(rightOperand)
		{ }

		virtual size_t zip(Communication::Component* dst, const Math::AffineTransformation& globalTransformation) const
		{
			auto transformation = localTransformation.combine(globalTransformation);

			size_t leftOperandSize = leftOperand->zip(dst + 1, transformation);
			size_t rightOperandSize = rightOperand->zip(dst + 1 + leftOperandSize, transformation);

			*dst = Communication::Component(
				OperationType,
				transformation,
				1,
				1 + leftOperandSize,
				PathTracer::Shading::Shader()
			);

			return 1 + leftOperandSize + rightOperandSize;
		}

		virtual size_t zipSize() const
		{
			return 1 + leftOperand->zipSize() + rightOperand->zipSize();
		}
	};

	class UnionComponent : public OperationComponent<PathTracer::ComponentType::Union> {
	public:
		UnionComponent(
			Math::AffineTransformation localTransformation,
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			OperationComponent(localTransformation, leftOperand, rightOperand)
		{ }

		UnionComponent(
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			UnionComponent(Math::AffineTransformation(), leftOperand, rightOperand)
		{ }
	};

	class DifferenceComponent : public OperationComponent<PathTracer::ComponentType::Difference> {
	public:
		DifferenceComponent(
			Math::AffineTransformation localTransformation,
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			OperationComponent(localTransformation, leftOperand, rightOperand)
		{ }

		DifferenceComponent(
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			DifferenceComponent(Math::AffineTransformation(), leftOperand, rightOperand)
		{ }
	};

	class IntersectionComponent : public OperationComponent<PathTracer::ComponentType::Intersection> {
	public:
		IntersectionComponent(
			Math::AffineTransformation localTransformation,
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			OperationComponent(localTransformation, leftOperand, rightOperand)
		{ }

		IntersectionComponent(
			std::shared_ptr<Component> leftOperand,
			std::shared_ptr<Component> rightOperand
		) :
			IntersectionComponent(Math::AffineTransformation(), leftOperand, rightOperand)
		{ }
	};
}