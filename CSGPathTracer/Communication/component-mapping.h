#pragma once

namespace Communication
{
	__device__ void mapComponents(
		Communication::Component* zippedComponents, size_t componentsNumber,
		PathTracer::Component*& components,
		PathTracer::Component**& rootComponents, size_t& rootComponentsNumber)
	{
		rootComponentsNumber = 0;

		components = new PathTracer::Component[componentsNumber];
		rootComponents = new PathTracer::Component*[componentsNumber];

		for (size_t i = 0; i < componentsNumber; i++)
		{
			PathTracer::Component& component = components[i];
			Communication::Component& zippedComponent = zippedComponents[i];

			component.globalTransformation = Math::TwoWayAffineTransformation(zippedComponent.globalTransformation);
			component.shader = zippedComponent.shader;
			component.type = zippedComponent.type;

			if (int(component.type) & int(PathTracer::ComponentType::Operation))
			{
				component.leftOperand = &components[i + zippedComponent.leftOperandOffset];
				component.leftOperand->parent = &component;
				component.leftOperand->normalDirection = component.normalDirection;

				if (component.type == PathTracer::ComponentType::Intersection)
					component.leftOperand->normalDirection *= -1.;

				component.rightOperand = &components[i + zippedComponent.rightOperandOffset];
				component.rightOperand->parent = &component;
				component.rightOperand->normalDirection = component.normalDirection;

				if (component.type == PathTracer::ComponentType::Intersection || component.type == PathTracer::ComponentType::Difference)
					component.rightOperand->normalDirection *= -1.;
			}

			if (component.parent == NULL)
				rootComponents[rootComponentsNumber++] = &component;
		}

		for (size_t i = 0; i < componentsNumber; i++)
		{
			PathTracer::Component& component = components[componentsNumber - i - 1];

			component.totalPhotons += component.shader.photons;

			if (component.parent)
				component.parent->totalPhotons += component.totalPhotons;
		}
	}
}