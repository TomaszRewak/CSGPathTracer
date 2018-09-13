#pragma once

#include "component-configurations.h"

namespace Communication
{
	__device__ void mapComponents(
		Communication::Component* zippedComponents, size_t componentsNumber,
		PathTracer::Component*& components,
		PathTracer::Component**& shapeComponents, size_t& shapeComponentsNumber,
		PathTracer::Component**& lightComponents, size_t& lightComponentsNumber)
	{
		shapeComponentsNumber = 0;
		lightComponentsNumber = 0;

		components = new PathTracer::Component[componentsNumber];
		shapeComponents = new PathTracer::Component*[componentsNumber];
		lightComponents = new PathTracer::Component*[componentsNumber];

		for (size_t i = 0; i < componentsNumber; i++)
		{
			PathTracer::Component& component = components[i];
			Communication::Component& shape = zippedComponents[i];

			component.globalTransformation = Math::TwoWayAffineTransformation(shape.globalTransformation);
			component.shader = shape.shader;

			switch (shape.type)
			{
			case Communication::ComponentType::Sphere:
				component.configuration = Configurations::sphereConfiguration();
				break;
			case Communication::ComponentType::Cylinder:
				component.configuration = Configurations::cylinderConfiguration();
				break;
			case Communication::ComponentType::Plane:
				component.configuration = Configurations::planeConfiguration();
				break;
			case Communication::ComponentType::Union:
				component.configuration = Configurations::unionConfiguration();
				component.leftOperand = &components[i + shape.leftOperandOffset];
				component.rightOperand = &components[i + shape.rightOperandOffset];
				break;
			case Communication::ComponentType::Difference:
				component.configuration = Configurations::differenceConfiguration();
				component.leftOperand = &components[i + shape.leftOperandOffset];
				component.rightOperand = &components[i + shape.rightOperandOffset];
				break;
			case Communication::ComponentType::Intersection:
				component.configuration = Configurations::intersectionConfiguration();
				component.leftOperand = &components[i + shape.leftOperandOffset];
				component.rightOperand = &components[i + shape.rightOperandOffset];
				break;
			}

			if (component.leftOperand)
			{
				component.leftOperand->parent = &component;
				component.leftOperand->normalDirection = component.normalDirection * component.configuration.leftChildNormalDirection;
			}

			if (component.rightOperand)
			{
				component.rightOperand->parent = &component;
				component.rightOperand->normalDirection = component.normalDirection * component.configuration.rightChildNormalDirection;
			}

			if (component.configuration.localIntersectionFunction)
				shapeComponents[shapeComponentsNumber++] = &component;

			if (component.shader.isLightSource())
				lightComponents[lightComponentsNumber++] = &component;
		}
	}
}