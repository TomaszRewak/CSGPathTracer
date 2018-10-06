#pragma once

#include "component-type.h"
#include "component-intersection.h"
#include "component-photon.h"
#include "../../Math/shapes.h"

namespace PathTracer
{
#define RAY_EPSYLON 0.1f

	struct Component
	{
		ComponentType type;

		Math::TwoWayAffineTransformation globalTransformation;

		Shading::Shader shader;

		Component* parent;
		Component* leftOperand;
		Component* rightOperand;

		float normalDirection;
		float totalPhotons;

		__device__ Component() :
			parent(NULL), leftOperand(NULL), rightOperand(NULL),
			normalDirection(1.f),
			totalPhotons(0.f)
		{ }

		__device__ void intersect(
			const Math::Ray& ray,
			ComponentIntersection &intersection) const
		{
			float minDistance = 0;

			while (minDistance < intersection.distance)
			{
				ComponentIntersection intersectionCandidate(intersection.distance);

				getIntersectionCandidate(ray, minDistance, intersectionCandidate);

				if (!intersectionCandidate.component)
					break;

				Math::Point point = ray.point(intersectionCandidate.distance);

				if (pointInside(point, intersectionCandidate.component) == InsideKind::Surface)
					intersection = intersectionCandidate;

				minDistance = intersectionCandidate.distance;
			}
		}

		__device__ void generatePhoton(ComponentPhoton &photon, float& photonsLeft, curandState &rand) const
		{
			if (photon.component != NULL)
				return;

			generatePhotonCandidate(photon, photonsLeft, rand);

			if (photon.component != NULL && pointInside(photon.ray.begin, photon.component) != InsideKind::Surface)
				photon.strength = 0;

			photon.ray.begin = photon.ray.begin + photon.ray.direction.unitVector() * RAY_EPSYLON;
		}

		__device__ Math::Vector normalVector(const Math::Point &point) const
		{
			Math::Point localPoint = globalTransformation.inverse(point);
			Math::Vector vector;

			switch (type)
			{
			case ComponentType::Sphere:
				vector = Math::Sphere::getNormalVector(localPoint);
				break;
			case ComponentType::Cylinder:
				vector = Math::Cylinder::getNormalVector(localPoint);
				break;
			case ComponentType::Plane:
				vector = Math::Plane::getNormalVector(localPoint);
				break;
			}

			return globalTransformation.transform(vector * normalDirection);
		}

	private:
		__device__ void getIntersectionCandidate(
			const Math::Ray &ray,
			float minDistance,
			ComponentIntersection &intersection) const
		{
			if (int(type) & int(ComponentType::Shape))
			{
				Math::Ray localRay = globalTransformation.inverse(ray);

				float newDistance = INFINITY;

				switch (type)
				{
				case ComponentType::Sphere:
					newDistance = Math::Sphere::intersect(localRay, minDistance);
					break;
				case ComponentType::Cylinder:
					newDistance = Math::Cylinder::intersect(localRay, minDistance);
					break;
				case ComponentType::Plane:
					newDistance = Math::Plane::intersect(localRay, minDistance);
					break;
				}

				if (newDistance < intersection.distance)
				{
					intersection.distance = newDistance;
					intersection.component = this;
				}
			}
			else
			{
				leftOperand->getIntersectionCandidate(ray, minDistance, intersection);
				rightOperand->getIntersectionCandidate(ray, minDistance, intersection);
			}
		}

		__device__ void generatePhotonCandidate(ComponentPhoton& photon, float& photonsLeft, curandState& rand) const
		{
			if (totalPhotons <= 0 || photonsLeft <= 0)
				return;

			if (int(type) & int(ComponentType::Shape))
			{
				if (photonsLeft > 0 && curand_uniform(&rand) < totalPhotons / photonsLeft)
				{
					Math::Ray newRayCandidate;

					switch (type)
					{
					case ComponentType::Sphere:
						newRayCandidate = Math::Sphere::generateRandomSurfaceRay(rand);
						break;
					case ComponentType::Cylinder:
						newRayCandidate = Math::Cylinder::generateRandomSurfaceRay(rand);
						break;
					case ComponentType::Plane:
						newRayCandidate = Math::Plane::generateRandomSurfaceRay(rand);
						break;
					}

					photon.ray = globalTransformation.transform(newRayCandidate);
					photon.ray.direction = photon.ray.direction * this->normalDirection;
					photon.component = this;
					photon.strength = totalPhotons;
					photonsLeft = 0;
				}
				else
				{
					photonsLeft -= totalPhotons;
				}
			}
			else
			{
				leftOperand->generatePhotonCandidate(photon, photonsLeft, rand);
				rightOperand->generatePhotonCandidate(photon, photonsLeft, rand);
			}
		}

		enum struct InsideKind {
			Outside = 0,
			Inside = 1,
			Surface = 2
		};

		__device__ InsideKind pointInside(const Math::Point& point, const Component* surfaceComponent) const
		{
			if (surfaceComponent == this)
				return InsideKind::Surface;

			if (int(type) & int(ComponentType::Shape))
			{
				Math::Point localPoint = globalTransformation.inverse(point);
				bool inside = false;

				switch (type)
				{
				case ComponentType::Sphere:
					return InsideKind(Math::Sphere::pointInside(localPoint));
				case ComponentType::Cylinder:
					return InsideKind(Math::Cylinder::pointInside(localPoint));
				case ComponentType::Plane:
					return InsideKind(Math::Plane::pointInside(localPoint));
				}
			}
			else
			{
				InsideKind left = leftOperand->pointInside(point, surfaceComponent);
				InsideKind right = rightOperand->pointInside(point, surfaceComponent);

				if (left == InsideKind::Surface)
				{
					if (type == ComponentType::Union && right == InsideKind::Outside ||
						type == ComponentType::Intersection && right == InsideKind::Inside ||
						type == ComponentType::Difference && right == InsideKind::Outside)
						return InsideKind::Surface;
					else
						return InsideKind::Outside;
				}

				if (right == InsideKind::Surface)
				{
					if (type == ComponentType::Union && left == InsideKind::Outside ||
						type == ComponentType::Intersection && left == InsideKind::Inside ||
						type == ComponentType::Difference && left == InsideKind::Inside)
						return InsideKind::Surface;
					else
						return InsideKind::Outside;
				}

				switch (type)
				{
				case ComponentType::Union:
					return InsideKind(left == InsideKind::Inside || right == InsideKind::Inside);
				case ComponentType::Intersection:
					return InsideKind(left == InsideKind::Inside && right == InsideKind::Inside);
				case ComponentType::Difference:
					return InsideKind(left == InsideKind::Inside && right == InsideKind::Outside);
				}
			}
		}
	};
}