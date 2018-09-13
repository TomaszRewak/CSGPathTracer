#pragma once

#include "color.h"

namespace PathTracer
{
	namespace Shading
	{
		enum struct ShaderType : int
		{
			None,
			Uniform
		};

		struct Shading
		{
			float emission;
			float specularity;
			float density;

			float reflectionProbability;
			float refractionProbability;

			Color color;

			__host__ __device__ Shading(
				float emission, float specularity, float density, 
				float reflectionProbability, float refractionProbability, 
				Color color
			) :
				emission(emission), specularity(specularity), density(density), 
				reflectionProbability(reflectionProbability), refractionProbability(refractionProbability), 
				color(color)
			{}
		};

		class Shader
		{
		private:
			ShaderType shaderType;
			Shading shading;

		public:
			__host__ __device__ Shader() :
				shaderType(ShaderType::None),
				shading(0, 0, 0, 0, 0, Color())
			{ }

			__host__ Shader(ShaderType shaderType, Shading shading) :
				shaderType(shaderType),
				shading(shading)
			{ }

			__host__ __device__ Shader(const Shader& shader) = default;

			__device__ Shading getShading(const Math::Point& point) const
			{
				return shading;
			}

			__device__ bool isLightSource() const
			{
				return shading.emission > 0;
			}
		};
	}
}