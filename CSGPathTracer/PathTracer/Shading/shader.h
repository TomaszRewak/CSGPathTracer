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

			float translucency;
			float density;

			float reflectance;
			float roughness;

			Color color;

			__host__ __device__ Shading(
				float emission, 
				float translucency, float density,
				float reflectance, float roughness, 
				Color color
			) :
				emission(emission), 
				translucency(translucency), density(density),
				reflectance(reflectance), roughness(roughness),
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