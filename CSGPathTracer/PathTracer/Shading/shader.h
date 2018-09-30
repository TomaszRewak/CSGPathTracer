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
			float density;
			float reflectance;
			float roughness;

			Color color;

			__host__ __device__ Shading() :
				emission(0),
				density(0),
				reflectance(0), 
				roughness(0)
			{}

			__host__ __device__ Shading(
				float emission, 
				float density,
				float reflectance, 
				float roughness, 
				const Color &color
			) :
				emission(emission), 
				density(density),
				reflectance(reflectance), 
				roughness(roughness),
				color(color)
			{}
		};

		class Shader
		{
		private:
			ShaderType shaderType;
			Shading shading;

		public:
			float photons;

			__host__ __device__ Shader() :
				shaderType(ShaderType::None),
				shading(0, 0, 0, 0, Color()),
				photons(0)
			{ }

			__host__ Shader(ShaderType shaderType, Shading shading, float photons = 0) :
				shaderType(shaderType),
				shading(shading),
				photons(photons)
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