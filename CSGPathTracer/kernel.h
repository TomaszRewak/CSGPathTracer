#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "PathTracer/Communication/component.hpp"

namespace PathTracer
{
	void renderRect(
		float4* image, 
		const size_t imageWidth, const size_t imageHeight, 
		Communication::Component* zippedComponents, size_t zippedComponentsNumber,
		size_t frameNumber);
}