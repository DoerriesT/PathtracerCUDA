#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <cstdint>

__global__ void tonemap(uchar4 *resultBuffer, float4 *accumBuffer, uint32_t width, uint32_t height, uint32_t accumulatedSampleCount);