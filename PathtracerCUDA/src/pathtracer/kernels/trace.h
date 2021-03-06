#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <cstdint>
#include "./../Camera.h"

class Hittable;
struct BVHNode;

__global__ void traceKernel(
	float4 *accumBuffer,
	bool ignoreHistory,
	uint32_t accumulatedSampleCount,
	uint32_t width,
	uint32_t height,
	uint32_t spp,
	uint32_t hittableCount,
	Hittable *world,
	uint32_t bvhNodesCount,
	BVHNode *bvhNodes,
	curandState *randState,
	Camera camera,
	uint32_t skyboxTextureHandle,
	cudaTextureObject_t *textures);