#include "initRandState.h"
#include <cstdint>

__global__ void initRandState(int width, int height, curandState *randState)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	uint32_t dstIdx = threadIDx + threadIDy * width;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984 + dstIdx, 0, 0, &randState[dstIdx]);
}