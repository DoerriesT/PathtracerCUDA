#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>

__global__ void initRandState(int width, int height, curandState *randState);