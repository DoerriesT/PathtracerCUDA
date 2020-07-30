#include "tonemap.h"
#include "./../vec3.h"

__global__ void tonemap(uchar4 *resultBuffer, float4 *accumBuffer, uint32_t width, uint32_t height, uint32_t accumulatedSampleCount)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	uint32_t dstIdx = threadIDx + threadIDy * width;

	vec3 resultColor = vec3(accumBuffer[dstIdx].x, accumBuffer[dstIdx].y, accumBuffer[dstIdx].z) / float(accumulatedSampleCount);

	// reinhard tonemapping
	resultColor = resultColor / (resultColor + 1.0f);// saturate(resultColor);

	// gamma correction
	resultColor.r = pow(resultColor.r, 1.0f / 2.2f);
	resultColor.g = pow(resultColor.g, 1.0f / 2.2f);
	resultColor.b = pow(resultColor.b, 1.0f / 2.2f);

	resultBuffer[dstIdx] = { (unsigned char)(resultColor.x * 255.0f), (unsigned char)(resultColor.y * 255.0f) , (unsigned char)(resultColor.z * 255.0f), 255 };
}
