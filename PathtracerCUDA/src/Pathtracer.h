#pragma once
#include "BVH.h"
#include "device_types.h"

class Camera;
class Hittable;
struct cudaGraphicsResource;

class Pathtracer
{
public:
	explicit Pathtracer(uint32_t width, uint32_t height);
	~Pathtracer();
	void setBVH(uint32_t nodeCount, const BVHNode *nodes, uint32_t hittableCount, const Hittable *hittables);
	void render(const Camera &camera, bool ignoreHistory);
	float getTiming() const;

private:
	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_hittableCount;
	uint32_t m_nodeCount;
	float m_timing;
	uint32_t m_accumulatedFrames;

	// gpu resources
	unsigned int m_pixelBufferGL = 0;
	cudaGraphicsResource *m_pixelBufferCuda = nullptr;
	float4 *m_gpuAccumBuffer = nullptr;
	curandState *m_gpuRandState = nullptr;
	cudaEvent_t m_startEvent = nullptr;
	cudaEvent_t m_stopEvent = nullptr;
	BVHNode *m_gpuBVHNodes = nullptr;
	Hittable *m_gpuHittables = nullptr;
	cudaTextureObject_t m_skyboxTexture = 0;
	cudaArray *m_skyboxTextureMemory = nullptr;
};