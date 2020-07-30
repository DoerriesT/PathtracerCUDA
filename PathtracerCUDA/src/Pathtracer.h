#pragma once
#include "BVH.h"
#include "device_types.h"

class Camera;
class Hittable;
class CpuHittable;
struct cudaGraphicsResource;

class Pathtracer
{
public:
	explicit Pathtracer(uint32_t width, uint32_t height, unsigned int openglPixelBuffer = 0);
	~Pathtracer();
	void setScene(size_t count, const CpuHittable *hittables);
	void render(const Camera &camera, uint32_t spp, bool ignoreHistory);
	float getTiming() const;
	uint32_t loadTexture(const char *path);
	void setSkyboxTextureHandle(uint32_t handle);
	float *getImageData();

private:
	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_hittableCount;
	uint32_t m_nodeCount;
	float m_timing;
	uint32_t m_accumulatedFrames;
	uint32_t m_textureCount = 0;
	uint32_t m_skyboxTextureHandle = 0;

	// gpu resources
	cudaGraphicsResource *m_pixelBufferCuda = nullptr;
	float4 *m_gpuAccumBuffer = nullptr;
	curandState *m_gpuRandState = nullptr;
	cudaEvent_t m_startEvent = nullptr;
	cudaEvent_t m_stopEvent = nullptr;
	BVHNode *m_gpuBVHNodes = nullptr;
	Hittable *m_gpuHittables = nullptr;
	cudaTextureObject_t *m_textures = nullptr;
	cudaTextureObject_t *m_gpuTextures = nullptr;
	cudaArray **m_textureMemory = nullptr;
};