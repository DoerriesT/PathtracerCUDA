#pragma once
#include "BVH.h"
#include "device_types.h"

class Camera;
class Hittable;
class CpuHittable;

// Pathtracer class renders images of a given scene and camera configuration using CUDA.
// the resulting image can optionally be copied to an OpenGL pixel buffer to be used in an
// OpenGL context.
class Pathtracer
{
public:
	explicit Pathtracer(uint32_t width, uint32_t height, unsigned int openglPixelBuffer = 0);
	Pathtracer(const Pathtracer &) = delete;
	Pathtracer(const Pathtracer &&) = delete;
	Pathtracer &operator= (const Pathtracer &) = delete;
	Pathtracer &operator= (const Pathtracer &&) = delete;
	~Pathtracer();

	void setScene(size_t count, const CpuHittable *hittables);

	// spp is the number of rays that are traced per pixel.
	// if ignoreHistory is false, samples are accumulated with previously computed results
	void render(const Camera &camera, uint32_t spp, bool ignoreHistory);
	
	// returns the gpu duration of the last render() call in milliseconds
	float getTiming() const;
	
	// tries to load a texture from the given path and returns a value != 0 on success.
	// this handle can be used to refer to the texture when creating a Material instance
	// or when setting the skybox. currently textures cannot be freed, instead a new Pathtracer
	// instance needs to be constructed. texture freeing could easily implemented, if the need arises
	uint32_t loadTexture(const char *path);
	
	// sets the skybox texture. a value of 0 indicates that there is no texture (initial state).
	void setSkyboxTextureHandle(uint32_t handle);

	// gets raw image data of the last render() call. each pixel consists of 4 floats
	float *getHDRImageData();
	
	// gets tonemapped and gamma-corrected data of the last render() call. each pixel consists of 4 chars
	char *getImageData();

private:
	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_hittableCount;
	uint32_t m_nodeCount;
	float m_timing;
	uint32_t m_accumulatedFrames;
	uint32_t m_textureCount = 0;
	uint32_t m_skyboxTextureHandle = 0;
	float *m_cpuAccumBuffer = nullptr;
	char *m_cpuResultBuffer = nullptr;

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