#include "Pathtracer.h"
#include <glad/glad.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <curand_kernel.h>
#include "kernels/initRandState.h"
#include "kernels/trace.h"
#include "kernels/tonemap.h"
#include "stb_image.h"
#include <iostream>

// we internally manage an array of textures. MAX_TEXTURE_COUNT is the size of this array 
#define MAX_TEXTURE_COUNT 64

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
static void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

Pathtracer::Pathtracer(uint32_t width, uint32_t height, unsigned int openglPixelBuffer)
	:m_width(width),
	m_height(height),
	m_hittableCount(),
	m_nodeCount(),
	m_timing(),
	m_accumulatedFrames(),
	m_textures(new cudaTextureObject_t[MAX_TEXTURE_COUNT]),
	m_textureMemory(new cudaArray *[MAX_TEXTURE_COUNT])
{
	checkCudaErrors(cudaSetDevice(0));

	// optionally register an opengl pixel buffer with cuda
	if (openglPixelBuffer)
	{
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_pixelBufferCuda, openglPixelBuffer, cudaGraphicsMapFlagsWriteDiscard));
	}

	// alloc memory for prng state
	checkCudaErrors(cudaMalloc((void **)&m_gpuRandState, m_width * m_height * sizeof(curandState)));

	// init prng state
	dim3 threads(8, 8, 1);
	dim3 blocks((m_width + 7) / 8, (m_height + 7) / 8, 1);
	initRandState << <blocks, threads >> > (width, height, m_gpuRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// alloc memory for accum buffer
	checkCudaErrors(cudaMalloc((void **)&m_gpuAccumBuffer, m_width * m_height * sizeof(float4)));
	m_cpuAccumBuffer = new float[m_width * m_height * 4];

	m_cpuResultBuffer = new char[m_width * m_height * 4];

	checkCudaErrors(cudaEventCreate(&m_startEvent));
	checkCudaErrors(cudaEventCreate(&m_stopEvent));

	checkCudaErrors(cudaMalloc((void **)&m_gpuTextures, MAX_TEXTURE_COUNT * sizeof(cudaTextureObject_t)));
}

Pathtracer::~Pathtracer()
{
	checkCudaErrors(cudaDeviceSynchronize());

	// unregister pixel buffer
	if (m_pixelBufferCuda)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(m_pixelBufferCuda));
	}
	
	// free cuda memory
	checkCudaErrors(cudaFree(m_gpuAccumBuffer));
	checkCudaErrors(cudaFree(m_gpuRandState));
	if (m_gpuBVHNodes)
	{
		checkCudaErrors(cudaFree(m_gpuBVHNodes));
	}
	if (m_gpuHittables)
	{
		checkCudaErrors(cudaFree(m_gpuHittables));
	}

	// destroy events
	checkCudaErrors(cudaEventDestroy(m_startEvent));
	checkCudaErrors(cudaEventDestroy(m_stopEvent));

	// free textures
	for (size_t i = 0; i < m_textureCount; ++i)
	{
		checkCudaErrors(cudaDestroyTextureObject(m_textures[i]));
		checkCudaErrors(cudaFreeArray(m_textureMemory[i]));
	}

	checkCudaErrors(cudaFree(m_gpuTextures));

	delete[] m_cpuAccumBuffer;
	delete[] m_cpuResultBuffer;
	delete[] m_textures;
	delete[] m_textureMemory;
}

void Pathtracer::setScene(size_t count, const CpuHittable *hittables)
{
	if (count == 0)
	{
		printf("Setting an empty scene is not allowed!\n");
		return;
	}

	// construct a BVH from the given CpuHittables
	BVH bvh;
	bvh.build(count, hittables, 4);
	assert(bvh.validate());

	// translate hittables to their gpu version
	std::vector<Hittable> gpuHittables;
	{
		auto &bvhElements = bvh.getElements();
		gpuHittables.reserve(bvh.getElements().size());
		for (const auto &e : bvhElements)
		{
			gpuHittables.push_back(e.getGpuHittable());
		}
	}

	// upload to gpu
	{
		// free memory of previous allocations
		if (m_gpuBVHNodes)
		{
			checkCudaErrors(cudaFree(m_gpuBVHNodes));
			m_gpuBVHNodes = nullptr;
		}
		if (m_gpuHittables)
		{
			checkCudaErrors(cudaFree(m_gpuHittables));
			m_gpuHittables = nullptr;
		}

		// allocate memory for nodes and leaves
		checkCudaErrors(cudaMalloc((void **)&m_gpuBVHNodes, bvh.getNodes().size() * sizeof(BVHNode)));
		checkCudaErrors(cudaMalloc((void **)&m_gpuHittables, gpuHittables.size() * sizeof(Hittable)));

		// copy from cpu to gpu memory
		checkCudaErrors(cudaMemcpy(m_gpuBVHNodes, bvh.getNodes().data(), bvh.getNodes().size() * sizeof(BVHNode), cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(m_gpuHittables, gpuHittables.data(), gpuHittables.size() * sizeof(Hittable), cudaMemcpyKind::cudaMemcpyHostToDevice));

		m_nodeCount = (uint32_t)bvh.getNodes().size();
		m_hittableCount = (uint32_t)gpuHittables.size();
	}
}

void Pathtracer::render(const Camera &camera, uint32_t spp, bool ignoreHistory)
{
	if (ignoreHistory)
	{
		m_accumulatedFrames = 0;
	}
	m_timing = 0;

	// start tracing
	checkCudaErrors(cudaEventRecord(m_startEvent));

	// kernels expect a BVH, so skip tracing if there is none
	if (m_nodeCount >= 1 && m_hittableCount >= 1 && spp > 0)
	{
		dim3 threads(8, 8, 1);
		dim3 blocks((m_width + 7) / 8, (m_height + 7) / 8, 1);
		traceKernel << <blocks, threads >> > (
			m_gpuAccumBuffer,
			ignoreHistory,
			m_accumulatedFrames,
			m_width,
			m_height,
			spp,
			m_hittableCount,
			m_gpuHittables,
			m_nodeCount,
			m_gpuBVHNodes,
			m_gpuRandState,
			camera,
			m_skyboxTextureHandle,
			m_gpuTextures);
	}

	// end tracing
	checkCudaErrors(cudaEventRecord(m_stopEvent));
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaEventSynchronize(m_stopEvent));

	checkCudaErrors(cudaEventElapsedTime(&m_timing, m_startEvent, m_stopEvent));


	// copy tonemapped image to opengl pixel buffer if available
	if (m_pixelBufferCuda && m_nodeCount >= 1 && m_hittableCount >= 1 && spp > 0)
	{
		// map pixel buffer
		uchar4 *deviceMem = nullptr;
		size_t numBytes;
		checkCudaErrors(cudaGraphicsMapResources(1, &m_pixelBufferCuda));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&deviceMem, &numBytes, m_pixelBufferCuda));

		// tonemap
		dim3 threads(8, 8, 1);
		dim3 blocks((m_width + 7) / 8, (m_height + 7) / 8, 1);
		tonemap << <blocks, threads >> > (deviceMem, m_gpuAccumBuffer, m_width, m_height, m_accumulatedFrames + 1);

		checkCudaErrors(cudaDeviceSynchronize());

		// unmap pixel buffer
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_pixelBufferCuda));
	}

	++m_accumulatedFrames;
}

float Pathtracer::getTiming() const
{
	return m_timing;
}

uint32_t Pathtracer::loadTexture(const char *path)
{
	// we allocated the maximum number of textures, return a null handle
	if (m_textureCount >= MAX_TEXTURE_COUNT)
	{
		return 0;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	// does the texture contain hdr data?
	const bool hdrTexture = stbi_is_hdr(path);

	// load texture from file
	int width;
	int height;
	int channelCount;
	void *data = hdrTexture ? (void *)stbi_loadf(path, &width, &height, &channelCount, 4) : (void *)stbi_load(path, &width, &height, &channelCount, 4);

	// loading the texture failed, return a null handle
	if (!data)
	{
		return 0;
	}

	// allocate device memory for texture
	int channelSize = hdrTexture ? 32 : 8;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(channelSize, channelSize, channelSize, channelSize, hdrTexture ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaMallocArray(&m_textureMemory[m_textureCount], &channelDesc, width, height));

	// copy texture to device memory
	size_t textureWidthBytes = width * 4 * (hdrTexture ? 4 : 1);
	checkCudaErrors(cudaMemcpy2DToArray(m_textureMemory[m_textureCount], 0, 0, data, textureWidthBytes, textureWidthBytes, height, cudaMemcpyHostToDevice));

	// free cpu memory of texture
	stbi_image_free(data);

	// create texture object
	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = m_textureMemory[m_textureCount];

	cudaTextureDesc texDesc{};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = hdrTexture ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	checkCudaErrors(cudaCreateTextureObject(&m_textures[m_textureCount], &resDesc, &texDesc, nullptr));

	// copy array of textures to gpu memory
	checkCudaErrors(cudaMemcpy(m_gpuTextures, m_textures, MAX_TEXTURE_COUNT * sizeof(cudaTextureObject_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());

	++m_textureCount;

	return m_textureCount;
}

void Pathtracer::setSkyboxTextureHandle(uint32_t handle)
{
	m_skyboxTextureHandle = handle;
}

float *Pathtracer::getHDRImageData()
{
	checkCudaErrors(cudaDeviceSynchronize());

	// copy back to cpu
	checkCudaErrors(cudaMemcpy(m_cpuAccumBuffer, m_gpuAccumBuffer, m_width * m_height * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// divide by number of accumulated samples
	float invSampleCount = 1.0f / fmaxf((float)m_accumulatedFrames, 1.0f);

	for (size_t i = 0; i < (m_width * m_height * 4); ++i)
	{
		m_cpuAccumBuffer[i] *= invSampleCount;
	}

	return m_cpuAccumBuffer;
}

char *Pathtracer::getImageData()
{
	checkCudaErrors(cudaDeviceSynchronize());

	// alloc memory for gpu result buffer
	uchar4 *gpuResultBuffer = nullptr;
	checkCudaErrors(cudaMalloc((void **)&gpuResultBuffer, m_width * m_height * sizeof(uchar4)));

	// tonemap
	dim3 threads(8, 8, 1);
	dim3 blocks((m_width + 7) / 8, (m_height + 7) / 8, 1);
	tonemap << <blocks, threads >> > (gpuResultBuffer, m_gpuAccumBuffer, m_width, m_height, m_accumulatedFrames);

	checkCudaErrors(cudaDeviceSynchronize());

	// copy back to cpu
	checkCudaErrors(cudaMemcpy(m_cpuResultBuffer, gpuResultBuffer, m_width * m_height * sizeof(uchar4), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// free gpu memory
	checkCudaErrors(cudaFree(gpuResultBuffer));

	return m_cpuResultBuffer;
}
