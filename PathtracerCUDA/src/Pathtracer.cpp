#include "Pathtracer.h"
#include <glad/glad.h>
#include <cassert>
#include "Utility.h"
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <curand_kernel.h>
#include "kernels/initRandState.h"
#include "kernels/trace.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
static void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

Pathtracer::Pathtracer(uint32_t width, uint32_t height)
	:m_width(width),
	m_height(height),
	m_hittableCount(),
	m_nodeCount(),
	m_timing(),
	m_accumulatedFrames()
{
	// init opengl
	{
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			Utility::fatalExit("Failed to initialize GLAD!", EXIT_FAILURE);
		}

		glViewport(0, 0, m_width, m_height);
		assert(glGetError() == GL_NO_ERROR);
		glGenBuffers(1, &m_pixelBufferGL);
		assert(glGetError() == GL_NO_ERROR);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixelBufferGL);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		assert(glGetError() == GL_NO_ERROR);
	}

	// init cuda
	{
		checkCudaErrors(cudaSetDevice(0));
		// register with cuda
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_pixelBufferCuda, m_pixelBufferGL, cudaGraphicsMapFlagsWriteDiscard));

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

		checkCudaErrors(cudaEventCreate(&m_startEvent));
		checkCudaErrors(cudaEventCreate(&m_stopEvent));
	}
}

Pathtracer::~Pathtracer()
{
	checkCudaErrors(cudaDeviceSynchronize());

	// unregister pixel buffer
	checkCudaErrors(cudaGraphicsUnregisterResource(m_pixelBufferCuda));

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

	// delete pixel buffer object
	glDeleteBuffers(1, &m_pixelBufferGL);
	assert(glGetError() == GL_NO_ERROR);
}

void Pathtracer::setBVH(uint32_t nodeCount, const BVHNode *nodes, uint32_t hittableCount, const Hittable *hittables)
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
	checkCudaErrors(cudaMalloc((void **)&m_gpuBVHNodes, nodeCount * sizeof(BVHNode)));
	checkCudaErrors(cudaMalloc((void **)&m_gpuHittables, hittableCount * sizeof(Hittable)));
	
	// copy from cpu to gpu memory
	checkCudaErrors(cudaMemcpy(m_gpuBVHNodes, nodes, nodeCount * sizeof(BVHNode), cudaMemcpyKind::cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_gpuHittables, hittables, hittableCount * sizeof(Hittable), cudaMemcpyKind::cudaMemcpyHostToDevice));
	
	m_nodeCount = nodeCount;
	m_hittableCount = hittableCount;
}

void Pathtracer::render(const Camera &camera, bool ignoreHistory)
{
	if (ignoreHistory)
	{
		m_accumulatedFrames = 0;
	}
	m_timing = 0;

	uchar4 *deviceMem;
	size_t numBytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &m_pixelBufferCuda));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&deviceMem, &numBytes, m_pixelBufferCuda));

	// start tracing
	checkCudaErrors(cudaEventRecord(m_startEvent));

	// kernels expect a BVH, so skip tracing if there is none
	if (m_nodeCount >= 1 && m_hittableCount >= 1)
	{
		dim3 threads(8, 8, 1);
		dim3 blocks((m_width + 7) / 8, (m_height + 7) / 8, 1);
		traceKernel << <blocks, threads >> > (deviceMem, m_gpuAccumBuffer, ignoreHistory, m_accumulatedFrames, m_width, m_height, m_hittableCount, m_gpuHittables, m_nodeCount, m_gpuBVHNodes, m_gpuRandState, camera);
	}

	// end tracing
	checkCudaErrors(cudaEventRecord(m_stopEvent));
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaEventSynchronize(m_stopEvent));
	
	checkCudaErrors(cudaEventElapsedTime(&m_timing, m_startEvent, m_stopEvent));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_pixelBufferCuda));

	// copy to backbuffer
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(-1, -1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixelBufferGL);
	glDrawPixels(m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	assert(glGetError() == GL_NO_ERROR);

	++m_accumulatedFrames;
}

float Pathtracer::getTiming() const
{
	return m_timing;
}
