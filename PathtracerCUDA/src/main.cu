#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cassert>
#include "Utility.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_gl_interop.h"

#include <stdio.h>

#include "Window.h"
#include <iostream>

#include "Ray.h"
#include "vec3.h"
#include "Hittable.h"
#include "Camera.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
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

__device__ vec3 getColor(const Ray &r, Hittable **world)
{
	HitRecord rec;
	if ((*world)->hit(r, 0.0f, FLT_MAX, rec))
	{
		return rec.m_normal * 0.5f + 0.5f;
	}
	vec3 unitDir = normalize(r.m_dir);
	float t = unitDir.y * 0.5f + 0.5f;
	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void traceKernel(uchar4 *resultBuffer, uint32_t width, uint32_t height, Hittable **world)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	float u = threadIDx / float(width);
	float v = threadIDy / float(height);
	Camera cam((float)width / height);
	Ray r = cam.getRay(u, v);
	auto color = getColor(r, world);

	uint32_t dstIdx = threadIDx + threadIDy * width;
	resultBuffer[dstIdx] = { (unsigned char)(color.x * 255.0f), (unsigned char)(color.y * 255.0f) , (unsigned char)(color.z * 255.0f), 255 };

	//
	//bool white = (threadID.x / 8) & 1 != 0;
	//white = (threadID.y / 8) & 1 != 0 ? !white : white;
	//unsigned char color = white ? 255 : 0;
	//result[dstIdx] = { color , color , color , 255 };
}

__global__ void createWorld(Hittable **d_list, Hittable **d_world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		d_list[0] = new Sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
		d_list[1] = new Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
		*d_world = new HittableList(d_list, 2);
	}
}

__global__ void freeWorld(Hittable **d_list, Hittable **d_world)
{
	delete d_list[0];
	delete d_list[1];
	delete *d_world;
}

int main()
{
	Window window(1600, 900, "Pathtracer CUDA");

	uint32_t width = window.getWidth();
	uint32_t height = window.getHeight();

	GLuint pixelBufferGL = 0;
	cudaGraphicsResource *pixelBufferCuda = nullptr;
	Hittable **d_list;
	Hittable **d_world;
	

	// init opengl
	{
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			Utility::fatalExit("Failed to initialize GLAD!", EXIT_FAILURE);
		}

		glViewport(0, 0, width, height);
		assert(glGetError() == GL_NO_ERROR);
		glGenBuffers(1, &pixelBufferGL);
		assert(glGetError() == GL_NO_ERROR);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferGL);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		assert(glGetError() == GL_NO_ERROR);
	}

	// init cuda
	{
		checkCudaErrors(cudaSetDevice(0));
		// register with cuda
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&pixelBufferCuda, pixelBufferGL, cudaGraphicsMapFlagsWriteDiscard));
	
		checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(Hittable *)));
		checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));
		createWorld << <1, 1 >> > (d_list, d_world);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	while (!window.shouldClose())
	{
		window.pollEvents();

		uchar4 *deviceMem;
		size_t numBytes;
		checkCudaErrors(cudaGraphicsMapResources(1, &pixelBufferCuda));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&deviceMem, &numBytes, pixelBufferCuda));

		// do something with cuda
		dim3 threads(8, 8, 1);
		dim3 blocks((width + 7) / 8, (height + 7) / 8, 1);
		traceKernel << <blocks, threads >> > (deviceMem, width, height, d_world);
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBufferCuda));

		// render
		glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glRasterPos2i(-1, -1);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferGL);
		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		assert(glGetError() == GL_NO_ERROR);

		window.present();
	}

	checkCudaErrors(cudaDeviceSynchronize());

	// unregister pixel buffer
	checkCudaErrors(cudaGraphicsUnregisterResource(pixelBufferCuda));

	// free cuda memory
	freeWorld << <1, 1 >> > (d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));

	// delete pixel buffer object
	glDeleteBuffers(1, &pixelBufferGL);
	assert(glGetError() == GL_NO_ERROR);

	return EXIT_SUCCESS;
}


