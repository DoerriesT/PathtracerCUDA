#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cassert>
#include "Utility.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_gl_interop.h"

#include <stdio.h>

#include "Window.h"

__global__ void traceKernel(uchar4 *result, uint32_t width, uint32_t height)
{
	uint2 threadID;
	threadID.x = threadIdx.x + blockIdx.x * blockDim.x;
	threadID.y = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadID.x >= width || threadID.y >= height)
	{
		return;
	}

	uint32_t dstIdx = threadID.x + threadID.y * width;
	bool white = (threadID.x / 8) & 1 != 0;
	white = (threadID.y / 8) & 1 != 0 ? !white : white;
	unsigned char color = white ? 255 : 0;
	result[dstIdx] = { color , color , color , 255 };
}

int main()
{
	Window window(1600, 900, "Pathtracer CUDA");

	uint32_t width = window.getWidth();
	uint32_t height = window.getHeight();

	GLuint pixelBufferGL = 0;
	cudaGraphicsResource *pixelBufferCuda = nullptr;

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
		cudaError_t result;

		result = cudaSetDevice(0);
		assert(result == cudaSuccess);

		// register with cuda
		result = cudaGraphicsGLRegisterBuffer(&pixelBufferCuda, pixelBufferGL, cudaGraphicsMapFlagsWriteDiscard);
		assert(result == cudaSuccess);
	}

	while (!window.shouldClose())
	{
		window.pollEvents();

		cudaError_t result;
		uchar4 *deviceMem;
		size_t numBytes;
		result = cudaGraphicsMapResources(1, &pixelBufferCuda);
		assert(result == cudaSuccess);
		result = cudaGraphicsResourceGetMappedPointer((void **)&deviceMem, &numBytes, pixelBufferCuda);
		assert(result == cudaSuccess);

		// do something with cuda
		dim3 threads(8, 8, 1);
		dim3 blocks((width + 7) / 8, (height + 7) / 8, 1);
		traceKernel << <blocks, threads >> > (deviceMem, width, height);

		result = cudaGraphicsUnmapResources(1, &pixelBufferCuda);
		assert(result == cudaSuccess);

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

	cudaError_t result;

	// unregister pixel buffer
	result = cudaGraphicsUnregisterResource(pixelBufferCuda);
	assert(result == cudaSuccess);

	// delete pixel buffer object
	glDeleteBuffers(1, &pixelBufferGL);
	assert(glGetError() == GL_NO_ERROR);

	return EXIT_SUCCESS;
}