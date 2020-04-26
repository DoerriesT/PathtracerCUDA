#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cassert>
#include "Utility.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_gl_interop.h"
#include <curand_kernel.h>

#include <stdio.h>

#include "Window.h"
#include <iostream>

#include "Ray.h"
#include "vec3.h"
#include "Hittable.h"
#include "Camera.h"
#include "Material.h"

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

__device__ vec3 random_in_hemisphere(const vec3 &normal, curandState &randState)
{
	vec3 in_unit_sphere = normalize(random_unit_vec(randState));
	return dot(in_unit_sphere, normal) > 0.0 ? in_unit_sphere : -in_unit_sphere;
}

__host__ __device__ float clamp(float x, float a, float b)
{
	x = x < a ? a : x;
	x = x > b ? b : x;
	return x;
}

__host__ __device__ vec3 clamp(vec3 x, vec3 a, vec3 b)
{
	return vec3(clamp(x.x, 0.0f, 1.0f), clamp(x.y, 0.0f, 1.0f), clamp(x.z, 0.0f, 1.0f));
}

__host__ __device__ vec3 saturate(vec3 x)
{
	return clamp(x, vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f));
}

__device__ vec3 getColor(const Ray &r, Hittable **world, curandState &randState)
{
	vec3 att = vec3(1.0f, 1.0f, 1.0f);
	Ray ray = r;
	for (int iteration = 0; iteration < 5; ++iteration)
	{
		HitRecord rec;
		if ((*world)->hit(ray, 0.001f, FLT_MAX, rec))
		{
			Ray scattered;
			vec3 attenuation;
			if (rec.m_material->scatter(ray, rec, randState, attenuation, scattered))
			{
				att *= attenuation;
				ray = scattered;
			}
			else
			{
				return vec3(0.0f, 0.0f, 0.0f);
			}
		}
		else
		{
			vec3 unitDir = normalize(ray.m_dir);
			float t = unitDir.y * 0.5f + 0.5f;
			vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
			return c * att;
		}
	}

	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void traceKernel(uchar4 *resultBuffer, float4 *accumBuffer, bool ignoreHistory, uint32_t frame, uint32_t width, uint32_t height, Hittable **world, curandState *randState)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	const uint32_t dstIdx = threadIDx + threadIDy * width;

	float4 inputColor4 = accumBuffer[dstIdx];
	vec3 inputColor(inputColor4.x, inputColor4.y, inputColor4.z);

	curandState &localRandState = randState[dstIdx];

	float u = (threadIDx + curand_uniform(&localRandState)) / float(width);
	float v = (threadIDy + curand_uniform(&localRandState)) / float(height);
	Camera cam((float)width / height);
	Ray r = cam.getRay(u, v);
	vec3 color = getColor(r, world, localRandState);

	color += inputColor;

	vec3 resultColor = color / float(frame + 1.0f);
	resultColor = saturate(resultColor);
	resultColor.r = sqrt(resultColor.r);
	resultColor.g = sqrt(resultColor.g);
	resultColor.b = sqrt(resultColor.b);

	accumBuffer[dstIdx] = { color.r, color.g, color.b, 1.0f };
	resultBuffer[dstIdx] = { (unsigned char)(resultColor.x * 255.0f), (unsigned char)(resultColor.y * 255.0f) , (unsigned char)(resultColor.z * 255.0f), 255 };
}

__global__ void createWorld(Hittable **d_list, Hittable **d_world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		d_list[0] = new Sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f,
			new Lambertian(vec3(0.7f, 0.3f, 0.3f)));
		d_list[1] = new Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f,
			new Lambertian(vec3(0.8f, 0.8f, 0.0f)));

		d_list[2] = new Sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f,
			new Metal(vec3(0.8f, 0.6f, 0.2f), 1.0f));

		d_list[3] = new Sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f,
			new Metal(vec3(0.8f, 0.8f, 0.8f), 0.3f));

		*d_world = new HittableList(d_list, 4);
	}
}

__global__ void freeWorld(Hittable **d_list, Hittable **d_world)
{
	delete d_list[0];
	delete d_list[1];
	delete *d_world;
}

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

int main()
{
	Window window(1600, 900, "Pathtracer CUDA");

	uint32_t width = window.getWidth();
	uint32_t height = window.getHeight();

	GLuint pixelBufferGL = 0;
	cudaGraphicsResource *pixelBufferCuda = nullptr;
	float4 *accumBuffer = nullptr;
	Hittable **d_list;
	Hittable **d_world;
	curandState *d_randState;


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
		checkCudaErrors(cudaMalloc((void **)&accumBuffer, width * height * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&d_list, 4 * sizeof(Hittable *)));
		checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));
		createWorld << <1, 1 >> > (d_list, d_world);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMalloc((void **)&d_randState, width * height * sizeof(curandState)));

		dim3 threads(8, 8, 1);
		dim3 blocks((width + 7) / 8, (height + 7) / 8, 1);
		initRandState << <blocks, threads >> > (width, height, d_randState);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	uint32_t frame = 0;
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
		traceKernel << <blocks, threads >> > (deviceMem, accumBuffer, frame == 0, frame, width, height, d_world, d_randState);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

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
		window.setTitle(std::to_string(frame));
		++frame;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	// unregister pixel buffer
	checkCudaErrors(cudaGraphicsUnregisterResource(pixelBufferCuda));

	// free cuda memory
	freeWorld << <1, 1 >> > (d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_randState));

	// delete pixel buffer object
	glDeleteBuffers(1, &pixelBufferGL);
	assert(glGetError() == GL_NO_ERROR);

	return EXIT_SUCCESS;
}


