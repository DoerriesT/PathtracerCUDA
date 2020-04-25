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
struct vec3
{
	union
	{
		struct { float x, y, z; };
		struct { float r, g, b; };
		float e[3];
	};

	__host__ __device__ vec3() : e{ 0,0,0 } {}
	__host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

	__host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ float operator[](int i) const { return e[i]; }
	__host__ __device__ float &operator[](int i) { return e[i]; }

	__host__ __device__ vec3 &operator+=(const vec3 &v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__host__ __device__ vec3 &operator*=(const float t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	__host__ __device__ vec3 &operator/=(const float t)
	{
		return *this *= 1 / t;
	}
};

inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3 &u, float v)
{
	return vec3(u.e[0] + v, u.e[1] + v, u.e[2] + v);
}

__host__ __device__ inline vec3 operator+(float u, const vec3 &v)
{
	return v + u;
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, float v)
{
	return vec3(u.e[0] - v, u.e[1] - v, u.e[2] - v);
}

__host__ __device__ inline vec3 operator-(const float u, const vec3 &v)
{
	return -v + u;
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
	return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, const vec3 &t)
{
	return vec3(1.0f / t.x, 1.0f / t.y, 1.0f / t.z) * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t)
{
	return (1.0f / t) * v;
}

__host__ __device__ inline vec3 operator/(float v, const vec3 &t)
{
	return vec3(1.0f / t.x, 1.0f / t.y, 1.0f / t.z) * v;
}

__host__ __device__ inline bool operator==(const vec3 &u, const vec3 &v)
{
	return u.x == v.x && u.y == v.y && u.z == v.z;
}

__host__ __device__ inline double dot(const vec3 &u, const vec3 &v)
{
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ float length_squared(vec3 v)
{
	return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2];
}

__host__ __device__ float length(vec3 v)
{
	return sqrt(length_squared(v));
}

__host__ __device__ inline vec3 normalize(vec3 v)
{
	return v / length(v);
}

struct Ray
{
	vec3 m_origin;
	vec3 m_dir;

	__device__ Ray() {}
	__device__ Ray(const vec3 &origin, const vec3 &direction)
		:m_origin(origin),
		m_dir(direction)
	{
	};

	__device__ vec3 origin() const { return m_origin; };
	__device__ vec3 direction() const { return m_dir; };

	__device__ vec3 at(float t) const
	{
		return m_origin + t * m_dir;
	}
};

__device__ float hitSphere(const vec3 &center, float radius, const Ray &r)
{
	vec3 oc = r.m_origin - center;
	auto a = length_squared(r.m_dir);
	auto half_b = dot(oc, r.m_dir);
	auto c = length_squared(oc) - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0.0f)
	{
		return -1.0f;
	}
	else
	{
		return (-half_b - sqrt(discriminant)) / a;
	}
}

__device__ vec3 getColor(const Ray &r)
{
	auto t = hitSphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, r);
	if (t > 0.0f)
	{
		vec3 N = normalize(r.at(t) - vec3(0.0f, 0.0f, -1.0f));
		return N * 0.5f + 0.5f;
	}
	vec3 unitDir = normalize(r.m_dir);
	t = 0.5f * (unitDir.y + 1.0f);
	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void traceKernel(uchar4 *resultBuffer, uint32_t width, uint32_t height)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	float u = threadIDx / float(width);
	float v = threadIDy / float(height);
	vec3 dir(u * 2.0f - 1.0f, v * 2.0f - 1.0f, -1.0f);
	dir.x *= (float)width / height;
	Ray r(vec3(0.0f, 0.0f, 0.0f), dir);
	auto color = getColor(r);

	uint32_t dstIdx = threadIDx + threadIDy * width;
	resultBuffer[dstIdx] = { (unsigned char)(color.x * 255.0f), (unsigned char)(color.y * 255.0f) , (unsigned char)(color.z * 255.0f), 255 };

	//
	//bool white = (threadID.x / 8) & 1 != 0;
	//white = (threadID.y / 8) & 1 != 0 ? !white : white;
	//unsigned char color = white ? 255 : 0;
	//result[dstIdx] = { color , color , color , 255 };
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
		checkCudaErrors(cudaSetDevice(0));
		// register with cuda
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&pixelBufferCuda, pixelBufferGL, cudaGraphicsMapFlagsWriteDiscard));
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
		traceKernel << <blocks, threads >> > (deviceMem, width, height);
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

	// unregister pixel buffer
	checkCudaErrors(cudaGraphicsUnregisterResource(pixelBufferCuda));

	// delete pixel buffer object
	glDeleteBuffers(1, &pixelBufferGL);
	assert(glGetError() == GL_NO_ERROR);

	return EXIT_SUCCESS;
}