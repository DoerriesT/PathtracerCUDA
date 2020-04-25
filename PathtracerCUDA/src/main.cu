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

struct HitRecord
{
	vec3 m_p;
	vec3 m_normal;
	float m_t;
	bool m_frontFace;

	__device__ inline void setFaceNormal(const Ray &r, const vec3 &outwardNormal)
	{
		m_frontFace = dot(r.direction(), outwardNormal) < 0.0f;
		m_normal = m_frontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};

class HittableList : public Hittable
{
public:
	__device__ HittableList() {}
	__device__ HittableList(Hittable **l, int n) : m_list(l), m_listSize(n) {}
	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;

public:
	Hittable **m_list;
	int m_listSize;
};

__device__ bool HittableList::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
	HitRecord tempRec;
	bool hitAnything = false;
	auto closest = t_max;

	for (int i = 0; i < m_listSize; ++i)
	{
		if (m_list[i]->hit(r, t_min, closest, tempRec))
		{
			hitAnything = true;
			closest = tempRec.m_t;
			rec = tempRec;
		}
	}

	return hitAnything;
}

class Sphere : public Hittable
{
public:
	__device__ Sphere() {}
	__device__ Sphere(vec3 center, float radius) : m_center(center), m_radius(radius) {};

	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;

public:
	vec3 m_center;
	float m_radius;
};

__device__ bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
	vec3 oc = r.origin() - m_center;
	auto a = length_squared(r.direction());
	auto half_b = dot(oc, r.direction());
	auto c = length_squared(oc) - m_radius * m_radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant > 0.0f)
	{
		auto root = sqrt(discriminant);
		auto temp = (-half_b - root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.m_t = temp;
			rec.m_p = r.at(rec.m_t);
			vec3 outwardNormal = (rec.m_p - m_center) / m_radius;
			rec.setFaceNormal(r, outwardNormal);
			return true;
		}
		temp = (-half_b + root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.m_t = temp;
			rec.m_p = r.at(rec.m_t);
			vec3 outwardNormal = (rec.m_p - m_center) / m_radius;
			rec.setFaceNormal(r, outwardNormal);
			return true;
		}
	}

	return false;
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
	vec3 dir(u * 2.0f - 1.0f, v * 2.0f - 1.0f, -1.0f);
	dir.x *= (float)width / height;
	Ray r(vec3(0.0f, 0.0f, 0.0f), dir);
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


