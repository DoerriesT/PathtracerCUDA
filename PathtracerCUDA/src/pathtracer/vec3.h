#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>

#define PI (3.14159265358979323846f)

struct vec3
{
	union
	{
		struct { float x, y, z; };
		struct { float r, g, b; };
		float e[3];
	};

	__host__ __device__ vec3();
	__host__ __device__ vec3(float e0, float e1, float e2);
	__host__ __device__ vec3(float e);

	__host__ __device__ vec3 operator-() const;
	__host__ __device__ float operator[](int i) const;
	__host__ __device__ float &operator[](int i);
	__host__ __device__ vec3 &operator+=(const vec3 &v);
	__host__ __device__ vec3 &operator*=(const float t);
	__host__ __device__ vec3 &operator*=(const vec3 &t);
	__host__ __device__ vec3 &operator/=(const float t);
};

__host__ __device__ vec3 operator+(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 operator+(const vec3 &u, float v);
__host__ __device__ vec3 operator+(float u, const vec3 &v);
__host__ __device__ vec3 operator-(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 operator-(const vec3 &u, float v);
__host__ __device__ vec3 operator-(const float u, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 operator*(float t, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &v, float t);
__host__ __device__ vec3 operator/(const vec3 &v, const vec3 &t);
__host__ __device__ vec3 operator/(const vec3 &v, float t);
__host__ __device__ vec3 operator/(float v, const vec3 &t);
__host__ __device__ bool operator==(const vec3 &u, const vec3 &v);
__host__ __device__ float dot(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 cross(const vec3 &u, const vec3 &v);
__host__ __device__ float length_squared(const vec3 &v);
__host__ __device__ float length(const vec3 &v);
__host__ __device__ vec3 min(const vec3 &a, const vec3 &b);
__host__ __device__ vec3 max(const vec3 &a, const vec3 &b);
__host__ __device__ vec3 normalize(const vec3 &v);
__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n);
__host__ __device__ vec3 refract(const vec3 &uv, const vec3 &n, float etaiOverEtat);
__host__ __device__ vec3 rotateAroundVector(const vec3 &v, const vec3 &axis, float cosAngle, float sinAngle);
__host__ __device__ vec3 rotateAroundVector(const vec3 &v, const vec3 &axis, float angle);
__host__ __device__ vec3 lerp(const vec3 &x, const vec3 &y, const vec3 &a);
__host__ __device__ vec3 lerp(const vec3 &x, const vec3 &y, float a);
__host__ __device__ float lerp(float x, float y, float a);
//__host__ __device__ vec3 tangentToWorld(const vec3 &N, const vec3 &v);
//__host__ __device__ vec3 cosineSampleHemisphere(float u0, float u1, float &pdf);
__host__ __device__ float clamp(float x, float a = 0.0f, float b = 1.0f);
__host__ __device__ vec3 clamp(const vec3 &x, const vec3 &a, const vec3 &b);
__host__ __device__ vec3 saturate(const vec3 &x);


// IMPLEMENTATION
#include "vec3.inl"