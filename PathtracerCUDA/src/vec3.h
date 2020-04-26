#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <curand_kernel.h>

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

	__host__ __device__ vec3 &operator*=(const vec3 &t)
	{
		e[0] *= t.x;
		e[1] *= t.y;
		e[2] *= t.z;
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

__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n) 
{
	return v - 2.0f * dot(v, n) * n;
}

__device__ inline vec3 random_unit_vec(curandState &randState)
{
	auto a = curand_uniform(&randState) * 2.0f * 3.14159265358979323846f;
	auto z = curand_uniform(&randState) * 2.0f - 1.0f;
	auto r = sqrt(1.0f - z * z);
	return vec3(r * cos(a), r * sin(a), z);
}

__device__ vec3 random_in_unit_sphere(curandState &randState) {
	vec3 p;
	do
	{
		p = vec3(curand_uniform(&randState), curand_uniform(&randState), curand_uniform(&randState)) * 2.0f - 1.0f;
	} while (length_squared(p) >= 1.0f);
	return p;
}