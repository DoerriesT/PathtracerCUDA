#pragma once
#include "vec3.h"

__host__ __device__ inline vec3::vec3()
	: e{ 0,0,0 }
{
}

__host__ __device__ inline vec3::vec3(float e0, float e1, float e2)
	: e{ e0, e1, e2 }
{
}

__host__ __device__ inline vec3::vec3(float e)
	: e{ e, e, e }
{
}

__host__ __device__ inline vec3 vec3::operator-() const
{
	return vec3(-e[0], -e[1], -e[2]);
}

__host__ __device__ inline float vec3::operator[](int i) const
{
	return e[i];
}

__host__ __device__ inline float &vec3::operator[](int i)
{
	return e[i];
}

__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &t)
{
	e[0] *= t.x;
	e[1] *= t.y;
	e[2] *= t.z;
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const float t)
{
	return *this *= 1 / t;
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

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
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

__host__ __device__ inline float length_squared(const vec3 &v)
{
	return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2];
}

__host__ __device__ inline float length(const vec3 &v)
{
	return sqrtf(length_squared(v));
}

__host__ __device__ inline vec3 min(const vec3 &a, const vec3 &b)
{
	vec3 x = a;
	x.x = x.x < b.x ? x.x : b.x;
	x.y = x.y < b.y ? x.y : b.y;
	x.z = x.z < b.z ? x.z : b.z;
	return x;
}

__host__ __device__ inline vec3 max(const vec3 &a, const vec3 &b)
{
	vec3 x = a;
	x.x = x.x >= b.x ? x.x : b.x;
	x.y = x.y >= b.y ? x.y : b.y;
	x.z = x.z >= b.z ? x.z : b.z;
	return x;
}

__host__ __device__ inline vec3 normalize(const vec3 &v)
{
	return v / length(v);
}

__host__ __device__ inline vec3 reflect(const vec3 &v, const vec3 &n)
{
	return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3 &uv, const vec3 &n, float etaiOverEtat)
{
	auto cos_theta = dot(-uv, n);
	vec3 rOutParallel = etaiOverEtat * (uv + cos_theta * n);
	vec3 rOutPerp = -sqrtf(1.0f - length_squared(rOutParallel)) * n;
	return rOutParallel + rOutPerp;
}

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
__host__ __device__ inline vec3 rotateAroundVector(const vec3 &v, const vec3 &axis, float cosAngle, float sinAngle)
{
	return v * cosAngle + cross(axis, v) * sinAngle + axis * dot(axis, v) * (1.0f - cosAngle);
}

__host__ __device__ inline vec3 rotateAroundVector(const vec3 &v, const vec3 &axis, float angle)
{
	const float cosAngle = cosf(angle);
	const float sinAngle = sinf(angle);
	return rotateAroundVector(v, axis, cosAngle, sinAngle);
}

__host__ __device__ inline vec3 lerp(const vec3 &x, const vec3 &y, const vec3 &a)
{
	return x * (1.0f - a) + y * a;
}

__host__ __device__ inline vec3 lerp(const vec3 &x, const vec3 &y, float a)
{
	return x * (1.0f - a) + y * a;
}

__host__ __device__ inline float lerp(float x, float y, float a)
{
	return x * (1.0f - a) + y * a;
}

__host__ __device__ inline float clamp(float x, float a, float b)
{
	x = x < a ? a : x;
	x = x > b ? b : x;
	return x;
}

__host__ __device__ inline vec3 clamp(const vec3 &x, const vec3 &a, const vec3 &b)
{
	return vec3(clamp(x.x, 0.0f, 1.0f), clamp(x.y, 0.0f, 1.0f), clamp(x.z, 0.0f, 1.0f));
}

__host__ __device__ inline vec3 saturate(const vec3 &x)
{
	return clamp(x, vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f));
}