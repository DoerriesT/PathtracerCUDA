#pragma once
#include "cuda_runtime.h"
#include "vec3.h"
#include "Ray.h"

struct Ray;

class AABB
{
public:
	vec3 m_min;
	vec3 m_max;

	__host__ __device__ AABB();
	__host__ __device__ AABB(const vec3 &a, const vec3 &b);
	__host__ __device__ AABB(const AABB &a, const AABB &b);
	__host__ __device__ bool hit(const Ray &r, float tMin, float tMax) const;
	__host__ __device__ bool intersect(const Ray &r, float tMin, float tMax, float &t) const;
};


// IMPLEMENTATION
#include "AABB.inl"