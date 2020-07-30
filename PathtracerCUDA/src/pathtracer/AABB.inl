#pragma once
#include "AABB.h"

__host__ __device__  inline AABB::AABB()
{

}

__host__ __device__ inline AABB::AABB(const vec3 &a, const vec3 &b)
	:m_min(a),
	m_max(b)
{

}

__host__ __device__ inline AABB::AABB(const AABB &a, const AABB &b)
	: m_min(min(a.m_min, b.m_min)),
	m_max(max(a.m_max, b.m_max))
{
}

__host__ __device__ inline bool AABB::hit(const Ray &r, float tMin, float tMax) const
{
	for (int a = 0; a < 3; ++a)
	{
		float invD = 1.0f / r.m_dir[a];
		float t0 = (m_min[a] - r.m_origin[a]) * invD;
		float t1 = (m_max[a] - r.m_origin[a]) * invD;
		if (invD < 0.0f)
		{
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = t0 > tMin ? t0 : tMin;
		tMax = t1 < tMax ? t1 : tMax;

		if (tMax <= tMin)
		{
			return false;
		}
	}
	return true;
}

__host__ __device__ inline bool AABB::intersect(const Ray &r, float tMin, float tMax, float &t) const
{
	for (int a = 0; a < 3; ++a)
	{
		float invD = 1.0f / r.m_dir[a];
		float t0 = (m_min[a] - r.m_origin[a]) * invD;
		float t1 = (m_max[a] - r.m_origin[a]) * invD;
		if (invD < 0.0f)
		{
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = t0 > tMin ? t0 : tMin;
		tMax = t1 < tMax ? t1 : tMax;

		if (tMax <= tMin)
		{
			return false;
		}
	}
	t = tMin;
	return true;
}