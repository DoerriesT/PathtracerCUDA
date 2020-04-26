#pragma once
#include "vec3.h"

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