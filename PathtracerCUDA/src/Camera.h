#pragma once
#include "Ray.h"

class Camera
{
public:
	__host__ __device__ Camera(const vec3 &position, const vec3 &lookat, const vec3 &up, float fovy, float aspectRatio)
		:m_aspectRatio(aspectRatio),
		m_origin(position),
		m_lowerLeftCorner(-1.0f, -1.0f, -1.0f),
		m_horizontal(2.0f, 0.0f, 0.0f),
		m_vertical(0.0f, 2.0f, 0.0f)
	{
		auto halfHeight = tan(fovy * 0.5f);
		auto halfWidth = m_aspectRatio * halfHeight;
		vec3 w = normalize(m_origin - lookat);
		vec3 u = normalize(cross(up, w));
		vec3 v = cross(w, u);

		m_lowerLeftCorner = m_origin - halfWidth * u - halfHeight * v - w;
		m_horizontal = 2.0f * halfWidth * u;
		m_vertical = 2.0f * halfHeight * v;
	}

	__device__ Ray getRay(float u, float v)
	{
		vec3 dir(m_lowerLeftCorner + u * m_horizontal + v * m_vertical - m_origin);
		return Ray(m_origin, dir);
	}

public:
	float m_aspectRatio;
	vec3 m_origin;
	vec3 m_lowerLeftCorner;
	vec3 m_horizontal;
	vec3 m_vertical;
};