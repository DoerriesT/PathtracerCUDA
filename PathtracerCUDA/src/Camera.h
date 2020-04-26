#pragma once
#include "Ray.h"

class Camera
{
public:
	__device__ Camera(float aspectRatio)
		:m_aspectRatio(aspectRatio),
		m_origin(0.0f, 0.0f, 0.0f),
		m_lowerLeftCorner(-1.0f, -1.0f, -1.0f),
		m_horizontal(2.0f, 0.0f, 0.0f),
		m_vertical(0.0f, 2.0f, 0.0f)
	{}

	__device__ Ray getRay(float u, float v)
	{
		vec3 dir(m_lowerLeftCorner + u * m_horizontal + v * m_vertical - m_origin);
		dir.x *= m_aspectRatio;
		return Ray(m_origin, dir);
	}

public:
	float m_aspectRatio;
	vec3 m_origin;
	vec3 m_lowerLeftCorner;
	vec3 m_horizontal;
	vec3 m_vertical;
};