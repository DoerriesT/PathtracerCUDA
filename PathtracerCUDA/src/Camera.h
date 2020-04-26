#pragma once
#include "Ray.h"

class Camera
{
public:
	__host__ __device__ Camera(
		const vec3 &position, 
		const vec3 &lookat, 
		const vec3 &up, 
		float fovy, 
		float aspectRatio,
		float aperture,
		float focusDist)
		:m_lensRadius(aperture * 0.5f),
		m_origin(position),
		m_lowerLeftCorner(-1.0f, -1.0f, -1.0f),
		m_horizontal(2.0f, 0.0f, 0.0f),
		m_vertical(0.0f, 2.0f, 0.0f)
	{
		auto halfHeight = tan(fovy * 0.5f);
		auto halfWidth = aspectRatio * halfHeight;
		m_w = normalize(m_origin - lookat);
		m_u = normalize(cross(up, m_w));
		m_v = cross(m_w, m_u);

		m_lowerLeftCorner = m_origin 
			- halfWidth * m_u * focusDist
			- halfHeight * m_v * focusDist
			- m_w * focusDist;
		m_horizontal = 2.0f * halfWidth * m_u * focusDist;
		m_vertical = 2.0f * halfHeight * m_v * focusDist;
	}

	__device__ Ray getRay(float s, float t, curandState &randState)
	{
		vec3 rd = m_lensRadius * random_in_unit_disk(randState);
		vec3 offset = m_u *rd.x + m_v * rd.y;
		return Ray(m_origin + offset, m_lowerLeftCorner + s * m_horizontal + t * m_vertical - m_origin - offset);
	}

public:
	float m_lensRadius;
	vec3 m_origin;
	vec3 m_lowerLeftCorner;
	vec3 m_horizontal;
	vec3 m_vertical;
	vec3 m_u;
	vec3 m_v;
	vec3 m_w;
};