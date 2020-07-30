#pragma once
#include "Camera.h"

__host__ __device__ inline Camera::Camera(
	const vec3 &position,
	const vec3 &lookat,
	const vec3 &up,
	float fovy,
	float aspectRatio,
	float aperture,
	float focusDist)
	:m_tanHalfFovy(tan(fovy * 0.5f)),
	m_aspectRatio(aspectRatio),
	m_lensRadius(aperture * 0.5f),
	m_focusDist(focusDist),
	m_origin(position),
	m_lowerLeftCorner(-1.0f, -1.0f, -1.0f),
	m_horizontal(2.0f, 0.0f, 0.0f),
	m_vertical(0.0f, 2.0f, 0.0f)
{
	m_backward = normalize(m_origin - lookat);
	m_right = normalize(cross(up, m_backward));
	m_up = cross(m_backward, m_right);

	update();
}

__device__ inline Ray Camera::getRay(float s, float t, curandState &randState)
{
	vec3 rd = m_lensRadius * random_in_unit_disk(randState);
	vec3 offset = m_right * rd.x + m_up * rd.y;
	return Ray(m_origin + offset, normalize(m_lowerLeftCorner + s * m_horizontal + t * m_vertical - m_origin - offset));
}

__host__ __device__ inline void Camera::rotate(float pitch, float yaw, float roll)
{
	// rotate around local x axis
	const float cosPitch = cos(-pitch);
	const float sinPitch = sin(-pitch);
	m_up = rotateAroundVector(m_up, m_right, cosPitch, sinPitch);
	m_backward = rotateAroundVector(m_backward, m_right, cosPitch, sinPitch);

	// rotate around up axis
	const float cosYaw = cos(-yaw);
	const float sinYaw = sin(-yaw);
	m_right = rotateAroundVector(m_right, vec3(0.0f, 1.0f, 0.0f), cosYaw, sinYaw);
	m_up = rotateAroundVector(m_up, vec3(0.0f, 1.0f, 0.0f), cosYaw, sinYaw);
	m_backward = rotateAroundVector(m_backward, vec3(0.0f, 1.0f, 0.0f), cosYaw, sinYaw);

	update();
}

__host__ __device__ inline void Camera::translate(float x, float y, float z)
{
	m_origin += x * m_right + y * m_up + z * m_backward;
	update();
}

__host__ __device__ inline void Camera::update()
{
	float halfHeight = m_tanHalfFovy;
	float halfWidth = m_aspectRatio * halfHeight;

	m_lowerLeftCorner = m_origin
		- halfWidth * m_right * m_focusDist
		- halfHeight * m_up * m_focusDist
		- m_backward * m_focusDist;
	m_horizontal = 2.0f * halfWidth * m_right * m_focusDist;
	m_vertical = 2.0f * halfHeight * m_up * m_focusDist;
}
