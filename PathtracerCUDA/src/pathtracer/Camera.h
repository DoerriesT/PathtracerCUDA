#pragma once
#include "Ray.h"

class Camera
{
public:
	__host__ __device__ Camera(const vec3 &position, const vec3 &lookat, const vec3 &up, float fovy, float aspectRatio, float aperture, float focusDist);
	__device__ Ray getRay(float s, float t, curandState &randState);
	__host__ __device__ void rotate(float pitch, float yaw, float roll);
	__host__ __device__ void translate(float x, float y, float z);
	__host__ __device__ void update();

public:
	float m_tanHalfFovy;
	float m_aspectRatio;
	float m_lensRadius;
	float m_focusDist;
	vec3 m_origin;
	vec3 m_lowerLeftCorner;
	vec3 m_horizontal;
	vec3 m_vertical;
	vec3 m_right;
	vec3 m_up;
	vec3 m_backward;
};


// IMPLEMENTATION
#include "Camera.inl"