#pragma once
#include "vec3.h"
#include "Ray.h"

class Material2;

struct HitRecord
{
	vec3 m_p;
	vec3 m_normal;
	const Material2 *m_material;
	float m_t;
	float m_texCoordU;
	float m_texCoordV;
	bool m_frontFace;

	__host__ __device__ inline void setFaceNormal(const Ray &r, const vec3 &outwardNormal)
	{
		m_frontFace = dot(r.direction(), outwardNormal) < 0.0f;
		m_normal = m_frontFace ? outwardNormal : -outwardNormal;
	}
};