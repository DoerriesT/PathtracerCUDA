#include "Hittable.h"
#include "AABB.h"

CpuHittable::CpuHittable()
	: m_invTransformRow0({ 1.0f, 0.0f, 0.0f, 0.0f }),
	m_invTransformRow1({ 0.0f, 1.0f, 0.0f, 0.0f }),
	m_invTransformRow2({ 0.0f, 0.0f, 1.0f, 0.0f }),
	m_material(),
	m_aabb({vec3(-1.0f), vec3(1.0f)}),
	m_type(HittableType::SPHERE)
{
}

CpuHittable::CpuHittable(HittableType type, const vec3 &position, const vec3 &rotation, const vec3 &scale, const Material2 &material)
	: m_invTransformRow0({ 1.0f, 0.0f, 0.0f, 0.0f }),
	m_invTransformRow1({ 0.0f, 1.0f, 0.0f, 0.0f }),
	m_invTransformRow2({ 0.0f, 0.0f, 1.0f, 0.0f }),
	m_material(material),
	m_aabb({ vec3(-1.0f), vec3(1.0f) }),
	m_type(type)
{
	// compute transform and its inverse
	float4 worldToLocalRows[3];
	float4 localToWorldRows[3];
	worldTransform(position, rotation, scale, localToWorldRows, worldToLocalRows);

	m_invTransformRow0 = worldToLocalRows[0];
	m_invTransformRow1 = worldToLocalRows[1];
	m_invTransformRow2 = worldToLocalRows[2];

	// compute aabb
	{
		m_aabb.m_min = vec3(FLT_MAX);
		m_aabb.m_max = vec3(-FLT_MAX);

		float xExtend[2]{ -1.0f, 1.0f };
		float yExtend[2]{ -1.0f, 1.0f };
		float zExtend[2]{ -1.0f, 1.0f };

		if (m_type == HittableType::DISK || m_type == HittableType::QUAD)
		{
			yExtend[0] = -0.01f;
			yExtend[1] = 0.01f;
		}
		else if (m_type == HittableType::PARABOLOID)
		{
			yExtend[0] = 0.0f;
		}

		for (int z = 0; z < 2; ++z)
		{
			for (int y = 0; y < 2; ++y)
			{
				for (int x = 0; x < 2; ++x)
				{
					vec3 pos;
					pos.x = dot(vec3(xExtend[x], yExtend[y], zExtend[z]), vec3(localToWorldRows[0].x, localToWorldRows[0].y, localToWorldRows[0].z)) + localToWorldRows[0].w;
					pos.y = dot(vec3(xExtend[x], yExtend[y], zExtend[z]), vec3(localToWorldRows[1].x, localToWorldRows[1].y, localToWorldRows[1].z)) + localToWorldRows[1].w;
					pos.z = dot(vec3(xExtend[x], yExtend[y], zExtend[z]), vec3(localToWorldRows[2].x, localToWorldRows[2].y, localToWorldRows[2].z)) + localToWorldRows[2].w;

					m_aabb.m_min = min(m_aabb.m_min, pos);
					m_aabb.m_max = max(m_aabb.m_max, pos);
				}
			}
		}
	}
}

const AABB &CpuHittable::getAABB() const
{
	return m_aabb;
}

Hittable CpuHittable::getGpuHittable() const
{
	float4 invTransformRows[]{ m_invTransformRow0, m_invTransformRow1, m_invTransformRow2 };
	return Hittable(m_type, invTransformRows, m_material);
}
