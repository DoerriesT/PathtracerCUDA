#include "Hittable.h"
#include "AABB.h"
#include <limits>

static void worldTransform(const vec3 &position, const vec3 &rotation, const vec3 &scale, float4 *localToWorldRows, float4 *worldToLocalRows)
{
	auto quatToRotMat = [](const float4 &q, float(&rotMat)[3][3])
	{
		float qxx(q.x * q.x);
		float qyy(q.y * q.y);
		float qzz(q.z * q.z);
		float qxz(q.x * q.z);
		float qxy(q.x * q.y);
		float qyz(q.y * q.z);
		float qwx(q.w * q.x);
		float qwy(q.w * q.y);
		float qwz(q.w * q.z);

		rotMat[0][0] = 1.0f - 2.0f * (qyy + qzz);
		rotMat[0][1] = 2.0f * (qxy + qwz);
		rotMat[0][2] = 2.0f * (qxz - qwy);

		rotMat[1][0] = 2.0f * (qxy - qwz);
		rotMat[1][1] = 1.0f - 2.0f * (qxx + qzz);
		rotMat[1][2] = 2.0f * (qyz + qwx);

		rotMat[2][0] = 2.0f * (qxz + qwy);
		rotMat[2][1] = 2.0f * (qyz - qwx);
		rotMat[2][2] = 1.0f - 2.0f * (qxx + qyy);
	};

	// compute quaternion representing the rotation
	float4 q;
	{
		vec3 c = vec3(cosf(rotation.x * 0.5f), cosf(rotation.y * 0.5f), cosf(rotation.z * 0.5f));
		vec3 s = vec3(sinf(rotation.x * 0.5f), sinf(rotation.y * 0.5f), sinf(rotation.z * 0.5f));

		q.w = c.x * c.y * c.z + s.x * s.y * s.z;
		q.x = s.x * c.y * c.z - c.x * s.y * s.z;
		q.y = c.x * s.y * c.z + s.x * c.y * s.z;
		q.z = c.x * c.y * s.z - s.x * s.y * c.z;
	}

	// quaternion representing the inverse rotation
	float4 invQ;
	{
		float invDot = (1.0f / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w));

		// conjugate
		float4 tmp = { -q.x, -q.y, -q.z, q.w };

		invQ.x = tmp.x * invDot;
		invQ.y = tmp.y * invDot;
		invQ.z = tmp.z * invDot;
		invQ.w = tmp.w * invDot;
	}

	// compute invScale * invRot * invTrans
	{
		// compute inverse rotation matrix
		float invRotMat[3][3];
		quatToRotMat(invQ, invRotMat);

		vec3 invScale = 1.0f / scale;
		worldToLocalRows[0].x = invScale.x * invRotMat[0][0];
		worldToLocalRows[0].y = invScale.x * invRotMat[1][0];
		worldToLocalRows[0].z = invScale.x * invRotMat[2][0];
		worldToLocalRows[0].w = invScale.x * dot(vec3(invRotMat[0][0], invRotMat[1][0], invRotMat[2][0]), -position);

		worldToLocalRows[1].x = invScale.y * invRotMat[0][1];
		worldToLocalRows[1].y = invScale.y * invRotMat[1][1];
		worldToLocalRows[1].z = invScale.y * invRotMat[2][1];
		worldToLocalRows[1].w = invScale.y * dot(vec3(invRotMat[0][1], invRotMat[1][1], invRotMat[2][1]), -position);

		worldToLocalRows[2].x = invScale.z * invRotMat[0][2];
		worldToLocalRows[2].y = invScale.z * invRotMat[1][2];
		worldToLocalRows[2].z = invScale.z * invRotMat[2][2];
		worldToLocalRows[2].w = invScale.z * dot(vec3(invRotMat[0][2], invRotMat[1][2], invRotMat[2][2]), -position);
	}

	// compute trans * rot * scale;
	{
		// compute rotation matrix
		float rotMat[3][3];
		quatToRotMat(q, rotMat);

		localToWorldRows[0].x = scale.x * rotMat[0][0];
		localToWorldRows[0].y = scale.y * rotMat[1][0];
		localToWorldRows[0].z = scale.z * rotMat[2][0];
		localToWorldRows[0].w = position.x;

		localToWorldRows[1].x = scale.x * rotMat[0][1];
		localToWorldRows[1].y = scale.y * rotMat[1][1];
		localToWorldRows[1].z = scale.z * rotMat[2][1];
		localToWorldRows[1].w = position.y;

		localToWorldRows[2].x = scale.x * rotMat[0][2];
		localToWorldRows[2].y = scale.y * rotMat[1][2];
		localToWorldRows[2].z = scale.z * rotMat[2][2];
		localToWorldRows[2].w = position.z;
	}
}

CpuHittable::CpuHittable()
	: m_invTransformRow0({ 1.0f, 0.0f, 0.0f, 0.0f }),
	m_invTransformRow1({ 0.0f, 1.0f, 0.0f, 0.0f }),
	m_invTransformRow2({ 0.0f, 0.0f, 1.0f, 0.0f }),
	m_material(),
	m_aabb({ vec3(-1.0f), vec3(1.0f) }),
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
	vec3 adjustedScale = scale;
	if (m_type == HittableType::DISK || m_type == HittableType::QUAD)
	{
		adjustedScale.y = 1.0f;
	}

	// compute transform and its inverse
	float4 worldToLocalRows[3];
	float4 localToWorldRows[3];
	worldTransform(position, rotation, adjustedScale, localToWorldRows, worldToLocalRows);

	m_invTransformRow0 = worldToLocalRows[0];
	m_invTransformRow1 = worldToLocalRows[1];
	m_invTransformRow2 = worldToLocalRows[2];

	// compute aabb
	{
		m_aabb.m_min = vec3(std::numeric_limits<float>::max());
		m_aabb.m_max = vec3(-std::numeric_limits<float>::max());

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
