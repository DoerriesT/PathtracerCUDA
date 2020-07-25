#pragma once
#include "Hittable.h"
#include "HitRecord.h"
#include "AABB.h"

__host__ __device__ inline bool quadratic(float a, float b, float c, float &t0, float &t1)
{
	// find discriminant
	const float discriminant = b * b - 4.0f * a * c;
	if (discriminant < 0.0f)
	{
		return false;
	}
	const float discriminantRoot = sqrtf(discriminant);

	// compute t values
	const float q = b < 0.0f ? -0.5f * (b - discriminantRoot) : -0.5f * (b + discriminantRoot);
	t0 = q / a;
	t1 = c / q;
	if (t0 > t1)
	{
		const float tmp = t0;
		t0 = t1;
		t1 = tmp;
	}
	return true;
}

// http://skuld.bmsc.washington.edu/people/merritt/graphics/quadrics.html
template<int A = 0, int B = 0, int C = 0, int D = 0, int E = 0, int F = 0, int G = 0, int H = 0, int I = 0, int J = 0>
__host__ __device__ inline bool hitQuadric(const vec3 &o, const vec3 &d, float &t0, float &t1)
{
	float a = (A * d.x * d.x) + (B * d.y * d.y) + (C * d.z * d.z) + (D * d.x * d.y)
		+ (E * d.x * d.z) + (F * d.y * d.z);

	float b = (2.0f * A * o.x * d.x) + (2.0f * B * o.y * d.y) + (2.0f * C * o.z * d.z) + (D * (o.x * d.y + o.y * d.x))
		+ (E * (o.x * d.z + o.z * d.x)) + (F * (o.y * d.z + d.y * o.z)) + (G * d.x) + (H * d.y) + (I * d.z);

	float c = (A * o.x * o.x) + (B * o.y * o.y) + (C * o.z * o.z) + (D * o.x * o.y)
		+ (E * o.x * o.z) + (F * o.y * o.z) + (G * o.x) + (H * o.y) + (I * o.z) + J;

	return quadratic(a, b, c, t0, t1);
}

// http://skuld.bmsc.washington.edu/people/merritt/graphics/quadrics.html
template<int A = 0, int B = 0, int C = 0, int D = 0, int E = 0, int F = 0, int G = 0, int H = 0, int I = 0, int J = 0>
__host__ __device__ inline vec3 quadricNormal(const vec3 &p)
{
	vec3 normal;
	normal.x = 2.0f * (A * p.x) + (D * p.y) + (E * p.z) + G;
	normal.y = 2.0f * (B * p.y) + (D * p.x) + (F * p.z) + H;
	normal.z = 2.0f * (C * p.z) + (E * p.x) + (F * p.y) + I;

	return normal;
}

__host__ __device__ inline Hittable::Hittable()
	:m_type(SPHERE),
	m_material(),
	m_payload(Sphere{ vec3(0.0f), 1.0f })
{
	m_invTransformRow0 = { 1.0f, 0.0f, 0.0f, 0.0f };
	m_invTransformRow1 = { 0.0f, 1.0f, 0.0f, 0.0f };
	m_invTransformRow2 = { 0.0f, 0.0f, 1.0f, 0.0f };

	m_aabb = { vec3(-1.0f), vec3(1.0f) };
}

__host__ __device__ inline Hittable::Hittable(Type type, const vec3 &position, const vec3 &rotation, const vec3 &scale, const Payload &payload, const Material2 &material)
	: m_type(type),
	m_material(material),
	m_payload(payload)
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

		for (int z = -1; z < 2; z += 2)
		{
			for (int y = -1; y < 2; y += 2)
			{
				for (int x = -1; x < 2; x += 2)
				{
					vec3 pos;
					pos.x = dot(vec3((float)x, (float)y, (float)z), vec3(localToWorldRows[0].x, localToWorldRows[0].y, localToWorldRows[0].z)) + localToWorldRows[0].w;
					pos.y = dot(vec3((float)x, (float)y, (float)z), vec3(localToWorldRows[1].x, localToWorldRows[1].y, localToWorldRows[1].z)) + localToWorldRows[1].w;
					pos.z = dot(vec3((float)x, (float)y, (float)z), vec3(localToWorldRows[2].x, localToWorldRows[2].y, localToWorldRows[2].z)) + localToWorldRows[2].w;

					m_aabb.m_min = min(m_aabb.m_min, pos);
					m_aabb.m_max = max(m_aabb.m_max, pos);
				}
			}
		}
	}
}

__host__ __device__ inline bool Hittable::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	// transform ray into local space
	Ray lr;
	lr.m_origin.x = dot(r.m_origin, vec3(m_invTransformRow0.x, m_invTransformRow0.y, m_invTransformRow0.z)) + m_invTransformRow0.w;
	lr.m_origin.y = dot(r.m_origin, vec3(m_invTransformRow1.x, m_invTransformRow1.y, m_invTransformRow1.z)) + m_invTransformRow1.w;
	lr.m_origin.z = dot(r.m_origin, vec3(m_invTransformRow2.x, m_invTransformRow2.y, m_invTransformRow2.z)) + m_invTransformRow2.w;

	lr.m_dir.x = dot(r.m_dir, vec3(m_invTransformRow0.x, m_invTransformRow0.y, m_invTransformRow0.z));
	lr.m_dir.y = dot(r.m_dir, vec3(m_invTransformRow1.x, m_invTransformRow1.y, m_invTransformRow1.z));
	lr.m_dir.z = dot(r.m_dir, vec3(m_invTransformRow2.x, m_invTransformRow2.y, m_invTransformRow2.z));

	bool result = false;
	float t;
	vec3 normal;

	switch (m_type)
	{
	case SPHERE:
		result = hitSphere(lr, tMin, tMax, t, normal); break;
	case CYLINDER:
		result = hitCylinder(lr, tMin, tMax, t, normal); break;
	case DISK:
		result = hitDisk(lr, tMin, tMax, t, normal); break;
	case CONE:
		result = hitCone(lr, tMin, tMax, t, normal); break;
	case PARABOLOID:
		result = hitParaboloid(lr, tMin, tMax, t, normal); break;
	default:
		break;
	}

	// transform normal to world space
	if (result)
	{
		vec3 tmp;
		tmp.x = dot(normal, vec3(m_invTransformRow0.x, m_invTransformRow1.x, m_invTransformRow2.x));
		tmp.y = dot(normal, vec3(m_invTransformRow0.y, m_invTransformRow1.y, m_invTransformRow2.y));
		tmp.z = dot(normal, vec3(m_invTransformRow0.z, m_invTransformRow1.z, m_invTransformRow2.z));

		rec.m_t = t;
		rec.m_p = r.at(rec.m_t);
		rec.setFaceNormal(r, normalize(tmp));
		rec.m_material = &m_material;
	}

	return result;
}

inline bool Hittable::boundingBox(AABB &outputBox) const
{
	outputBox = m_aabb;
	return true;
}

__host__ __device__ inline bool Hittable::hitSphere(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const
{
	vec3 oc = r.origin();
	float a = length_squared(r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = length_squared(oc) - 1.0f;

	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// get the closest t that is greater than tMin
	t = t0 > tMin ? t0 : t1;
	normal = r.at(t);
	return true;
}

__host__ __device__ inline bool Hittable::hitCylinder(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const
{
	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, 0, 1, 0, 0, 0, 0, 0, 0, -1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	const float hitPointHeight0 = r.m_dir.y * t0 + r.origin().y;
	const float hitPointHeight1 = r.m_dir.y * t1 + r.origin().y;
	const bool hitPointValid0 = t0 > tMin && t0 <= tMax && hitPointHeight0 >= -1.0f && hitPointHeight0 <= 1.0f;
	const bool hitPointValid1 = t1 > tMin && t1 <= tMax && hitPointHeight1 >= -1.0f && hitPointHeight1 <= 1.0f;

	// both hitpoints are invalid
	if (!hitPointValid0 && !hitPointValid1)
	{
		return false;
	}

	// get t of closest valid hitpoint
	t = hitPointValid0 ? t0 : t1;
	vec3 p = r.at(t);
	normal = vec3(p.x, 0.0f, p.z);
	return true;
}

__host__ __device__ inline bool Hittable::hitDisk(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const
{
	// ray is parallel to disk -> no intersection
	if (r.m_dir.y == 0.0f)
	{
		return false;
	}

	vec3 oc = r.origin();

	// intersection t of ray and plane of disk
	t = -r.m_origin.y / r.m_dir.y;

	if (t <= tMin || t > tMax)
	{
		return false;
	}

	// check that hit point is inside the radius of the disk
	float hitPointX = oc.x + r.m_dir.x * t;
	float hitPointZ = oc.z + r.m_dir.z * t;
	if ((hitPointX * hitPointX + hitPointZ * hitPointZ) >= 1.0f)
	{
		return false;
	}

	normal = vec3(0.0f, 1.0f, 0.0f);
	return true;
}

__host__ __device__ inline bool Hittable::hitCone(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const
{
	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, -1, 1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	const float hitPointHeight0 = r.m_dir.y * t0 + r.origin().y;
	const float hitPointHeight1 = r.m_dir.y * t1 + r.origin().y;
	const bool hitPointValid0 = t0 > tMin && t0 <= tMax && hitPointHeight0 >= -1.0f && hitPointHeight0 <= 1.0f;
	const bool hitPointValid1 = t1 > tMin && t1 <= tMax && hitPointHeight1 >= -1.0f && hitPointHeight1 <= 1.0f;

	// both hitpoints are invalid
	if (!hitPointValid0 && !hitPointValid1)
	{
		return false;
	}

	// get t of closest valid hitpoint
	t = hitPointValid0 ? t0 : t1;

	vec3 p = r.at(t);
	normal = quadricNormal<1, -1, 1>(p);

	return true;
}

__host__ __device__ inline bool Hittable::hitParaboloid(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const
{
	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, 0, 1, 0, 0, 0, 0, -1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	const float hitPointHeight0 = r.m_dir.y * t0 + r.origin().y;
	const float hitPointHeight1 = r.m_dir.y * t1 + r.origin().y;
	const bool hitPointValid0 = t0 > tMin && t0 <= tMax && hitPointHeight0 >= -1.0f && hitPointHeight0 <= 1.0f;
	const bool hitPointValid1 = t1 > tMin && t1 <= tMax && hitPointHeight1 >= -1.0f && hitPointHeight1 <= 1.0f;

	// both hitpoints are invalid
	if (!hitPointValid0 && !hitPointValid1)
	{
		return false;
	}

	// get t of closest valid hitpoint
	t = hitPointValid0 ? t0 : t1;

	vec3 p = r.at(t);
	normal = quadricNormal<1, 0, 1, 0, 0, 0, 0, -1>(p);

	return true;
}