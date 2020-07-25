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
		result = hitCylinder(lr, tMin, tMax, rec); break;
	case DISK:
		result = hitDisk(lr, tMin, tMax, rec); break;
	case CONE:
		result = hitCone(lr, tMin, tMax, rec); break;
	case PARABOLOID:
		result = hitParaboloid(lr, tMin, tMax, rec); break;
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
	const auto &sphere = m_payload.m_sphere;
	vec3 oc = r.origin();// -sphere.m_center;
	float a = length_squared(r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = length_squared(oc) - 1.0f;// sphere.m_radius *sphere.m_radius;

	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// get the closest t that is greater than tMin
	t = t0 > tMin ? t0 : t1;
	normal = r.at(t);// (rec.m_p - sphere.m_center) / sphere.m_radius;
	return true;
}

__host__ __device__ inline bool Hittable::hitCylinder(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	const auto &cylinder = m_payload.m_cylinder;
	vec3 oc = r.origin() - cylinder.m_center;
	float a = r.m_dir.x * r.m_dir.x + r.m_dir.z * r.m_dir.z;
	float b = 2.0f * (r.m_dir.x * oc.x + r.m_dir.z * oc.z);
	float c = oc.x * oc.x + oc.z * oc.z - cylinder.m_radius * cylinder.m_radius;

	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	const float hitPointHeight0 = r.m_dir.y * t0 + oc.y;
	const float hitPointHeight1 = r.m_dir.y * t1 + oc.y;
	const bool hitPointValid0 = t0 > tMin && t0 <= tMax && hitPointHeight0 >= -cylinder.m_halfHeight && hitPointHeight0 <= cylinder.m_halfHeight;
	const bool hitPointValid1 = t1 > tMin && t1 <= tMax && hitPointHeight1 >= -cylinder.m_halfHeight && hitPointHeight1 <= cylinder.m_halfHeight;

	// both hitpoints are invalid
	if (!hitPointValid0 && !hitPointValid1)
	{
		return false;
	}

	// get t of closest valid hitpoint
	float t = hitPointValid0 ? t0 : t1;

	rec.m_t = t;
	rec.m_p = r.at(rec.m_t);
	vec3 outwardNormal = (rec.m_p - vec3(cylinder.m_center.x, rec.m_p.y, cylinder.m_center.z)) / cylinder.m_radius;
	rec.setFaceNormal(r, outwardNormal);
	rec.m_material = &m_material;
	return true;
}

__host__ __device__ inline bool Hittable::hitDisk(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	// ray is parallel to disk -> no intersection
	if (r.m_dir.y == 0.0f)
	{
		return false;
	}

	const auto &disk = m_payload.m_disk;
	vec3 oc = r.origin() - disk.m_center;

	// intersection t of ray and plane of disk
	float t = (disk.m_center.y - r.m_origin.y) / r.m_dir.y;

	if (t <= tMin || t > tMax)
	{
		return false;
	}

	// check that hit point is inside the radius of the disk
	float hitPointX = oc.x + r.m_dir.x * t;
	float hitPointZ = oc.z + r.m_dir.z * t;
	if ((hitPointX * hitPointX + hitPointZ * hitPointZ) >= (disk.m_radius * disk.m_radius))
	{
		return false;
	}

	rec.m_t = t;
	rec.m_p = r.at(t);
	vec3 outwardNormal = vec3(0.0f, 1.0f, 0.0f);
	rec.setFaceNormal(r, outwardNormal);
	rec.m_material = &m_material;
	return true;
}

__host__ __device__ inline bool Hittable::hitCone(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	const auto &cone = m_payload.m_cone;
	vec3 oc = r.origin() - cone.m_center;

	float k = cone.m_radius / cone.m_height;
	k *= k;
	float a = r.m_dir.x * r.m_dir.x + r.m_dir.z * r.m_dir.z - k * r.m_dir.y * r.m_dir.y;
	float b = 2.0f * (r.m_dir.x * oc.x + r.m_dir.z * oc.z - k * r.m_dir.y * (oc.y - cone.m_height));
	float c = oc.x * oc.x + oc.z * oc.z - k * (oc.y - cone.m_height) * (oc.y - cone.m_height);

	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	const float hitPointHeight0 = r.m_dir.y * t0 + oc.y;
	const float hitPointHeight1 = r.m_dir.y * t1 + oc.y;
	const bool hitPointValid0 = t0 > tMin && t0 <= tMax && hitPointHeight0 >= 0.0f && hitPointHeight0 <= cone.m_height;
	const bool hitPointValid1 = t1 > tMin && t1 <= tMax && hitPointHeight1 >= 0.0f && hitPointHeight1 <= cone.m_height;

	// both hitpoints are invalid
	if (!hitPointValid0 && !hitPointValid1)
	{
		return false;
	}

	// get t of closest valid hitpoint
	float t = hitPointValid0 ? t0 : t1;

	rec.m_t = t;
	rec.m_p = r.at(rec.m_t);

	// calculate normal
	{
		// x, z components of vector from center to p
		float vX = rec.m_p.x - cone.m_center.x;
		float vZ = rec.m_p.z - cone.m_center.z;
		// normalize the vector
		float normFactor = 1.0f / sqrtf(vX * vX + vZ * vZ);
		vX *= normFactor;
		vZ *= normFactor;
		float heightOverRadius = cone.m_height / cone.m_radius;
		vec3 outwardNormal = vec3(vX * heightOverRadius, cone.m_radius / cone.m_height, vZ * heightOverRadius);

		rec.setFaceNormal(r, outwardNormal);
	}
	rec.m_material = &m_material;
	return true;
}

__host__ __device__ inline bool Hittable::hitParaboloid(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	const auto &para = m_payload.m_paraboloid;
	vec3 oc = r.origin() - para.m_center;

	float k = para.m_height / (para.m_radius * para.m_radius);
	float a = k * (r.m_dir.x * r.m_dir.x + r.m_dir.z * r.m_dir.z);
	float b = 2.0f * k * (r.m_dir.x * oc.x + r.m_dir.z * oc.z) - r.m_dir.y;
	float c = k * (oc.x * oc.x + oc.z * oc.z) - oc.y;

	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	const float hitPointHeight0 = r.m_dir.y * t0 + oc.y;
	const float hitPointHeight1 = r.m_dir.y * t1 + oc.y;
	const bool hitPointValid0 = t0 > tMin && t0 <= tMax && hitPointHeight0 >= 0.0f && hitPointHeight0 <= para.m_height;
	const bool hitPointValid1 = t1 > tMin && t1 <= tMax && hitPointHeight1 >= 0.0f && hitPointHeight1 <= para.m_height;

	// both hitpoints are invalid
	if (!hitPointValid0 && !hitPointValid1)
	{
		return false;
	}

	// get t of closest valid hitpoint
	float t = hitPointValid0 ? t0 : t1;

	rec.m_t = t;
	rec.m_p = r.at(rec.m_t);

	// calculate normal
	{
		vec3 localHit = oc + t * r.m_dir;
		const float phiMax = 2.0f * PI;
		vec3 ddx = vec3(-phiMax * localHit.z, 0.0f, phiMax * localHit.x);
		vec3 ddy = para.m_height * vec3(localHit.x / (2.0f * localHit.y), 1.0f, localHit.z / (2.0f * localHit.y));


		vec3 outwardNormal = normalize(cross(ddx, ddy));

		rec.setFaceNormal(r, outwardNormal);
	}
	rec.m_material = &m_material;
	return true;
}