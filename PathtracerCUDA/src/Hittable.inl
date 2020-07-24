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
}

__host__ __device__ inline Hittable::Hittable(Type type, const Payload &payload, const Material2 &material)
	: m_type(type),
	m_material(material),
	m_payload(payload)
{
}

__host__ __device__ inline bool Hittable::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	switch (m_type)
	{
	case SPHERE:
		return hitSphere(r, tMin, tMax, rec);
	case CYLINDER:
		return hitCylinder(r, tMin, tMax, rec);
	case DISK:
		return hitDisk(r, tMin, tMax, rec);
	case CONE:
		return hitCone(r, tMin, tMax, rec);
	case PARABOLOID:
		return hitParaboloid(r, tMin, tMax, rec);
	default:
		break;
	}
	return false;
}

__host__ __device__ inline bool Hittable::boundingBox(AABB &outputBox) const
{
	switch (m_type)
	{
	case SPHERE:
		return sphereBoundingBox(outputBox);
	case CYLINDER:
		return cylinderBoundingBox(outputBox);
	case DISK:
		return diskBoundingBox(outputBox);
	case CONE:
		return coneBoundingBox(outputBox);
	case PARABOLOID:
		return paraboloidBoundingBox(outputBox);
	default:
		break;
	}
	return false;
}

__host__ __device__ inline bool Hittable::hitSphere(const Ray &r, float tMin, float tMax, HitRecord &rec) const
{
	const auto &sphere = m_payload.m_sphere;
	vec3 oc = r.origin() - sphere.m_center;
	float a = length_squared(r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = length_squared(oc) - sphere.m_radius * sphere.m_radius;

	// solve the quadratic equation
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// get the closest t that is greater than tMin
	rec.m_t = t0 > tMin ? t0 : t1;
	rec.m_p = r.at(rec.m_t);
	vec3 outwardNormal = (rec.m_p - sphere.m_center) / sphere.m_radius;
	rec.setFaceNormal(r, outwardNormal);
	rec.m_material = &m_material;
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

__host__ __device__ inline bool Hittable::sphereBoundingBox(AABB &outputBox) const
{
	const auto &sphere = m_payload.m_sphere;
	outputBox = AABB(
		sphere.m_center - vec3(sphere.m_radius, sphere.m_radius, sphere.m_radius),
		sphere.m_center + vec3(sphere.m_radius, sphere.m_radius, sphere.m_radius)
	);
	return true;
}

__host__ __device__ inline bool Hittable::cylinderBoundingBox(AABB &outputBox) const
{
	const auto &cylinder = m_payload.m_cylinder;
	outputBox = AABB(
		cylinder.m_center - vec3(cylinder.m_radius, cylinder.m_halfHeight, cylinder.m_radius),
		cylinder.m_center + vec3(cylinder.m_radius, cylinder.m_halfHeight, cylinder.m_radius)
	);
	return true;
}

__host__ __device__ inline bool Hittable::diskBoundingBox(AABB &outputBox) const
{
	const auto &disk = m_payload.m_disk;
	outputBox = AABB(
		disk.m_center - vec3(disk.m_radius, -FLT_EPSILON, disk.m_radius),
		disk.m_center + vec3(disk.m_radius, FLT_EPSILON, disk.m_radius)
	);
	return true;
}

__host__ __device__ inline bool Hittable::coneBoundingBox(AABB &outputBox) const
{
	const auto &cone = m_payload.m_cone;
	outputBox = AABB(
		cone.m_center - vec3(cone.m_radius, 0.0f, cone.m_radius),
		cone.m_center + vec3(cone.m_radius, cone.m_height, cone.m_radius)
	);
	return true;
}

__host__ __device__ inline bool Hittable::paraboloidBoundingBox(AABB &outputBox) const
{
	const auto &para = m_payload.m_paraboloid;
	outputBox = AABB(
		para.m_center - vec3(para.m_radius, 0.0f, para.m_radius),
		para.m_center + vec3(para.m_radius, para.m_height, para.m_radius)
	);
	return true;
}
