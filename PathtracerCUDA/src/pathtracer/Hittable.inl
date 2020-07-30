#pragma once
#include "Hittable.h"
#include "HitRecord.h"
#include "AABB.h"

// solve quadratic equation of the form ax^2 + bx + c = 0
__host__ __device__ inline bool quadratic(float a, float b, float c, float &t0, float &t1)
{
	// roots are given by: x = -b +- sqrt(b^2 - 4ac) / 2a
	// b^2 - 4ac is the discriminant
	// discriminant > 0:
	//    root1 = -b + sqrt(b^2 - 4ac) / 2a
	//    root2 = -b - sqrt(b^2 - 4ac) / 2a
	// discriminant = 0:
	//    root1 = root2 = -b / 2a
	// discriminant < 0 has complex solutions can be disregarded for our purpose

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

	// swap values so that t0 <= t1
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
	: m_invTransformRow0({ 1.0f, 0.0f, 0.0f, 0.0f }),
	m_invTransformRow1({ 0.0f, 1.0f, 0.0f, 0.0f }),
	m_invTransformRow2({ 0.0f, 0.0f, 1.0f, 0.0f }),
	m_material(),
	m_type(HittableType::SPHERE)
{
}

__host__ __device__ inline Hittable::Hittable(HittableType type, const float4 *invTransformRows, const Material &material)
	: m_invTransformRow0(invTransformRows[0]),
	m_invTransformRow1(invTransformRows[1]),
	m_invTransformRow2(invTransformRows[2]),
	m_material(material),
	m_type(type)
{

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
	float u;
	float v;
	vec3 normal;

	// we could use inheritance instead of using a single class and switching on an enum,
	// but doing so gives much worse performance with CUDA
	switch (m_type)
	{
	case HittableType::SPHERE:
		result = hitSphere(lr, tMin, tMax, t, normal, u, v); break;
	case HittableType::CYLINDER:
		result = hitCylinder(lr, tMin, tMax, t, normal, u, v); break;
	case HittableType::DISK:
		result = hitDisk(lr, tMin, tMax, t, normal, u, v); break;
	case HittableType::CONE:
		result = hitCone(lr, tMin, tMax, t, normal, u, v); break;
	case HittableType::PARABOLOID:
		result = hitParaboloid(lr, tMin, tMax, t, normal, u, v); break;
	case HittableType::QUAD:
		result = hitQuad(lr, tMin, tMax, t, normal, u, v); break;
	case HittableType::CUBE:
		result = hitBox(lr, tMin, tMax, t, normal, u, v); break;
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
		rec.m_texCoordU = u;
		rec.m_texCoordV = v;
	}

	return result;
}

__host__ __device__ inline bool Hittable::hitSphere(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
{
	// intersect ray with quadric
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, 1, 1, 0, 0, 0, 0, 0, 0, -1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// get the closest t that is greater than tMin
	t = t0 > tMin ? t0 : t1;
	normal = normalize(r.at(t));

	// convert cartesian coordinates (normal) to spherical coordinates (texture coordinate)
	float theta = acosf(normal.y);
	float phi = atan2f(normal.z, normal.x);
	u = 1.0f - phi / (2.0f * PI);
	v = theta / PI;


	return true;
}

__host__ __device__ inline bool Hittable::hitCylinder(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
{
	// intersect ray with quadric
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, 0, 1, 0, 0, 0, 0, 0, 0, -1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// cylinder is infinite, but we limit it to a maximum extent
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

	float phi = atan2f(normal.z, normal.x);
	u = 1.0f - phi / (2.0f * PI);
	v = 1.0f - (p.y * 0.5f + 0.5f);

	return true;
}

__host__ __device__ inline bool Hittable::hitDisk(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
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
	u = hitPointX * 0.5f + 0.5f;
	v = 1.0f - (hitPointZ * 0.5f + 0.5f);
	return true;
}

__host__ __device__ inline bool Hittable::hitCone(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
{
	// intersect ray with quadric
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, -1, 1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// the cone is infinite, but we limit it to a maximum extent
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

__host__ __device__ inline bool Hittable::hitParaboloid(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
{
	// intersect ray with quadric
	float t0 = 0.0f;
	float t1 = 0.0f;
	if (!hitQuadric<1, 0, 1, 0, 0, 0, 0, -1>(r.origin(), r.direction(), t0, t1) || t0 > tMax || t1 <= tMin)
	{
		return false;
	}

	// the paraboliod is infinite in the y direction, but we limit it to a maximum extent
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

inline __host__ __device__ bool Hittable::hitQuad(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
{
	// ray is parallel to quad -> no intersection
	if (r.m_dir.y == 0.0f)
	{
		return false;
	}

	vec3 oc = r.origin();

	// intersection t of ray and plane of quad
	t = -r.m_origin.y / r.m_dir.y;

	if (t <= tMin || t > tMax)
	{
		return false;
	}

	// check that hit point is inside the extent of the quad
	float hitPointX = oc.x + r.m_dir.x * t;
	float hitPointZ = oc.z + r.m_dir.z * t;
	if (fabsf(hitPointX) > 1.0f || fabsf(hitPointZ) > 1.0f)
	{
		return false;
	}

	normal = vec3(0.0f, 1.0f, 0.0f);
	u = hitPointX * 0.5f + 0.5f;
	v = 1.0f - (hitPointZ * 0.5f + 0.5f);
	return true;
}

inline __host__ __device__ bool Hittable::hitBox(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const
{
	// reuse AABB intersection for this shape
	AABB aabb = { vec3(-1.0f), vec3(1.0f) };
	if (!aabb.intersect(r, tMin, tMax, t))
	{
		return false;
	}

	// compute normal by finding the maximum absolute component of a point on the box 
	// (box is centered around the local coordinate system center)
	normal = r.at(t);
	vec3 absN = vec3(fabsf(normal.x), fabsf(normal.y), fabsf(normal.z));
	if (absN.x > absN.y && absN.x > absN.z)
	{
		normal = vec3(normal.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
	}
	else if (absN.y > absN.x && absN.y > absN.z)
	{
		normal = vec3(0.0f, normal.y > 0.0f ? 1.0f : -1.0f, 0.0f);
	}
	else
	{
		normal = vec3(0.0f, 0.0f, normal.z > 0.0f ? 1.0f : -1.0f);
	}

	return true;
}
