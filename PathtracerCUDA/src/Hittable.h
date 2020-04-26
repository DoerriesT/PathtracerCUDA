#pragma once
#include "Ray.h"
#include "vec3.h"

struct HitRecord
{
	vec3 m_p;
	vec3 m_normal;
	float m_t;
	bool m_frontFace;

	__device__ inline void setFaceNormal(const Ray &r, const vec3 &outwardNormal)
	{
		m_frontFace = dot(r.direction(), outwardNormal) < 0.0f;
		m_normal = m_frontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};

class HittableList : public Hittable
{
public:
	__device__ HittableList() {}
	__device__ HittableList(Hittable **l, int n) : m_list(l), m_listSize(n) {}
	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;

public:
	Hittable **m_list;
	int m_listSize;
};

__device__ bool HittableList::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
	HitRecord tempRec;
	bool hitAnything = false;
	auto closest = t_max;

	for (int i = 0; i < m_listSize; ++i)
	{
		if (m_list[i]->hit(r, t_min, closest, tempRec))
		{
			hitAnything = true;
			closest = tempRec.m_t;
			rec = tempRec;
		}
	}

	return hitAnything;
}

class Sphere : public Hittable
{
public:
	__device__ Sphere() {}
	__device__ Sphere(vec3 center, float radius) : m_center(center), m_radius(radius) {};

	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;

public:
	vec3 m_center;
	float m_radius;
};

__device__ bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
	vec3 oc = r.origin() - m_center;
	auto a = length_squared(r.direction());
	auto half_b = dot(oc, r.direction());
	auto c = length_squared(oc) - m_radius * m_radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant > 0.0f)
	{
		auto root = sqrt(discriminant);
		auto temp = (-half_b - root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.m_t = temp;
			rec.m_p = r.at(rec.m_t);
			vec3 outwardNormal = (rec.m_p - m_center) / m_radius;
			rec.setFaceNormal(r, outwardNormal);
			return true;
		}
		temp = (-half_b + root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.m_t = temp;
			rec.m_p = r.at(rec.m_t);
			vec3 outwardNormal = (rec.m_p - m_center) / m_radius;
			rec.setFaceNormal(r, outwardNormal);
			return true;
		}
	}

	return false;
}