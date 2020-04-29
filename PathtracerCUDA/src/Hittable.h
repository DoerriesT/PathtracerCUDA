#pragma once
#include "Ray.h"
#include "vec3.h"

class Material;
class Material2;

__host__ __device__ float fresnelSchlick(float cosine, float ior)
{
	auto r0 = (1.0f - ior) / (1.0f + ior);
	r0 = r0 * r0;
	float powerTerm = 1.0f - cosine;
	powerTerm *= powerTerm;
	powerTerm *= powerTerm;
	powerTerm *= 1.0f - cosine;
	return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

__host__ __device__ float ffmin(float a, float b)
{
	return a < b ? a : b;
}

struct HitRecord2
{
	vec3 m_p;
	vec3 m_normal;
	const Material2 *m_material;
	float m_t;
	bool m_frontFace;

	__host__ __device__ inline void setFaceNormal(const Ray &r, const vec3 &outwardNormal)
	{
		m_frontFace = dot(r.direction(), outwardNormal) < 0.0f;
		m_normal = m_frontFace ? outwardNormal : -outwardNormal;
	}
};

class Material2
{
public:
	enum Type : uint32_t
	{
		LAMBERTIAN, METAL, DIELECTRIC
	};

	__host__ __device__ Material2(Type type, const vec3 &albedo, float fuzz = 0.0f, float ior = 1.0f)
		:m_type(type),
		m_albedo(albedo),
		m_fuzz(fuzz),
		m_ior(ior)
	{

	}

	__device__ bool scatter(const Ray &rIn, const HitRecord2 &rec, curandState &randState, vec3 &attenuation, Ray &scattered)  const
	{
		switch (m_type)
		{
		case Material2::LAMBERTIAN:
			return scatterLambertian(rIn, rec, randState, attenuation, scattered);
		case Material2::METAL:
			return scatterMetal(rIn, rec, randState, attenuation, scattered);
		case Material2::DIELECTRIC:
			return scatterDielectric(rIn, rec, randState, attenuation, scattered);
		default:
			break;
		}
		return false;
	}

private:
	vec3 m_albedo;
	Type m_type;
	float m_fuzz;
	float m_ior;

	__device__ bool scatterLambertian(const Ray &rIn, const HitRecord2 &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
	{
		vec3 scatterDir = rec.m_normal + random_unit_vec(randState);
		scattered = Ray(rec.m_p, scatterDir);
		attenuation = m_albedo;
		return true;
	}

	__device__ bool scatterMetal(const Ray &rIn, const HitRecord2 &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
	{
		vec3 reflected = reflect(normalize(rIn.direction()), rec.m_normal);
		scattered = Ray(rec.m_p, reflected + m_fuzz * random_in_unit_sphere(randState));
		attenuation = m_albedo;
		return (dot(scattered.direction(), rec.m_normal) > 0.0f);
	}

	__device__ bool scatterDielectric(const Ray &rIn, const HitRecord2 &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
	{
		attenuation = vec3(1.0f, 1.0f, 1.0f);
		float etaiOverEtat = rec.m_frontFace ? 1.0f / m_ior : m_ior;

		vec3 unitDir = normalize(rIn.direction());
		float cosTheta = ffmin(dot(-unitDir, rec.m_normal), 1.0f);
		float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

		if (etaiOverEtat * sinTheta > 1.0f)
		{
			vec3 reflected = reflect(unitDir, rec.m_normal);
			scattered = Ray(rec.m_p, reflected);
			return true;
		}

		float reflectProb = fresnelSchlick(cosTheta, etaiOverEtat);
		if (curand_uniform(&randState) < reflectProb)
		{
			vec3 reflected = reflect(unitDir, rec.m_normal);
			scattered = Ray(rec.m_p, reflected);
			return true;
		}

		vec3 refracted = refract(unitDir, rec.m_normal, etaiOverEtat);
		scattered = Ray(rec.m_p, refracted);
		return true;
	}
};

struct Sphere2
{
	vec3 m_center;
	float m_radius;
};

class Hittable2
{
public:
	enum Type : uint32_t
	{
		SPHERE
	};

	union Payload
	{
		Sphere2 m_sphere;
	};

	__host__ __device__ Hittable2(Type type, const Material2 &material, const Payload &payload)
		: m_type(type),
		m_material(material),
		m_payload(payload)
	{
	}

	__host__ __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord2 &rec) const
	{
		switch (m_type)
		{
		case SPHERE:
			return hitSphere(r, tMin, tMax, rec);
		default:
			break;
		}
		return false;
	}

private:
	Type m_type;
	Material2 m_material;
	Payload m_payload;

	__host__ __device__ bool hitSphere(const Ray &r, float tMin, float tMax, HitRecord2 &rec) const
	{
		const auto &sphere = m_payload.m_sphere;
		vec3 oc = r.origin() - sphere.m_center;
		auto a = length_squared(r.direction());
		auto half_b = dot(oc, r.direction());
		auto c = length_squared(oc) - sphere.m_radius * sphere.m_radius;
		auto discriminant = half_b * half_b - a * c;

		if (discriminant > 0.0f)
		{
			auto root = sqrt(discriminant);
			auto temp = (-half_b - root) / a;
			if (temp < tMax && temp > tMin)
			{
				rec.m_t = temp;
				rec.m_p = r.at(rec.m_t);
				vec3 outwardNormal = (rec.m_p - sphere.m_center) / sphere.m_radius;
				rec.setFaceNormal(r, outwardNormal);
				rec.m_material = &m_material;
				return true;
			}
			temp = (-half_b + root) / a;
			if (temp < tMax && temp > tMin)
			{
				rec.m_t = temp;
				rec.m_p = r.at(rec.m_t);
				vec3 outwardNormal = (rec.m_p - sphere.m_center) / sphere.m_radius;
				rec.setFaceNormal(r, outwardNormal);
				rec.m_material = &m_material;
				return true;
			}
		}

		return false;
	}
};

struct HitRecord
{
	vec3 m_p;
	vec3 m_normal;
	Material *m_material;
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
	__device__ Sphere(vec3 center, float radius, Material *material) 
		: m_center(center), 
		m_radius(radius),
		m_material(material)
	{};

	__device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;

public:
	vec3 m_center;
	float m_radius;
	Material *m_material;
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
			rec.m_material = m_material;
			return true;
		}
		temp = (-half_b + root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.m_t = temp;
			rec.m_p = r.at(rec.m_t);
			vec3 outwardNormal = (rec.m_p - m_center) / m_radius;
			rec.setFaceNormal(r, outwardNormal);
			rec.m_material = m_material;
			return true;
		}
	}

	return false;
}