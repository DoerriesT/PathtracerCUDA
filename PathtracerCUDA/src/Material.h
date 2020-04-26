#pragma once
#include "vec3.h"
#include "Hittable.h"

struct Ray;
struct HitRecord;
struct vec3;

class Material
{
public:
	__device__ virtual bool scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered)  const = 0;
};

class Lambertian : public Material
{
public:
	__device__ Lambertian(const vec3 &a) : m_albedo(a) {}

	__device__ bool scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const override
	{
		vec3 scatterDir = rec.m_normal + random_unit_vec(randState);
		scattered = Ray(rec.m_p, scatterDir);
		attenuation = m_albedo;
		return true;
	}

public:
	vec3 m_albedo;
};

class Metal : public Material
{
public:
	__device__ Metal(const vec3 &a, float f) : m_albedo(a), m_fuzz(f < 1.0f ? f : 1.0f) {}

	__device__ bool scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const override
	{
		vec3 reflected = reflect(normalize(rIn.direction()), rec.m_normal);
		scattered = Ray(rec.m_p, reflected + m_fuzz * random_in_unit_sphere(randState));
		attenuation = m_albedo;
		return (dot(scattered.direction(), rec.m_normal) > 0.0f);
	}

public:
	vec3 m_albedo;
	float m_fuzz;
};