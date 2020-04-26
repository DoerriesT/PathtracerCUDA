#pragma once
#include "vec3.h"
#include "Hittable.h"

struct Ray;
struct HitRecord;
struct vec3;

__device__ float fresnelSchlick(float cosine, float ior)
{
	auto r0 = (1.0f - ior) / (1.0f + ior);
	r0 = r0 * r0;
	float powerTerm = 1.0f - cosine;
	powerTerm *= powerTerm;
	powerTerm *= powerTerm;
	powerTerm *= 1.0f - cosine;
	return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

__device__ float ffmin(float a, float b)
{
	return a < b ? a : b;
}

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

class Dielectric : public Material
{
public:
	__device__ Dielectric(float ior) : m_ior(ior) {}

	__device__ bool scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const override
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

	float m_ior;
};