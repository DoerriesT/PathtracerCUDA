#pragma once
#include "cuda_runtime.h"
#include "vec3.h"

struct HitRecord;
struct Ray;

class Material2
{
public:
	__host__ __device__ Material2(const vec3 &baseColor = vec3(1.0f), const vec3 &emissive = vec3(0.0f), float roughness = 0.5f, float metalness = 0.0f, uint32_t textureIndex = 0);
	__device__ vec3 sample(const Ray &rIn, const HitRecord &rec, curandState &randState, Ray &scattered, float &pdf, cudaTextureObject_t *textures) const;
	__device__ vec3 getEmitted(const Ray &rIn, const HitRecord &rec) const;

private:
	vec3 m_baseColor;
	vec3 m_emissive;
	float m_roughness;
	float m_metalness;
	uint32_t m_textureIndex;
};

class MaterialOld
{
public:
	enum Type : uint32_t
	{
		LAMBERTIAN, METAL, DIELECTRIC
	};

	__host__ __device__ MaterialOld(Type type = LAMBERTIAN, const vec3 &albedo = vec3(1.0f), float fuzz = 0.0f, float ior = 1.0f);
	__device__ bool scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered)  const;

private:
	vec3 m_albedo;
	Type m_type;
	float m_fuzz;
	float m_ior;

	__device__ bool scatterLambertian(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const;
	__device__ bool scatterMetal(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const;
	__device__ bool scatterDielectric(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const;
};


// IMPLEMENTATION
#include "Material.inl"