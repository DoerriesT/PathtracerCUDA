#pragma once
#include "cuda_runtime.h"
#include "vec3.h"
#include <cstdint>

struct HitRecord;
struct Ray;

enum class MaterialType : uint32_t
{
	LAMBERT, GGX, LAMBERT_GGX
};

class Material2
{
public:
	__host__ __device__ Material2(MaterialType type = MaterialType::LAMBERT, const vec3 &baseColor = vec3(1.0f), const vec3 &emissive = vec3(0.0f), float roughness = 0.5f, float metalness = 0.0f, uint32_t textureIndex = 0);
	__device__ vec3 sample(const Ray &rIn, const HitRecord &rec, curandState &randState, Ray &scattered, float &pdf, cudaTextureObject_t *textures) const;
	__device__ vec3 getEmitted(const Ray &rIn, const HitRecord &rec) const;

private:
	vec3 m_baseColor;
	float m_roughness;
	vec3 m_emissive;
	float m_metalness;
	uint32_t m_textureIndex;
	MaterialType m_materialType;

	__device__ vec3 sampleLambert(const vec3 &baseColor, const vec3 &V, float rnd0, float rnd1, vec3 &scatteredDir, float &pdf) const;
	__device__ vec3 sampleGGX(const vec3 &baseColor, const vec3 &V, float rnd0, float rnd1, vec3 &scatteredDir, float &pdf) const;
	__device__ vec3 sampleLambertGGX(const vec3 &baseColor, const vec3 &V, float rnd0, float rnd1, vec3 &scatteredDir, float &pdf) const;
};


// IMPLEMENTATION
#include "Material.inl"