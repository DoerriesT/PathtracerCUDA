#pragma once
#include "Material.h"
#include "HitRecord.h"
#include "brdf.h"
#include <curand_kernel.h>



__host__ __device__ inline float fresnelSchlick(float cosine, float ior)
{
	auto r0 = (1.0f - ior) / (1.0f + ior);
	r0 = r0 * r0;
	float powerTerm = 1.0f - cosine;
	powerTerm *= powerTerm;
	powerTerm *= powerTerm;
	powerTerm *= 1.0f - cosine;
	return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

__host__ __device__ inline  float ffmin(float a, float b)
{
	return a < b ? a : b;
}


__host__ __device__ inline Material2::Material2(const vec3 &baseColor, const vec3 &emissive, float roughness, float metalness, uint32_t textureIndex)
	:m_baseColor(baseColor),
	m_emissive(emissive),
	m_roughness(roughness < 0.04f ? 0.04f : roughness),
	m_metalness(metalness),
	m_textureIndex(textureIndex)
{

}

__device__ vec3 inline Material2::sample(const Ray &rIn, const HitRecord &rec, curandState &randState, Ray &scattered, float &pdf, cudaTextureObject_t *textures) const
{
#if __CUDA_ARCH__ 
	const vec3 N = normalize(rec.m_normal);
	const vec3 V = normalize(rIn.m_dir);
	vec3 L = cosineSampleHemisphere(curand_uniform(&randState), curand_uniform(&randState), pdf);
	L = tangentToWorld(N, L);

	scattered = Ray(rec.m_p, L);

	float NdotV = abs(dot(N, N)) + 1e-5f;
	vec3 H = normalize(N + L);
	float VdotH = clamp(dot(V, H));
	float NdotH = clamp(dot(N, H));
	float NdotL = clamp(dot(N, L));

	float a = m_roughness * m_roughness;
	float a2 = a * a;

	vec3 baseColor = m_baseColor;
	if (m_textureIndex != 0)
	{
		float4 tap = tex2D<float4>(textures[m_textureIndex - 1], rec.m_texCoordU, rec.m_texCoordV);
		baseColor = vec3(tap.x, tap.y, tap.z);
		baseColor.x = pow(baseColor.x, 2.2f);
		baseColor.y = pow(baseColor.y, 2.2f);
		baseColor.z = pow(baseColor.z, 2.2f);
	}

	vec3 F0 = lerp(vec3(0.04f), baseColor, baseColor);
	vec3 kS = Specular_GGX(F0, NdotV, NdotL, NdotH, VdotH, a2);
	vec3 kD = Diffuse_Lambert(baseColor);
	//vec3 kD = (28.0f / (23.0f * PI)) * m_baseColor * (1.0f - F0) * (1.0f - pow5(1.0f - 0.5f * NdotL)) * (1.0f - (1.0f - pow5(NdotV)));

	return kD * (1.0f - m_metalness);// +kS;
#else
	return vec3();
#endif // __CUDA_ARCH__
}

__device__ inline vec3 Material2::getEmitted(const Ray &rIn, const HitRecord &rec) const
{
	return m_emissive;
}

__host__ __device__ inline MaterialOld::MaterialOld(Type type, const vec3 &albedo, float fuzz, float ior)
	:m_type(type),
	m_albedo(albedo),
	m_fuzz(fuzz),
	m_ior(ior)
{
}

__device__ inline bool MaterialOld::scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
{
	switch (m_type)
	{
	case MaterialOld::LAMBERTIAN:
		return scatterLambertian(rIn, rec, randState, attenuation, scattered);
	case MaterialOld::METAL:
		return scatterMetal(rIn, rec, randState, attenuation, scattered);
	case MaterialOld::DIELECTRIC:
		return scatterDielectric(rIn, rec, randState, attenuation, scattered);
	default:
		break;
	}
	return false;
}

__device__ inline bool MaterialOld::scatterLambertian(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
{
	vec3 scatterDir = rec.m_normal + random_unit_vec(randState);
	scattered = Ray(rec.m_p, scatterDir);
	attenuation = m_albedo;
	return true;
}

__device__ inline bool MaterialOld::scatterMetal(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
{
	vec3 reflected = reflect(normalize(rIn.direction()), rec.m_normal);
	scattered = Ray(rec.m_p, reflected + m_fuzz * random_in_unit_sphere(randState));
	attenuation = m_albedo;
	return (dot(scattered.direction(), rec.m_normal) > 0.0f);
}

__device__ inline bool MaterialOld::scatterDielectric(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
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
