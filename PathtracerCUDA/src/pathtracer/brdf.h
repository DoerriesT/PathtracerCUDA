#pragma once
#include "vec3.h"

__host__ __device__ inline float pow5(float v)
{
	float v2 = v * v;
	return v2 * v2 * v;
}

// "Moving Frostbite to Physically Based Rendering"
__host__ __device__ inline float D_GGX(float NdotH, float a2)
{
	float d = (NdotH * a2 - NdotH) * NdotH + 1.0f;
	return a2 / (PI * d * d);
}

// "Moving Frostbite to Physically Based Rendering"
__host__ __device__ inline float V_SmithGGXCorrelated(float NdotV, float NdotL, float a2)
{
	float lambdaGGXV = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
	float lambdaGGXL = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);

	return 0.5f / (lambdaGGXV + lambdaGGXL + 1e-5f); // avoids artifacs on some normal mapped surfaces
}

// https://google.github.io/filament/Filament.html#materialsystem/specularbrdf/fresnel(specularf)
__host__ __device__ inline vec3 F_Schlick(vec3 F0, float VdotH)
{
	float pow5Term = pow5(1.0f - VdotH);
	//return F0 + (1.0 - F0) * pow5Term;
	return pow5Term + F0 * (1.0f - pow5Term);
}

// https://google.github.io/filament/Filament.html#materialsystem/specularbrdf/fresnel(specularf)
__host__ __device__ inline vec3 F_Schlick(vec3 F0, float F90, float VdotH)
{
	return F0 + (F90 - F0) * pow5(1.0f - VdotH);
}

// renormalized according to "Moving Frostbite to Physically Based Rendering"
__host__ __device__ inline vec3 Diffuse_Disney(float NdotV, float NdotL, float VdotH, float roughness, vec3 baseColor)
{
	float energyBias = lerp(0.0f, 0.5f, roughness);
	float energyFactor = lerp(1.0f, 1.0f / 1.51f, roughness);
	float fd90 = energyBias + 2.0f * VdotH * VdotH * roughness;
	float lightScatter = (1.0f + (fd90 - 1.0f) * pow5(1.0f - NdotL));
	float viewScatter = (1.0f + (fd90 - 1.0f) * pow5(1.0f - NdotV));
	return lightScatter * viewScatter * energyFactor * baseColor * (1.0f / PI);
}

__host__ __device__ inline vec3 Diffuse_Lambert(vec3 baseColor)
{
	return baseColor * (1.0f / PI);
}

__host__ __device__ inline vec3 Specular_GGX(vec3 F0, float NdotV, float NdotL, float NdotH, float VdotH, float a2)
{
	float D = D_GGX(NdotH, a2);
	float V = V_SmithGGXCorrelated(NdotV, NdotL, a2);
	vec3 F = F_Schlick(F0, VdotH);
	return D * V * F;
}