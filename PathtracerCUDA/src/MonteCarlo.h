#pragma once
#include "brdf.h"

__host__ __device__ inline vec3 tangentToWorld(const vec3 &N, const vec3 &v)
{
	vec3 up = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	return normalize(tangent * v.x + bitangent * v.y + N * v.z);
}

__host__ __device__ inline vec3 worldToTangent(const vec3 &N, const vec3 &v)
{
	vec3 up = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	return normalize(vec3(tangent.x, bitangent.x, N.x) * v.x + vec3(tangent.y, bitangent.y, N.y) * v.y + vec3(tangent.z, bitangent.z, N.z) * v.z);
}

__host__ __device__ inline vec3 cosineSampleHemisphere(float u0, float u1)
{
	const float phi = 2.0f * PI * u0;
	const float cosTheta = sqrt(u1);
	const float sinTheta = sqrt(1.0f - u1);
	return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

__host__ __device__ inline float cosineSampleHemispherePdf(const vec3 &L)
{
	return L.z / PI;
}

__host__ __device__ inline vec3 uniformSampleHemisphere(float u0, float u1)
{
	const float phi = 2.0f * PI * u0;
	const float cosTheta = u1;
	const float sinTheta = sqrt(1.0f - u1 * u1);
	return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

__host__ __device__ inline float uniformSampleHemispherePdf()
{
	return 1.0f / (2.0f * PI);
}

// based on information from "Physically Based Shading at Disney" by Brent Burley
__host__ __device__ inline vec3 importanceSampleGGX(float u0, float u1, float a2)
{
	float phi = 2.0f * PI * u0;
	float cosTheta = sqrt((1.0f - u1) / (1.0f + (a2 - 1.0f) * u1 + 1e-5f) + 1e-5f);
	cosTheta = clamp(cosTheta);
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta + 1e-5f);

	float sinPhi = sin(phi);
	float cosPhi = cos(phi);
	vec3 H = normalize(vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
	return H;
}

__host__ __device__ inline float importanceSampleGGXPdf(const vec3 &H, const vec3 &V, float a2)
{
	float NdotH = H.z;//clamp(dot(N, Hw));
	float VdotH = clamp(dot(V, H));
	return (D_GGX(NdotH, a2) * NdotH + 1e-5f) / (4.0f * VdotH + 1e-5f);
}