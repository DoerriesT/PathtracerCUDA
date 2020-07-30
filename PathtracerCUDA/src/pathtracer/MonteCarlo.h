#pragma once
#include "brdf.h"

__host__ __device__ inline vec3 tangentToWorld(const vec3 &N, const vec3 &v)
{
	vec3 up = fabsf(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	return normalize(tangent * v.x + bitangent * v.y + N * v.z);
}

__host__ __device__ inline vec3 worldToTangent(const vec3 &N, const vec3 &v)
{
	vec3 up = fabsf(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	return normalize(vec3(tangent.x, bitangent.x, N.x) * v.x + vec3(tangent.y, bitangent.y, N.y) * v.y + vec3(tangent.z, bitangent.z, N.z) * v.z);
}

__host__ __device__ inline vec3 cosineSampleHemisphere(float u0, float u1)
{
	const float phi = 2.0f * PI * u0;
	const float cosTheta = sqrtf(u1);
	const float sinTheta = sqrtf(1.0f - u1);
	return vec3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}

__host__ __device__ inline float cosineSampleHemispherePdf(const vec3 &L)
{
	return L.z / PI;
}

__host__ __device__ inline vec3 uniformSampleHemisphere(float u0, float u1)
{
	const float phi = 2.0f * PI * u0;
	const float cosTheta = u1;
	const float sinTheta = sqrtf(1.0f - u1 * u1);
	return vec3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}

__host__ __device__ inline float uniformSampleHemispherePdf()
{
	return 1.0f / (2.0f * PI);
}

// based on information from "Physically Based Shading at Disney" by Brent Burley
__host__ __device__ inline vec3 importanceSampleGGX(float u0, float u1, float a2)
{
	float phi = 2.0f * PI * u0;
	float cosTheta = sqrtf((1.0f - u1) / (1.0f + (a2 - 1.0f) * u1 + 1e-5f) + 1e-5f);
	cosTheta = clamp(cosTheta);
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta + 1e-5f);

	float sinPhi = sinf(phi);
	float cosPhi = cosf(phi);
	vec3 H = normalize(vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
	return H;
}

// based on information from "Physically Based Shading at Disney" by Brent Burley
__host__ __device__ inline float importanceSampleGGXPdf(const vec3 &H, const vec3 &V, float a2)
{
	float NdotH = H.z;//clamp(dot(N, Hw));
	float VdotH = clamp(dot(V, H));
	return (D_GGX(NdotH, a2) * NdotH + 1e-5f) / (4.0f * VdotH + 1e-5f);
}

// http://jcgt.org/published/0007/04/01/paper.pdf "Sampling the GGX Distribution of Visible Normals" by Eric Heitz
__host__ __device__ inline vec3 importanceSampleGGXVNDF(const vec3 &V, float u0, float u1, float a)
{
	const float alpha_x = a;
	const float alpha_y = a;

	// transform the view direction to the hemisphere configuration
	vec3 Vh = normalize(vec3(alpha_x * V.x, alpha_y * V.y, V.z));

	// orthonormal basis (with special case if cross product is zero)
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	vec3 T1 = lensq > 0.0f ? vec3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(lensq)) : vec3(1.0f, 0.0f, 0.0f);
	vec3 T2 = cross(Vh, T1);
	
	// parameterization of the projected area
	float r = sqrtf(u0);
	float phi = 2.0f * PI * u1;
	float t1 = r * cosf(phi);
	float t2 = r * sinf(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
	
	// reprojection onto hemisphere
	vec3 Nh = t1 * T1 + t2 * T2 + sqrtf(clamp(1.0f - t1 * t1 - t2 * t2)) * Vh;
	
	// transforming the normal back to the ellipsoid configuration
	vec3 Ne = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, clamp(Nh.z)));
	
	return Ne;
}

// http://jcgt.org/published/0007/04/01/paper.pdf "Sampling the GGX Distribution of Visible Normals" by Eric Heitz
__host__ __device__ inline float importanceSampleGGXVNDFPdf(const vec3 &H, const vec3 &V, float a)
{
	float a2 = a * a;
	float NdotH = H.z;//clamp(dot(N, Hw));
	float VdotH = clamp(dot(V, H));
	
	float G1 = (2.0f * V.z) / (V.z + sqrtf(a2 + (1.0f - a2) * (V.z * V.z)));
	float Dv = (G1 * VdotH * D_GGX(NdotH, a2)) / V.z;

	return Dv / (4.0f * VdotH);
}