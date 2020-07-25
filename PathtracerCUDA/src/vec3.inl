#pragma once
#include "vec3.h"

__host__ __device__ inline vec3::vec3()
	: e{ 0,0,0 }
{
}

__host__ __device__ inline vec3::vec3(float e0, float e1, float e2)
	: e{ e0, e1, e2 }
{
}

__host__ __device__ inline vec3::vec3(float e)
	: e{ e, e, e }
{
}

__host__ __device__ inline vec3 vec3::operator-() const
{
	return vec3(-e[0], -e[1], -e[2]);
}

__host__ __device__ inline float vec3::operator[](int i) const
{
	return e[i];
}

__host__ __device__ inline float &vec3::operator[](int i)
{
	return e[i];
}

__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &t)
{
	e[0] *= t.x;
	e[1] *= t.y;
	e[2] *= t.z;
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const float t)
{
	return *this *= 1 / t;
}




inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3 &u, float v)
{
	return vec3(u.e[0] + v, u.e[1] + v, u.e[2] + v);
}

__host__ __device__ inline vec3 operator+(float u, const vec3 &v)
{
	return v + u;
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, float v)
{
	return vec3(u.e[0] - v, u.e[1] - v, u.e[2] - v);
}

__host__ __device__ inline vec3 operator-(const float u, const vec3 &v)
{
	return -v + u;
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
	return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, const vec3 &t)
{
	return vec3(1.0f / t.x, 1.0f / t.y, 1.0f / t.z) * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t)
{
	return (1.0f / t) * v;
}

__host__ __device__ inline vec3 operator/(float v, const vec3 &t)
{
	return vec3(1.0f / t.x, 1.0f / t.y, 1.0f / t.z) * v;
}

__host__ __device__ inline bool operator==(const vec3 &u, const vec3 &v)
{
	return u.x == v.x && u.y == v.y && u.z == v.z;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline float length_squared(vec3 v)
{
	return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2];
}

__host__ __device__ inline float length(vec3 v)
{
	return sqrt(length_squared(v));
}

__host__ __device__ inline vec3 min(const vec3 &a, const vec3 &b)
{
	vec3 x = a;
	x.x = x.x < b.x ? x.x : b.x;
	x.y = x.y < b.y ? x.y : b.y;
	x.z = x.z < b.z ? x.z : b.z;
	return x;
}

__host__ __device__ inline vec3 max(const vec3 &a, const vec3 &b)
{
	vec3 x = a;
	x.x = x.x >= b.x ? x.x : b.x;
	x.y = x.y >= b.y ? x.y : b.y;
	x.z = x.z >= b.z ? x.z : b.z;
	return x;
}

__host__ __device__ inline vec3 normalize(vec3 v)
{
	return v / length(v);
}

__host__ __device__ inline vec3 reflect(const vec3 &v, const vec3 &n)
{
	return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3 &uv, const vec3 &n, float etaiOverEtat)
{
	auto cos_theta = dot(-uv, n);
	vec3 rOutParallel = etaiOverEtat * (uv + cos_theta * n);
	vec3 rOutPerp = -sqrt(1.0f - length_squared(rOutParallel)) * n;
	return rOutParallel + rOutPerp;
}

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
__host__ __device__ inline vec3 rotateAroundVector(const vec3 &v, const vec3 &axis, float cosAngle, float sinAngle)
{
	return v * cosAngle + cross(axis, v) * sinAngle + axis * dot(axis, v) * (1.0f - cosAngle);
}

__host__ __device__ inline vec3 rotateAroundVector(const vec3 &v, const vec3 &axis, float angle)
{
	const float cosAngle = cos(angle);
	const float sinAngle = sin(angle);
	return rotateAroundVector(v, axis, cosAngle, sinAngle);
}

__device__ inline vec3 random_unit_vec(curandState &randState)
{
	auto a = curand_uniform(&randState) * 2.0f * 3.14159265358979323846f;
	auto z = curand_uniform(&randState) * 2.0f - 1.0f;
	auto r = sqrt(1.0f - z * z);
	return vec3(r * cos(a), r * sin(a), z);
}

__device__ vec3 inline random_in_unit_sphere(curandState &randState)
{
	vec3 p;
	do
	{
		p = vec3(curand_uniform(&randState), curand_uniform(&randState), curand_uniform(&randState)) * 2.0f - 1.0f;
	} while (length_squared(p) >= 1.0f);
	return p;
}

__device__ vec3 inline random_in_unit_disk(curandState &randState)
{
	vec3 p;
	do
	{
		p = vec3(curand_uniform(&randState), curand_uniform(&randState), 0.0f) * 2.0f - vec3(1.0f, 1.0f, 0.0f);
	} while (length_squared(p) >= 1.0f);
	return p;
}

__device__ vec3 inline random_vec(curandState &randState)
{
	return vec3(curand_uniform(&randState), curand_uniform(&randState), curand_uniform(&randState));
}

__host__ __device__ inline vec3 lerp(const vec3 &x, const vec3 &y, const vec3 &a)
{
	return x * (1.0f - a) + y * a;
}

__host__ __device__ inline vec3 lerp(const vec3 &x, const vec3 &y, float a)
{
	return x * (1.0f - a) + y * a;
}

__host__ __device__ inline float lerp(float x, float y, float a)
{
	return x * (1.0f - a) + y * a;
}

__device__ inline vec3 random_in_hemisphere(const vec3 &normal, curandState &randState)
{
	vec3 in_unit_sphere = normalize(random_unit_vec(randState));
	return dot(in_unit_sphere, normal) > 0.0 ? in_unit_sphere : -in_unit_sphere;
}

__host__ __device__ inline vec3 tangentToWorld(const vec3 &N, const vec3 &v)
{
	vec3 up = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	return normalize(tangent * v.x + bitangent * v.y + N * v.z);
}

__host__ __device__ inline vec3 cosineSampleHemisphere(float u0, float u1, float &pdf)
{
	const float phi = 2.0f * PI * u0;
	const float cosTheta = sqrt(u1);
	const float sinTheta = sqrt(1.0f - u1);
	pdf = cosTheta * (1.0f / PI);
	return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

__host__ __device__ inline float clamp(float x, float a, float b)
{
	x = x < a ? a : x;
	x = x > b ? b : x;
	return x;
}

__host__ __device__ inline vec3 clamp(vec3 x, vec3 a, vec3 b)
{
	return vec3(clamp(x.x, 0.0f, 1.0f), clamp(x.y, 0.0f, 1.0f), clamp(x.z, 0.0f, 1.0f));
}

__host__ __device__ inline vec3 saturate(vec3 x)
{
	return clamp(x, vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f));
}

__host__ __device__ inline void worldTransform(const vec3 &position, const vec3 &rotation, const vec3 &scale, float4 *localToWorldRows, float4 *worldToLocalRows)
{
	auto quatToRotMat = [](const float4 &q, float (&rotMat)[3][3])
	{
		float qxx(q.x * q.x);
		float qyy(q.y * q.y);
		float qzz(q.z * q.z);
		float qxz(q.x * q.z);
		float qxy(q.x * q.y);
		float qyz(q.y * q.z);
		float qwx(q.w * q.x);
		float qwy(q.w * q.y);
		float qwz(q.w * q.z);

		rotMat[0][0] = 1.0f - 2.0f * (qyy + qzz);
		rotMat[0][1] = 2.0f * (qxy + qwz);
		rotMat[0][2] = 2.0f * (qxz - qwy);

		rotMat[1][0] = 2.0f * (qxy - qwz);
		rotMat[1][1] = 1.0f - 2.0f * (qxx + qzz);
		rotMat[1][2] = 2.0f * (qyz + qwx);

		rotMat[2][0] = 2.0f * (qxz + qwy);
		rotMat[2][1] = 2.0f * (qyz - qwx);
		rotMat[2][2] = 1.0f - 2.0f * (qxx + qyy);
	};

	// compute quaternion representing the rotation
	float4 q;
	{
		vec3 c = vec3(cos(rotation.x * 0.5f), cos(rotation.y * 0.5f), cos(rotation.z * 0.5f));
		vec3 s = vec3(sin(rotation.x * 0.5f), sin(rotation.y * 0.5f), sin(rotation.z * 0.5f));

		q.w = c.x * c.y * c.z + s.x * s.y * s.z;
		q.x = s.x * c.y * c.z - c.x * s.y * s.z;
		q.y = c.x * s.y * c.z + s.x * c.y * s.z;
		q.z = c.x * c.y * s.z - s.x * s.y * c.z;
	}

	// quaternion representing the inverse rotation
	float4 invQ;
	{
		float invDot = (1.0f / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w));

		// conjugate
		float4 tmp = { -q.x, -q.y, -q.z, q.w };

		invQ.x = tmp.x * invDot;
		invQ.y = tmp.y * invDot;
		invQ.z = tmp.z * invDot;
		invQ.w = tmp.w * invDot;
	}

	// compute invScale * invRot * invTrans
	{
		// compute inverse rotation matrix
		float invRotMat[3][3];
		quatToRotMat(invQ, invRotMat);

		vec3 invScale = 1.0f / scale;
		worldToLocalRows[0].x = invScale.x * invRotMat[0][0];
		worldToLocalRows[0].y = invScale.x * invRotMat[1][0];
		worldToLocalRows[0].z = invScale.x * invRotMat[2][0];
		worldToLocalRows[0].w = invScale.x * dot(vec3(invRotMat[0][0], invRotMat[1][0], invRotMat[2][0]), -position);

		worldToLocalRows[1].x = invScale.y * invRotMat[0][1];
		worldToLocalRows[1].y = invScale.y * invRotMat[1][1];
		worldToLocalRows[1].z = invScale.y * invRotMat[2][1];
		worldToLocalRows[1].w = invScale.y * dot(vec3(invRotMat[0][1], invRotMat[1][1], invRotMat[2][1]), -position);

		worldToLocalRows[2].x = invScale.z * invRotMat[0][2];
		worldToLocalRows[2].y = invScale.z * invRotMat[1][2];
		worldToLocalRows[2].z = invScale.z * invRotMat[2][2];
		worldToLocalRows[2].w = invScale.z * dot(vec3(invRotMat[0][2], invRotMat[1][2], invRotMat[2][2]), -position);
	}
	
	// compute trans * rot * scale;
	{
		// compute rotation matrix
		float rotMat[3][3];
		quatToRotMat(q, rotMat);

		localToWorldRows[0].x = scale.x * rotMat[0][0];
		localToWorldRows[0].y = scale.y * rotMat[1][0];
		localToWorldRows[0].z = scale.z * rotMat[2][0];
		localToWorldRows[0].w = position.x;

		localToWorldRows[1].x = scale.x * rotMat[0][1];
		localToWorldRows[1].y = scale.y * rotMat[1][1];
		localToWorldRows[1].z = scale.z * rotMat[2][1];
		localToWorldRows[1].w = position.y;

		localToWorldRows[2].x = scale.x * rotMat[0][2];
		localToWorldRows[2].y = scale.y * rotMat[1][2];
		localToWorldRows[2].z = scale.z * rotMat[2][2];
		localToWorldRows[2].w = position.z;
	}
}
