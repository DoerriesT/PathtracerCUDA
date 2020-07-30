#pragma once
#include "vec3.h"
#include "Material.h"
#include "AABB.h"

class Hittable;
struct HitRecord;
class AABB;
struct Ray;

enum class HittableType : uint32_t
{
	SPHERE, CYLINDER, DISK, CONE, PARABOLOID, QUAD, CUBE
};

class Hittable
{
public:
	__host__ __device__ Hittable();
	__host__ __device__ Hittable(HittableType type, const float4 *invTransformRows, const Material2 &material);
	__host__ __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const;

private:
	float4 m_invTransformRow0;
	float4 m_invTransformRow1;
	float4 m_invTransformRow2;
	Material2 m_material;
	HittableType m_type;

	// intersection functions
	__host__ __device__ bool hitSphere(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
	__host__ __device__ bool hitCylinder(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
	__host__ __device__ bool hitDisk(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
	__host__ __device__ bool hitCone(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
	__host__ __device__ bool hitParaboloid(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
	__host__ __device__ bool hitQuad(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
	__host__ __device__ bool hitBox(const Ray &r, float tMin, float tMax, float &t, vec3 &normal, float &u, float &v) const;
};


// contains data necessary for building the BVH, but not required for traversing it on the gpu
class CpuHittable
{
public:
	explicit CpuHittable();
	explicit CpuHittable(HittableType type, const vec3 &position, const vec3 &rotation, const vec3 &scale, const Material2 &material);
	const AABB &getAABB() const;
	Hittable getGpuHittable() const;

private:
	float4 m_invTransformRow0;
	float4 m_invTransformRow1;
	float4 m_invTransformRow2;
	Material2 m_material;
	AABB m_aabb;
	HittableType m_type;
};


// IMPLEMENTATION
#include "Hittable.inl"