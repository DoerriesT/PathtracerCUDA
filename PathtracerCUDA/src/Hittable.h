#pragma once
#include "vec3.h"
#include "Material.h"

struct HitRecord;
class AABB;
struct Ray;

class Hittable
{
public:
	enum Type : uint32_t
	{
		SPHERE, CYLINDER, DISK, CONE, PARABOLOID, QUAD, CUBE
	};

	__host__ __device__ Hittable();
	__host__ __device__ Hittable(Type type, const vec3 &position, const vec3 &rotation, const vec3 &scale, const Material2 &material);
	__host__ __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const;
	bool boundingBox(AABB &outputBox) const;

private:
	float4 m_invTransformRow0;
	float4 m_invTransformRow1;
	float4 m_invTransformRow2;
	Type m_type;
	Material2 m_material;
	AABB m_aabb;

	// intersection functions
	__host__ __device__ bool hitSphere(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
	__host__ __device__ bool hitCylinder(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
	__host__ __device__ bool hitDisk(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
	__host__ __device__ bool hitCone(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
	__host__ __device__ bool hitParaboloid(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
	__host__ __device__ bool hitQuad(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
	__host__ __device__ bool hitBox(const Ray &r, float tMin, float tMax, float &t, vec3 &normal) const;
};


// IMPLEMENTATION
#include "Hittable.inl"