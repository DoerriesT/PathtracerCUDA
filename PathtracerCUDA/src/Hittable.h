#pragma once
#include "vec3.h"
#include "Material.h"

struct HitRecord;
class AABB;
struct Ray;

struct Sphere
{
	vec3 m_center;
	float m_radius;
};

struct Cylinder
{
	vec3 m_center;
	float m_radius;
	float m_halfHeight;
};

struct Disk
{
	vec3 m_center;
	float m_radius;
};

struct Cone
{
	vec3 m_center;
	float m_radius;
	float m_height;
};

struct Paraboloid
{
	vec3 m_center;
	float m_radius;
	float m_height;
};

class Hittable
{
public:
	enum Type : uint32_t
	{
		SPHERE, CYLINDER, DISK, CONE, PARABOLOID
	};

	union Payload
	{
		Sphere m_sphere;
		Cylinder m_cylinder;
		Disk m_disk;
		Cone m_cone;
		Paraboloid m_paraboloid;

		__host__ __device__ explicit Payload() { }
		__host__ __device__ explicit Payload(const Sphere &sphere) : m_sphere(sphere) { }
		__host__ __device__ explicit Payload(const Cylinder &cylinder) : m_cylinder(cylinder) { }
		__host__ __device__ explicit Payload(const Disk &disk) : m_disk(disk) { }
		__host__ __device__ explicit Payload(const Cone &cone) : m_cone(cone) { }
		__host__ __device__ explicit Payload(const Paraboloid &paraboloid) : m_paraboloid(paraboloid) { }
	};

	__host__ __device__ Hittable();
	__host__ __device__ Hittable(Type type, const Payload &payload, const Material2 &material);
	__host__ __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const;
	__host__ __device__ bool boundingBox(AABB &outputBox) const;

private:
	Type m_type;
	Material2 m_material;
	Payload m_payload;

	// intersection functions
	__host__ __device__ bool hitSphere(const Ray &r, float tMin, float tMax, HitRecord &rec) const;
	__host__ __device__ bool hitCylinder(const Ray &r, float tMin, float tMax, HitRecord &rec) const;
	__host__ __device__ bool hitDisk(const Ray &r, float tMin, float tMax, HitRecord &rec) const;
	__host__ __device__ bool hitCone(const Ray &r, float tMin, float tMax, HitRecord &rec) const;
	__host__ __device__ bool hitParaboloid(const Ray &r, float tMin, float tMax, HitRecord &rec) const;

	// bounding box functions
	__host__ __device__ bool sphereBoundingBox(AABB &outputBox) const;
	__host__ __device__ bool cylinderBoundingBox(AABB &outputBox) const;
	__host__ __device__ bool diskBoundingBox(AABB &outputBox) const;
	__host__ __device__ bool coneBoundingBox(AABB &outputBox) const;
	__host__ __device__ bool paraboloidBoundingBox(AABB &outputBox) const;
};


// IMPLEMENTATION
#include "Hittable.inl"