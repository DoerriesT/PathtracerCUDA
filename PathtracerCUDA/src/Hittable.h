#pragma once
#include "Ray.h"
#include "vec3.h"
#include "AABB.h"

class Material;

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

__host__ __device__ inline float ffmin(float a, float b)
{
	return a < b ? a : b;
}

__host__ __device__ inline bool quadratic(float a, float b, float c, float &t0, float &t1)
{
	// find discriminant
	const float discriminant = b * b - 4.0f * a * c;
	if (discriminant < 0.0f)
	{
		return false;
	}
	const float discriminantRoot = sqrtf(discriminant);

	// compute t values
	const float q = b < 0.0f ? -0.5f * (b - discriminantRoot) : -0.5f * (b + discriminantRoot);
	t0 = q / a;
	t1 = c / q;
	if (t0 > t1)
	{
		const float tmp = t0;
		t0 = t1;
		t1 = tmp;
	}
	return true;
}

struct HitRecord
{
	vec3 m_p;
	vec3 m_normal;
	const Material *m_material;
	float m_t;
	bool m_frontFace;

	__host__ __device__ inline void setFaceNormal(const Ray &r, const vec3 &outwardNormal)
	{
		m_frontFace = dot(r.direction(), outwardNormal) < 0.0f;
		m_normal = m_frontFace ? outwardNormal : -outwardNormal;
	}
};

class Material
{
public:
	enum Type : uint32_t
	{
		LAMBERTIAN, METAL, DIELECTRIC
	};

	__host__ __device__ Material(Type type = LAMBERTIAN, const vec3 &albedo = vec3(1.0f), float fuzz = 0.0f, float ior = 1.0f)
		:m_type(type),
		m_albedo(albedo),
		m_fuzz(fuzz),
		m_ior(ior)
	{

	}

	__device__ bool scatter(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered)  const
	{
		switch (m_type)
		{
		case Material::LAMBERTIAN:
			return scatterLambertian(rIn, rec, randState, attenuation, scattered);
		case Material::METAL:
			return scatterMetal(rIn, rec, randState, attenuation, scattered);
		case Material::DIELECTRIC:
			return scatterDielectric(rIn, rec, randState, attenuation, scattered);
		default:
			break;
		}
		return false;
	}

private:
	vec3 m_albedo;
	Type m_type;
	float m_fuzz;
	float m_ior;

	__device__ bool scatterLambertian(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
	{
		vec3 scatterDir = rec.m_normal + random_unit_vec(randState);
		scattered = Ray(rec.m_p, scatterDir);
		attenuation = m_albedo;
		return true;
	}

	__device__ bool scatterMetal(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
	{
		vec3 reflected = reflect(normalize(rIn.direction()), rec.m_normal);
		scattered = Ray(rec.m_p, reflected + m_fuzz * random_in_unit_sphere(randState));
		attenuation = m_albedo;
		return (dot(scattered.direction(), rec.m_normal) > 0.0f);
	}

	__device__ bool scatterDielectric(const Ray &rIn, const HitRecord &rec, curandState &randState, vec3 &attenuation, Ray &scattered) const
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
};

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

	__host__ __device__ Hittable()
		:m_type(SPHERE),
		m_material(),
		m_payload(Sphere{ vec3(0.0f), 1.0f })
	{
	}

	__host__ __device__ Hittable(Type type, const Payload &payload, const Material &material)
		: m_type(type),
		m_material(material),
		m_payload(payload)
	{
	}

	__host__ __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const
	{
		switch (m_type)
		{
		case SPHERE:
			return hitSphere(r, tMin, tMax, rec);
		case CYLINDER:
			return hitCylinder(r, tMin, tMax, rec);
		case DISK:
			return hitDisk(r, tMin, tMax, rec);
		case CONE:
			return hitCone(r, tMin, tMax, rec);
		case PARABOLOID:
			return hitParaboloid(r, tMin, tMax, rec);
		default:
			break;
		}
		return false;
	}

	__host__ __device__ bool boundingBox(AABB &outputBox) const
	{
		switch (m_type)
		{
		case SPHERE:
			return sphereBoundingBox(outputBox);
		case CYLINDER:
			return cylinderBoundingBox(outputBox);
		case DISK:
			return diskBoundingBox(outputBox);
		case CONE:
			return coneBoundingBox(outputBox);
		case PARABOLOID:
			return paraboloidBoundingBox(outputBox);
		default:
			break;
		}
		return false;
	}

private:
	Type m_type;
	Material m_material;
	Payload m_payload;

	__host__ __device__ bool hitSphere(const Ray &r, float tMin, float tMax, HitRecord &rec) const
	{
		const auto &sphere = m_payload.m_sphere;
		vec3 oc = r.origin() - sphere.m_center;
		float a = length_squared(r.direction());
		float b = 2.0f * dot(oc, r.direction());
		float c = length_squared(oc) - sphere.m_radius * sphere.m_radius;

		// solve the quadratic equation
		float t0 = 0.0f;
		float t1 = 0.0f;
		if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
		{
			return false;
		}

		// get the closest t that is greater than tMin
		rec.m_t = t0 > tMin ? t0 : t1;
		rec.m_p = r.at(rec.m_t);
		vec3 outwardNormal = (rec.m_p - sphere.m_center) / sphere.m_radius;
		rec.setFaceNormal(r, outwardNormal);
		rec.m_material = &m_material;
		return true;
	}

	__host__ __device__ bool hitCylinder(const Ray &r, float tMin, float tMax, HitRecord &rec) const
	{
		const auto &cylinder = m_payload.m_cylinder;
		vec3 oc = r.origin() - cylinder.m_center;
		float a = r.m_dir.x * r.m_dir.x + r.m_dir.z * r.m_dir.z;
		float b = 2.0f * (r.m_dir.x * oc.x + r.m_dir.z * oc.z);
		float c = oc.x * oc.x + oc.z * oc.z - cylinder.m_radius * cylinder.m_radius;

		// solve the quadratic equation
		float t0 = 0.0f;
		float t1 = 0.0f;
		if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
		{
			return false;
		}

		// get the closest t that is greater than tMin
		float t = t0 > tMin ? t0 : t1;
		float hitPointHeight = r.m_dir.y * t + oc.y;

		// check cylinder interval and use the second t if possible
		if (hitPointHeight < -cylinder.m_halfHeight || hitPointHeight > cylinder.m_halfHeight)
		{
			if (t == t1)
			{
				return false;
			}
			t = t1;
			// recalculate hit point height...
			hitPointHeight = r.m_dir.y * t + oc.y;
			// ... and check cylinder interval again
			if (hitPointHeight < -cylinder.m_halfHeight || hitPointHeight > cylinder.m_halfHeight)
			{
				return false;
			}
		}
		

		rec.m_t = t;
		rec.m_p = r.at(rec.m_t);
		vec3 outwardNormal = (rec.m_p - vec3(cylinder.m_center.x, rec.m_p.y, cylinder.m_center.z)) / cylinder.m_radius;
		rec.setFaceNormal(r, outwardNormal);
		rec.m_material = &m_material;
		return true;
	}

	__host__ __device__ bool hitDisk(const Ray &r, float tMin, float tMax, HitRecord &rec) const
	{
		// ray is parallel to disk -> no intersection
		if (r.m_dir.y == 0.0f)
		{
			return false;
		}

		const auto &disk = m_payload.m_disk;
		vec3 oc = r.origin() - disk.m_center;

		// intersection t of ray and plane of disk
		float t = (disk.m_center.y - r.m_origin.y) / r.m_dir.y;

		if (t < tMin || t > tMax)
		{
			return false;
		}

		// check that hit point is inside the radius of the disk
		float hitPointX = oc.x + r.m_dir.x * t;
		float hitPointZ = oc.z + r.m_dir.z * t;
		if ((hitPointX * hitPointX + hitPointZ * hitPointZ) >= (disk.m_radius * disk.m_radius))
		{
			return false;
		}

		rec.m_t = t;
		rec.m_p = r.at(t);
		vec3 outwardNormal = vec3(0.0f, 1.0f, 0.0f);
		rec.setFaceNormal(r, outwardNormal);
		rec.m_material = &m_material;
		return true;
	}

	__host__ __device__ bool hitCone(const Ray &r, float tMin, float tMax, HitRecord &rec) const
	{
		const auto &cone = m_payload.m_cone;
		vec3 oc = r.origin() - cone.m_center;

		float k = cone.m_radius / cone.m_height;
		k *= k;
		float a = r.m_dir.x * r.m_dir.x + r.m_dir.z * r.m_dir.z - k * r.m_dir.y * r.m_dir.y;
		float b = 2.0f * (r.m_dir.x * oc.x + r.m_dir.z * oc.z - k * r.m_dir.y * (oc.y - cone.m_height));
		float c = oc.x * oc.x + oc.z * oc.z - k * (oc.y - cone.m_height) * (oc.y - cone.m_height);

		// solve the quadratic equation
		float t0 = 0.0f;
		float t1 = 0.0f;
		if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
		{
			return false;
		}

		// get the closest t that is greater than tMin
		float t = t0 > tMin ? t0 : t1;
		float hitPointHeight = r.m_dir.y * t + oc.y;

		// check cone height
		if (hitPointHeight < 0.0f || hitPointHeight > cone.m_height)
		{
			if (t == t1)
			{
				return false;
			}
			t = t1;
			// test again
			hitPointHeight = r.m_dir.y * t + oc.y;
			if (hitPointHeight < 0.0f || hitPointHeight > cone.m_height)
			{
				return false;
			}
		}

		rec.m_t = t;
		rec.m_p = r.at(rec.m_t);
		
		// calculate normal
		{
			// x, z components of vector from center to p
			float vX = rec.m_p.x - cone.m_center.x;
			float vZ = rec.m_p.z - cone.m_center.z;
			// normalize the vector
			float normFactor = 1.0f / sqrtf(vX * vX + vZ * vZ);
			vX *= normFactor;
			vZ *= normFactor;
			float heightOverRadius = cone.m_height / cone.m_radius;
			vec3 outwardNormal = vec3(vX * heightOverRadius, cone.m_radius / cone.m_height, vZ * heightOverRadius);
			
			rec.setFaceNormal(r, outwardNormal);
		}
		rec.m_material = &m_material;
		return true;
	}

	__host__ __device__ bool hitParaboloid(const Ray &r, float tMin, float tMax, HitRecord &rec) const
	{
		const auto &para = m_payload.m_paraboloid;
		vec3 oc = r.origin() - para.m_center;

		float k = para.m_height / (para.m_radius * para.m_radius);
		float a = k * (r.m_dir.x * r.m_dir.x + r.m_dir.z * r.m_dir.z);
		float b = 2.0f * k * (r.m_dir.x * oc.x + r.m_dir.z * oc.z) - r.m_dir.y;
		float c = k * (oc.x * oc.x + oc.z * oc.z) - oc.y;

		// solve the quadratic equation
		float t0 = 0.0f;
		float t1 = 0.0f;
		if (!quadratic(a, b, c, t0, t1) || t0 > tMax || t1 <= tMin)
		{
			return false;
		}

		// get the closest t that is greater than tMin
		float t = t0 > tMin ? t0 : t1;
		float hitPointHeight = r.m_dir.y * t + oc.y;

		// check paraboloid height
		if (hitPointHeight < 0.0f || hitPointHeight > para.m_height)
		{
			if (t == t1)
			{
				return false;
			}
			t = t1;
			// test again
			hitPointHeight = r.m_dir.y * t + oc.y;
			if (hitPointHeight < 0.0f || hitPointHeight > para.m_height)
			{
				return false;
			}
		}

		rec.m_t = t;
		rec.m_p = r.at(rec.m_t);

		// calculate normal
		{
			vec3 localHit = oc + t * r.m_dir;
			const float phiMax = 2.0f * PI;
			vec3 ddx = vec3(-phiMax * localHit.z, 0.0f, phiMax * localHit.x);
			vec3 ddy = para.m_height * vec3(localHit.x / (2.0f * localHit.y), 1.0f, localHit.z / (2.0f * localHit.y));


			vec3 outwardNormal = normalize(cross(ddx, ddy));

			rec.setFaceNormal(r, outwardNormal);
		}
		rec.m_material = &m_material;
		return true;
	}

	__host__ __device__ bool sphereBoundingBox(AABB &outputBox) const
	{
		const auto &sphere = m_payload.m_sphere;
		outputBox = AABB(
			sphere.m_center - vec3(sphere.m_radius, sphere.m_radius, sphere.m_radius),
			sphere.m_center + vec3(sphere.m_radius, sphere.m_radius, sphere.m_radius)
		);
		return true;
	}

	__host__ __device__ bool cylinderBoundingBox(AABB &outputBox) const
	{
		const auto &cylinder = m_payload.m_cylinder;
		outputBox = AABB(
			cylinder.m_center - vec3(cylinder.m_radius, cylinder.m_halfHeight, cylinder.m_radius),
			cylinder.m_center + vec3(cylinder.m_radius, cylinder.m_halfHeight, cylinder.m_radius)
		);
		return true;
	}

	__host__ __device__ bool diskBoundingBox(AABB &outputBox) const
	{
		const auto &disk = m_payload.m_disk;
		outputBox = AABB(
			disk.m_center - vec3(disk.m_radius, -FLT_EPSILON, disk.m_radius),
			disk.m_center + vec3(disk.m_radius, FLT_EPSILON, disk.m_radius)
		);
		return true;
	}

	__host__ __device__ bool coneBoundingBox(AABB &outputBox) const
	{
		const auto &cone = m_payload.m_cone;
		outputBox = AABB(
			cone.m_center - vec3(cone.m_radius, 0.0f, cone.m_radius),
			cone.m_center + vec3(cone.m_radius, cone.m_height, cone.m_radius)
		);
		return true;
	}

	__host__ __device__ bool paraboloidBoundingBox(AABB &outputBox) const
	{
		const auto &para = m_payload.m_paraboloid;
		outputBox = AABB(
			para.m_center - vec3(para.m_radius, 0.0f, para.m_radius),
			para.m_center + vec3(para.m_radius, para.m_height, para.m_radius)
		);
		return true;
	}
};