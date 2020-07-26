#include "trace.h"
#include "BVH.h"
#include "Material.h"
#include "Hittable.h"


__device__ bool hitList(uint32_t hittableCount, Hittable *world, const Ray &r, float t_min, float t_max, HitRecord &rec)
{
	HitRecord tempRec;
	bool hitAnything = false;
	auto closest = t_max;

	for (uint32_t i = 0; i < hittableCount; ++i)
	{
		if (world[i].hit(r, t_min, closest, tempRec))
		{
			hitAnything = true;
			closest = tempRec.m_t;
			rec = tempRec;
		}
	}

	return hitAnything;
}

__device__ bool hitBVH(uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, const Ray &r, float t_min, float t_max, HitRecord &rec)
{
	vec3 invRayDir;
	invRayDir[0] = 1.0f / (r.m_dir[0] != 0.0f ? r.m_dir[0] : pow(2.0f, -80.0f));
	invRayDir[1] = 1.0f / (r.m_dir[1] != 0.0f ? r.m_dir[1] : pow(2.0f, -80.0f));
	invRayDir[2] = 1.0f / (r.m_dir[2] != 0.0f ? r.m_dir[2] : pow(2.0f, -80.0f));
	vec3 originDivDir = r.m_origin * invRayDir;
	bool dirIsNeg[3] = { invRayDir.x < 0.0f, invRayDir.y < 0.0f, invRayDir.z < 0.0f };

	uint32_t nodesToVisit[32];
	uint32_t toVisitOffset = 0;
	uint32_t currentNodeIndex = 0;

	uint32_t elemIdx = UINT32_MAX;
	uint32_t iterations = 0;

	while (true)
	{
		++iterations;
		const BVHNode &node = bvhNodes[currentNodeIndex];
		if (node.m_aabb.hit(r, t_min, t_max))
		{
			const uint32_t primitiveCount = (node.m_primitiveCountAxis >> 16);
			if (primitiveCount > 0)
			{
				for (uint32_t i = 0; i < primitiveCount; ++i)
				{
					Hittable &elem = world[node.m_offset + i];
					if (elem.hit(r, t_min, t_max, rec))
					{
						t_max = rec.m_t;
						elemIdx = node.m_offset + i;
					}
				}
				if (toVisitOffset == 0)
				{
					break;
				}
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else
			{
				if (dirIsNeg[(node.m_primitiveCountAxis >> 8) & 0xFF])
				{
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node.m_offset;
				}
				else
				{
					nodesToVisit[toVisitOffset++] = node.m_offset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else
		{
			if (toVisitOffset == 0)
			{
				break;
			}
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	return elemIdx != UINT32_MAX;
}

__device__ vec3 getColor(const Ray &r, uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, curandState &randState)
{
	vec3 beta = vec3(1.0f);
	vec3 L = vec3(0.0f);
	Ray ray = r;
	for (int iteration = 0; iteration < 5; ++iteration)
	{
		HitRecord rec;
		bool foundIntersection = hitBVH(hittableCount, world, bvhNodesCount, bvhNodes, ray, 0.001f, FLT_MAX, rec);

		// add sky light and exit loop
		if (!foundIntersection)
		{
			vec3 unitDir = normalize(ray.m_dir);
			float t = unitDir.y * 0.5f + 0.5f;
			vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
			//c = 0.0f;
			L += beta * c;
			break;
		}
		// process intersection
		else
		{
			// add emitted light
			L += beta * rec.m_material->getEmitted(ray, rec);

			// scatter
			Ray scattered;
			float pdf = 0.0f;
			vec3 attenuation = rec.m_material->sample(ray, rec, randState, scattered, pdf);
			if (attenuation == vec3(0.0f) || pdf == 0.0f)
			{
				break;
			}
			beta *= attenuation * abs(dot(normalize(scattered.m_dir), normalize(rec.m_normal))) / pdf;
			ray = scattered;
		}

		//if (hitBVH(hittableCount, world, bvhNodesCount, bvhNodes, ray, 0.001f, FLT_MAX, rec))
		////if (hitList(hittableCount, world, ray, 0.001f, FLT_MAX, rec))
		//{
		//	Ray scattered;
		//	vec3 attenuation;
		//	if (rec.m_material->scatter(ray, rec, randState, attenuation, scattered))
		//	{
		//		beta *= attenuation;
		//		ray = scattered;
		//	}
		//	else
		//	{
		//		return vec3(0.0f, 0.0f, 0.0f);
		//	}
		//}
		//else
		//{
		//	vec3 unitDir = normalize(ray.m_dir);
		//	float t = unitDir.y * 0.5f + 0.5f;
		//	vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
		//	return c * beta;
		//}
	}

	return L;
}

__global__ void traceKernel(uchar4 *resultBuffer, float4 *accumBuffer, bool ignoreHistory, uint32_t frame, uint32_t width, uint32_t height, uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, curandState *randState, Camera camera)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	const uint32_t dstIdx = threadIDx + threadIDy * width;

	float4 inputColor4 = accumBuffer[dstIdx];
	vec3 inputColor(inputColor4.x, inputColor4.y, inputColor4.z);

	curandState &localRandState = randState[dstIdx];

	float u = (threadIDx + curand_uniform(&localRandState)) / float(width);
	float v = (threadIDy + curand_uniform(&localRandState)) / float(height);
	Ray r = camera.getRay(u, v, localRandState);
	vec3 color = getColor(r, hittableCount, world, bvhNodesCount, bvhNodes, localRandState);

	color = ignoreHistory ? color : color + inputColor;

	vec3 resultColor = color / float(frame + 1.0f);
	resultColor = saturate(resultColor);
	resultColor.r = sqrt(resultColor.r);
	resultColor.g = sqrt(resultColor.g);
	resultColor.b = sqrt(resultColor.b);

	accumBuffer[dstIdx] = { color.r, color.g, color.b, 1.0f };
	resultBuffer[dstIdx] = { (unsigned char)(resultColor.x * 255.0f), (unsigned char)(resultColor.y * 255.0f) , (unsigned char)(resultColor.z * 255.0f), 255 };
}