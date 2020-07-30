#include "trace.h"
#include "./../BVH.h"
#include "./../Material.h"
#include "./../Hittable.h"


// iterate through all objects in the scene and find the closest intersection
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

// traverse the BVH and find the closest intersection
__device__ bool hitBVH(uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, const Ray &r, float t_min, float t_max, HitRecord &rec)
{
	// we are going to need to divide by the ray direction a lot, so compute its inverse once and use multiplications instead
	vec3 invRayDir;
	invRayDir[0] = 1.0f / (r.m_dir[0] != 0.0f ? r.m_dir[0] : 1e-7f);
	invRayDir[1] = 1.0f / (r.m_dir[1] != 0.0f ? r.m_dir[1] : 1e-7f);
	invRayDir[2] = 1.0f / (r.m_dir[2] != 0.0f ? r.m_dir[2] : 1e-7f);
	vec3 originDivDir = r.m_origin * invRayDir;
	bool dirIsNeg[3] = { invRayDir.x < 0.0f, invRayDir.y < 0.0f, invRayDir.z < 0.0f };

	// we dont use recursion to traverse the tree, so keep a stack of nodes that still need to be visited
	uint32_t nodesToVisit[32];
	uint32_t toVisitOffset = 0;
	uint32_t currentNodeIndex = 0;

	uint32_t elemIdx = UINT32_MAX;
	uint32_t iterations = 0;

	while (true)
	{
		++iterations;
		// intersect AABB of current node
		const BVHNode &node = bvhNodes[currentNodeIndex];
		if (node.m_aabb.hit(r, t_min, t_max))
		{
			// if primitiveCount is > 0, this is a leaf node -> intersect all primitives of the leaf node
			const uint32_t primitiveCount = (node.m_primitiveCountAxis >> 16);
			if (primitiveCount > 0)
			{
				for (uint32_t i = 0; i < primitiveCount; ++i)
				{
					if (world[node.m_offset + i].hit(r, t_min, t_max, rec))
					{
						t_max = rec.m_t;
						elemIdx = node.m_offset + i;
					}
				}
				// if our stack is empty, exit the loop
				if (toVisitOffset == 0)
				{
					break;
				}
				// otherwise pop the top element of the stack and visit it in the next iteration
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			// we hit an interior node
			else
			{
				// we need to decide which of the children we visit next and which is put on the stack.
				// we do this by visiting the child whose bounding box is closer to the ray origin first
				bool isNeg = dirIsNeg[(node.m_primitiveCountAxis >> 8) & 0xFF];

				nodesToVisit[toVisitOffset++] = isNeg ? (currentNodeIndex + 1) : node.m_offset;
				currentNodeIndex = isNeg ? node.m_offset : (currentNodeIndex + 1);
			}
		}
		// no intersection with the current node
		else
		{
			// if the stack is empty, we are done -> exit the loop
			if (toVisitOffset == 0)
			{
				break;
			}
			// otherwise pop the top element of the stack and visit it in the next iteration
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	return elemIdx != UINT32_MAX;
}

// traces a path starting with given ray r in the scene
__device__ vec3 getColor(const Ray &r, uint32_t hittableCount, Hittable *world, uint32_t bvhNodesCount, BVHNode *bvhNodes, curandState &randState, uint32_t skyboxTextureHandle, cudaTextureObject_t *textures)
{
	// keep track of path throughput and light reaching the camera
	vec3 throughput = vec3(1.0f);
	vec3 L = vec3(0.0f);
	Ray ray = r;

	// 5 iterations seem to be good enough
	for (int iteration = 0; iteration < 5; ++iteration)
	{
		HitRecord rec;
		bool foundIntersection = hitBVH(hittableCount, world, bvhNodesCount, bvhNodes, ray, 0.001f, FLT_MAX, rec);

		// add sky light and exit loop
		if (!foundIntersection)
		{
			vec3 c = 0.0f;// lerp(vec3(1.0f), vec3(0.5f, 0.7f, 1.0f), ray.m_dir.y * 0.5f + 0.5f);
			
			// optionally sample a skybox texture
			if (skyboxTextureHandle != 0)
			{
				// compute spherical coordinates from ray direction
				float theta = acos(ray.m_dir.y);
				float phi = atan2(ray.m_dir.z, ray.m_dir.x);
				// compute texture coordinates from spherical coordinates
				float v = theta / PI;
				float u = phi / (2.0f * PI);
				float4 sky = tex2D<float4>(textures[skyboxTextureHandle - 1], u, v);
				c = vec3(sky.x, sky.y, sky.z);
			}

			L += throughput * c;
			break;
		}
		// process intersection
		else
		{
			// add emitted light
			L += throughput * rec.m_material->getEmitted(ray, rec);

			// scatter ray
			Ray scattered;
			float pdf = 0.0f;
			vec3 attenuation = rec.m_material->sample(ray, rec, randState, scattered, pdf, textures);
			if (attenuation == vec3(0.0f) || pdf == 0.0f)
			{
				break;
			}
			// weight attenuation bei cosine term and divide by pdf
			throughput *= attenuation * abs(dot(scattered.m_dir, rec.m_normal)) / pdf;
			ray = scattered;
		}
	}

	return L;
}

__global__ void traceKernel(
	float4 *accumBuffer,
	bool ignoreHistory,
	uint32_t accumulatedSampleCount,
	uint32_t width,
	uint32_t height,
	uint32_t spp,
	uint32_t hittableCount,
	Hittable *world,
	uint32_t bvhNodesCount,
	BVHNode *bvhNodes,
	curandState *randState,
	Camera camera,
	uint32_t skyboxTextureHandle,
	cudaTextureObject_t *textures)
{
	int threadIDx = threadIdx.x + blockIdx.x * blockDim.x;
	int threadIDy = threadIdx.y + blockIdx.y * blockDim.y;

	if (threadIDx >= width || threadIDy >= height)
	{
		return;
	}

	const uint32_t dstIdx = threadIDx + threadIDy * width;
	curandState &localRandState = randState[dstIdx];

	// trace rays into the scene and accumulate the results
	vec3 color = 0.0f;
	for (uint32_t i = 0; i < spp; ++i)
	{
		// we randomly place the rays inside the pixel footprint to achieve anti-aliasing
		float u = (threadIDx + curand_uniform(&localRandState)) / float(width);
		float v = (threadIDy + curand_uniform(&localRandState)) / float(height);
		Ray r = camera.getRay(u, v);
		color += getColor(r, hittableCount, world, bvhNodesCount, bvhNodes, localRandState, skyboxTextureHandle, textures);
	}

	color = ignoreHistory ? color : (color + vec3(accumBuffer[dstIdx].x, accumBuffer[dstIdx].y, accumBuffer[dstIdx].z));

	accumBuffer[dstIdx] = { color.r, color.g, color.b, 1.0f };
}