#include "BVH.h"
#include <algorithm>
#include <cassert>

void BVH::build(size_t elementCount, const CpuHittable *elements, uint32_t maxLeafElements)
{
	m_maxLeafElements = maxLeafElements;
	m_nodes.clear();
	m_nodes.reserve(elementCount - 1 + elementCount);
	m_elements.clear();
	m_elements.resize(elementCount);
	memcpy(m_elements.data(), elements, elementCount * sizeof(CpuHittable));

	buildRecursive(0, elementCount);
}

const std::vector<BVHNode> &BVH::getNodes() const
{
	return m_nodes;
}

const std::vector<CpuHittable> &BVH::getElements() const
{
	return m_elements;
}

uint32_t BVH::getDepth(uint32_t node) const
{
	if ((m_nodes[node].m_primitiveCountAxis >> 16))
	{
		return 1;
	}
	return 1 + std::max(getDepth(node + 1), getDepth(m_nodes[node].m_offset));
}

bool BVH::validate()
{
	bool *reachedLeaves = new bool[m_elements.size()];
	memset(reachedLeaves, 0, m_elements.size() * sizeof(bool));
	bool ret = validateRecursive(0, reachedLeaves);
	for (size_t i = 0; i < m_elements.size(); ++i)
	{
		if (!reachedLeaves[i])
		{
			delete[] reachedLeaves;
			return false;
		}
	}
	delete[] reachedLeaves;
	return ret;
}

bool BVH::trace(const vec3 &origin, const vec3 &dir, float &t)
{
	return false;
}

static float calcSurfaceArea(const vec3 &minCorner, const vec3 &maxCorner)
{
	const vec3 extent = maxCorner - minCorner;
	if (extent.x <= 0.0f || extent.y <= 0.0f || extent.z <= 0.0f)
	{
		return 0.0f;
	}
	
	return (extent[0] * extent[1] + extent[0] * extent[2] + extent[1] * extent[2]) * 2.0f;
}

uint32_t BVH::buildRecursive(size_t begin, size_t end)
{
	uint32_t nodeIndex = static_cast<uint32_t>(m_nodes.size());
	m_nodes.push_back({});
	BVHNode node = {};
	node.m_aabb.m_min = vec3(std::numeric_limits<float>::max());
	node.m_aabb.m_max = vec3(std::numeric_limits<float>::lowest());

	// compute node aabb
	for (size_t i = begin; i < end; ++i)
	{
		node.m_aabb = AABB(node.m_aabb, m_elements[i].getAABB());
	}

	if (end - begin > m_maxLeafElements)
	{
		struct BinInfo
		{
			uint32_t m_count = 0;
			AABB m_aabb = AABB(vec3(std::numeric_limits<float>::max()), vec3(std::numeric_limits<float>::lowest()));
		};
		BinInfo bins[3][8];

		const auto nodeAabbExtent = max(node.m_aabb.m_max - node.m_aabb.m_min, vec3(0.00000001f));

		for (size_t i = begin; i < end; ++i)
		{
			AABB elemAabb = m_elements[i].getAABB();

			const vec3 centroid = (elemAabb.m_min + elemAabb.m_max) * 0.5f;
			const vec3 relativeOffset = (centroid - node.m_aabb.m_min) / nodeAabbExtent;

			for (int j = 0; j < 3; ++j)
			{
				int binIdx = int(relativeOffset[j] * 8.0f);
				binIdx = binIdx < 0 ? 0 : binIdx > 7 ? 7 : binIdx;
				auto &bin = bins[j][binIdx];
				bin.m_count += 1;

				bin.m_aabb = AABB(bin.m_aabb, elemAabb);
			}

		}

		const float invNodeSurfaceArea = 1.0f / std::max(calcSurfaceArea(node.m_aabb.m_min, node.m_aabb.m_max), 0.000000001f);
		float lowestCost = std::numeric_limits<float>::max();
		uint32_t bestAxis = 0;
		uint32_t bestBin = 0;

		float costs[3][8];
		for (uint32_t i = 0; i < 3; ++i)
		{
			for (uint32_t j = 0; j < 7; ++j)
			{
				AABB aabb0 = AABB(vec3(std::numeric_limits<float>::max()), vec3(std::numeric_limits<float>::lowest()));
				AABB aabb1 = AABB(vec3(std::numeric_limits<float>::max()), vec3(std::numeric_limits<float>::lowest()));

				uint32_t count0 = 0;
				uint32_t count1 = 0;
				for (uint32_t k = 0; k <= j; ++k)
				{
					aabb0 = AABB(aabb0, bins[i][k].m_aabb);
					count0 += bins[i][k].m_count;
				}
				for (uint32_t k = j + 1; k < 8; ++k)
				{
					aabb1 = AABB(aabb1, bins[i][k].m_aabb);
					count1 += bins[i][k].m_count;
				}
				float a0 = calcSurfaceArea(aabb0.m_min, aabb0.m_max);
				float a1 = calcSurfaceArea(aabb1.m_min, aabb1.m_max);
				float cost = 0.125f + (count0 * a0 + count1 * a1) * invNodeSurfaceArea;
				cost = count0 == 0 || count1 == 0 ? std::numeric_limits<float>::max() : cost; // avoid 0/N partitions
				costs[i][j] = cost;
				assert(cost == cost);
				if (cost < lowestCost)
				{
					lowestCost = cost;
					bestAxis = i;
					bestBin = j;
				}
			}
		}

		const CpuHittable *pMid = std::partition(m_elements.data() + begin, m_elements.data() + end, [&](const auto &item)
			{
				AABB elemAabb = item.getAABB();
				const vec3 centroid = (elemAabb.m_min + elemAabb.m_max) * 0.5f;
				float relativeOffset = ((centroid[bestAxis] - node.m_aabb.m_min[bestAxis]) / nodeAabbExtent[bestAxis]);
				int bucketIdx = int(8.0f * relativeOffset);
				bucketIdx = bucketIdx < 0 ? 0 : bucketIdx > 7 ? 7 : bucketIdx;
				return static_cast<uint32_t>(bucketIdx) <= bestBin;
			});

		uint32_t bestSplitCandidate = static_cast<uint32_t>(pMid - m_elements.data());

		// couldnt partition triangles; take middle instead
		if (bestSplitCandidate == begin || bestSplitCandidate == end)
		{
			bestAxis = nodeAabbExtent[0] < nodeAabbExtent[1] ? 0 : nodeAabbExtent[1] < nodeAabbExtent[2] ? 1 : 2;
			bestSplitCandidate = static_cast<uint32_t>((begin + end) / 2);
			std::nth_element(m_elements.data() + begin, m_elements.data() + bestSplitCandidate, m_elements.data() + end, [&](const auto &lhs, const auto &rhs)
				{
					AABB aabb;

					aabb = lhs.getAABB();
					const vec3 lhsCentroid = (aabb.m_min + aabb.m_max) * 0.5f;

					aabb = rhs.getAABB();
					const vec3 rhsCentroid = (aabb.m_min + aabb.m_max) * 0.5f;

					return lhsCentroid[bestAxis] < rhsCentroid[bestAxis];
				});
		}

		assert(bestSplitCandidate != begin && bestSplitCandidate != end);

		// compute children
		uint32_t leftChildIndex = buildRecursive(begin, bestSplitCandidate);
		assert(leftChildIndex == nodeIndex + 1);
		node.m_offset = buildRecursive(bestSplitCandidate, end);
		node.m_primitiveCountAxis |= (bestAxis << 8);
	}
	else
	{
		// leaf node
		node.m_offset = static_cast<uint32_t>(begin);
		node.m_primitiveCountAxis |= (end - begin) << 16;
	}
	m_nodes[nodeIndex] = node;
	return nodeIndex;
}

bool BVH::validateRecursive(uint32_t node, bool *reachedLeafs)
{
	if (m_nodes[node].m_primitiveCountAxis >> 16)
	{
		for (size_t i = 0; i < (m_nodes[node].m_primitiveCountAxis >> 16); ++i)
		{
			if (reachedLeafs[m_nodes[node].m_offset + i])
			{
				return false;
			}
			reachedLeafs[m_nodes[node].m_offset + i] = true;
		}

		return true;
	}
	else
	{
		return validateRecursive(node + 1, reachedLeafs) && validateRecursive(m_nodes[node].m_offset, reachedLeafs);
	}
}
