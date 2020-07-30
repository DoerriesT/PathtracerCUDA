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
	// validate that every leaf node is reachable through exactly one path
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

static float calcSurfaceArea(const vec3 &minCorner, const vec3 &maxCorner)
{
	const vec3 extent = maxCorner - minCorner;
	if (extent.x <= 0.0f || extent.y <= 0.0f || extent.z <= 0.0f)
	{
		return 0.0f;
	}

	// a * b + a * c + b * c * 2
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

	// if we still have more primitives than m_maxLeafElements, we need to partition them into two child nodes
	if ((end - begin) > m_maxLeafElements)
	{
		// we use the surface area heuristic (SAH) to get reasonably good node AABBs.
		// since exhaustively searching through all possible partitions for the optimal partition (smallest combined surface area of the AABBs of both child nodes)
		// would be too expensive, we try to approximate the optimal solution by using binning instead:
		// we split each axis of the AABB of the current node into 8 bins and then for each axis assign each primitive to the closest bin, inflating the bin's
		// aabb and incrementing its primitive count.
		// then for each axis we walk through the bins and search for the position at which splitting the bins into all bins before and after results in the lowest cost.
		// finally, we split all primitives along the best axis and at the best split

		// holds AABB and number of primitives for each bin
		struct BinInfo
		{
			uint32_t m_count = 0;
			AABB m_aabb = AABB(vec3(std::numeric_limits<float>::max()), vec3(std::numeric_limits<float>::lowest()));
		};
		BinInfo bins[3][8];

		const auto nodeAabbExtent = max(node.m_aabb.m_max - node.m_aabb.m_min, vec3(0.00000001f));

		// assign each primitive to a bin
		for (size_t i = begin; i < end; ++i)
		{
			AABB elemAabb = m_elements[i].getAABB();

			// center of primitive
			const vec3 centroid = (elemAabb.m_min + elemAabb.m_max) * 0.5f;
			// relative position [0..1] in parent AABB
			const vec3 relativeOffset = (centroid - node.m_aabb.m_min) / nodeAabbExtent;

			// for each axis, find the matching bin
			for (int j = 0; j < 3; ++j)
			{
				int binIdx = int(relativeOffset[j] * 8.0f);
				binIdx = binIdx < 0 ? 0 : binIdx > 7 ? 7 : binIdx;

				// increase primitive count and inflate bin AABB with primitive AABB
				auto &bin = bins[j][binIdx];
				bin.m_count += 1;
				bin.m_aabb = AABB(bin.m_aabb, elemAabb);
			}
		}

		const float invNodeSurfaceArea = 1.0f / std::max(calcSurfaceArea(node.m_aabb.m_min, node.m_aabb.m_max), 0.000000001f);
		float lowestCost = std::numeric_limits<float>::max();
		uint32_t bestAxis = 0;
		uint32_t bestBin = 0;

		// find split with lowest cost
		float costs[3][8];
		// for each axis
		for (uint32_t i = 0; i < 3; ++i)
		{
			// for each bin of that axis
			for (uint32_t j = 0; j < 7; ++j)
			{
				// count number of primitives and compute size of the AABB if we were to split after the current bin
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

				// compute surface area of potential child node AABBs
				float a0 = calcSurfaceArea(aabb0.m_min, aabb0.m_max);
				float a1 = calcSurfaceArea(aabb1.m_min, aabb1.m_max);

				// heuristic: ratio of the sum of both child node surface areas to the parent node surface area.
				// in order to avoid very uneven splits, we try to balance the cost a little, by taking the number of primitives
				// per child node into account.
				float cost = 0.125f + (count0 * a0 + count1 * a1) * invNodeSurfaceArea;
				cost = (count0 == 0 || count1 == 0) ? std::numeric_limits<float>::max() : cost; // avoid 0/N partitions
				costs[i][j] = cost;
				assert(cost == cost);

				// update best axis and bin
				if (cost < lowestCost)
				{
					lowestCost = cost;
					bestAxis = i;
					bestBin = j;
				}
			}
		}

		// partition all elements by the chose axis and split
		const CpuHittable *pMid = std::partition(m_elements.data() + begin, m_elements.data() + end, [&](const auto &item)
			{
				const AABB elemAabb = item.getAABB();
				const vec3 centroid = (elemAabb.m_min + elemAabb.m_max) * 0.5f;
				const float relativeOffset = ((centroid[bestAxis] - node.m_aabb.m_min[bestAxis]) / nodeAabbExtent[bestAxis]);
				
				// compute the index of the bin this primitive belongs to
				int binIdx = int(8.0f * relativeOffset);
				binIdx = binIdx < 0 ? 0 : binIdx > 7 ? 7 : binIdx;
				return static_cast<uint32_t>(binIdx) <= bestBin;
			});

		uint32_t bestSplitCandidate = static_cast<uint32_t>(pMid - m_elements.data());

		// couldnt partition primitives; take middle instead
		if (bestSplitCandidate == begin || bestSplitCandidate == end)
		{
			bestAxis = (nodeAabbExtent[0] < nodeAabbExtent[1]) ? 0 : (nodeAabbExtent[1] < nodeAabbExtent[2]) ? 1 : 2;
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
		// we build the tree depth first, so the left child immediately follows in memory
		assert(leftChildIndex == nodeIndex + 1);
		// the right child can be anywhere after the the left child in memory, so we need to store its index
		node.m_offset = buildRecursive(bestSplitCandidate, end);
		// store the axis that we split the node along. this is useful for traversal
		node.m_primitiveCountAxis |= (bestAxis << 8);
	}
	else
	{
		// leaf node
		node.m_offset = static_cast<uint32_t>(begin); // offset of first primitive
		node.m_primitiveCountAxis |= (end - begin) << 16; // number of primitives
	}
	m_nodes[nodeIndex] = node;
	return nodeIndex;
}

bool BVH::validateRecursive(uint32_t node, bool *reachedLeafs)
{
	// the upper 16 bit of m_primitiveCountAxis store the number of primitives of this node.
	// if it is 0, it is an interior node and we need to keep going down the tree until we find a leaf node
	if (m_nodes[node].m_primitiveCountAxis >> 16)
	{
		// for every primitive of this leaf node
		for (size_t i = 0; i < (m_nodes[node].m_primitiveCountAxis >> 16); ++i)
		{
			// if we already reached the primitive, validation failed
			if (reachedLeafs[m_nodes[node].m_offset + i])
			{
				return false;
			}
			// mark the primitive as reached
			reachedLeafs[m_nodes[node].m_offset + i] = true;
		}

		return true;
	}
	else
	{
		// recursively validate both children
		return validateRecursive(node + 1, reachedLeafs) && validateRecursive(m_nodes[node].m_offset, reachedLeafs);
	}
}
