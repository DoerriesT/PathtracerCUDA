#pragma once
#include "vec3.h"
#include "AABB.h"
#include <vector>
#include "Hittable.h"

struct BVHNode
{
	AABB m_aabb;
	uint32_t m_offset; // primitive offset for leaf; second child offset for interior node
	uint32_t m_primitiveCountAxis;  // 0-7 padding; 8-15 axis; 16-31 prim count (0 -> interior node)
};

class BVH
{
public:
	void build(size_t elementCount, const Hittable *elements, uint32_t maxLeafElements);
	const std::vector<BVHNode> &getNodes() const;
	const std::vector<Hittable> &getElements() const;
	uint32_t getDepth(uint32_t node = 0) const;
	bool validate();
	bool trace(const vec3 &origin, const vec3 &dir, float &t);

private:

	uint32_t m_maxLeafElements = 1;
	std::vector<BVHNode> m_nodes;
	std::vector<Hittable> m_elements;

	uint32_t buildRecursive(size_t begin, size_t end);
	bool validateRecursive(uint32_t node, bool *reachedLeafs);
};