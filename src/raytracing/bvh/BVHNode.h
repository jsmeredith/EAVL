/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include "Platform.h"
#include "Util.h"
using FW::AABB;

// TODO: remove m_probability. Node needed after all?

enum BVH_STAT
{
    BVH_STAT_NODE_COUNT,
    BVH_STAT_INNER_COUNT,
    BVH_STAT_LEAF_COUNT,
    BVH_STAT_TRIANGLE_COUNT,
    BVH_STAT_CHILDNODE_COUNT,
};

class BVHNode
{
public:
    BVHNode() : m_probability(1.f),m_parentProbability(1.f),m_treelet(-1),m_index(-1) {}
    virtual ~BVHNode() {};
    virtual bool        isLeaf() const = 0;
    virtual int         getNumChildNodes() const = 0;
    virtual BVHNode*    getChildNode(int i) const   = 0;
    virtual int         getNumTriangles() const { return 0; }
    BVHNode*            m_parent;
    float               getArea() const     { return m_bounds.area(); }

    AABB        m_bounds;

    // These are somewhat experimental, for some specific test and may be invalid...
    float       m_probability;          // probability of coming here (widebvh uses this)
    float       m_parentProbability;    // probability of coming to parent (widebvh uses this)

    int         m_treelet;              // for queuing tests (qmachine uses this)
    int         m_index;                // in linearized tree (qmachine uses this)

    // Subtree functions
    int     getSubtreeSize(BVH_STAT stat=BVH_STAT_NODE_COUNT) const;
    void    computeSubtreeProbabilities(const Platform& p, float parentProbability, float& sah);
    float   computeSubtreeSAHCost(const Platform& p) const;     // NOTE: assumes valid probabilities
    void    deleteSubtree();

    void    assignIndicesDepthFirst  (int index=0, bool includeLeafNodes=true);
    void    assignIndicesBreadthFirst(int index=0, bool includeLeafNodes=true);
};


class InnerNode : public BVHNode
{
public:
    InnerNode(const AABB& bounds,BVHNode* child0,BVHNode* child1)   { m_bounds=bounds; m_children[0]=child0; m_children[1]=child1; }
    InnerNode(const AABB& bounds,BVHNode* child0,BVHNode* child1,BVHNode * parent)   { m_bounds=bounds; m_children[0]=child0; m_children[1]=child1; m_parent=parent;}

    bool        isLeaf() const                  { return false; }
    int         getNumChildNodes() const        { return 2; }
    BVHNode*    getChildNode(int i) const       { if(i<0 && i>1){ cerr<<"getChildNode bad value : "<<i<<endl; exit(1);}; return m_children[i]; }

    BVHNode*    m_children[2];
};


class LeafNode : public BVHNode
{
public:
    LeafNode(const AABB& bounds,int lo,int hi)  { m_bounds=bounds; m_lo=lo; m_hi=hi; }
    LeafNode(const AABB& bounds,int lo,int hi, BVHNode* parent)  { m_bounds=bounds; m_lo=lo; m_hi=hi; m_parent=parent;}
    LeafNode(const LeafNode& s)                 { *this = s; }

    bool        isLeaf() const                  { return true; }
    int         getNumChildNodes() const        { return 0; }
    BVHNode*    getChildNode(int) const         { return NULL; }

    int         getNumTriangles() const         { return m_hi-m_lo; }
    int         m_lo;
    int         m_hi;
};



inline int BVHNode::getSubtreeSize(BVH_STAT stat) const
{
    int cnt;
    switch(stat)
    {
        default: {cerr<<"getSubtreeSize:  Unknown Mode."<<endl; exit(1);}  // unknown mode
        case BVH_STAT_NODE_COUNT:      cnt = 1; break;
        case BVH_STAT_LEAF_COUNT:      cnt = isLeaf() ? 1 : 0; break;
        case BVH_STAT_INNER_COUNT:     cnt = isLeaf() ? 0 : 1; break;
        case BVH_STAT_TRIANGLE_COUNT:  cnt = isLeaf() ? reinterpret_cast<const LeafNode*>(this)->getNumTriangles() : 0; break;
        case BVH_STAT_CHILDNODE_COUNT: cnt = getNumChildNodes(); break;
    }

    if(!isLeaf())
    {
        for(int i=0;i<getNumChildNodes();i++)
            cnt += getChildNode(i)->getSubtreeSize(stat);
    }

    return cnt;
}


inline void BVHNode::deleteSubtree()
{
    for(int i=0;i<getNumChildNodes();i++)
        getChildNode(i)->deleteSubtree();

    delete this;
}


inline void BVHNode::computeSubtreeProbabilities(const Platform& p,float probability, float& sah)
{
    //cerr<<this->m_index<<endl;
    //cerr<<"Num Child nodes : "<<this->getNumChildNodes()<<endl;
    //cerr<<"Tris : "<<this->getNumTriangles()<<endl;
    sah += probability * p.getCost(this->getNumChildNodes(),this->getNumTriangles());

    m_probability = probability;

    for(int i=0;i<getNumChildNodes();i++)
    {
        BVHNode* child = getChildNode(i);
        child->m_parentProbability = probability;
        float childProbability = 0.0f;
        if (probability > 0.0f)
            childProbability = probability * child->m_bounds.area()/this->m_bounds.area();
        child->computeSubtreeProbabilities(p, childProbability, sah );
    }
}


// TODO: requires valid probabilities...
inline float BVHNode::computeSubtreeSAHCost(const Platform& p) const
{
    float SAH = m_probability * p.getCost( getNumChildNodes(),getNumTriangles());

    for(int i=0;i<getNumChildNodes();i++)
        SAH += getChildNode(i)->computeSubtreeSAHCost(p);

    return SAH;
}

//-------------------------------------------------------------

inline void assignIndicesDepthFirstRecursive( BVHNode* node, int& index, bool includeLeafNodes )
{
    if(node->isLeaf() && !includeLeafNodes)
        return;

    node->m_index = index++;
    for(int i=0;i<node->getNumChildNodes();i++)
        assignIndicesDepthFirstRecursive(node->getChildNode(i), index, includeLeafNodes);
}

inline void BVHNode::assignIndicesDepthFirst( int index, bool includeLeafNodes )
{
    assignIndicesDepthFirstRecursive( this, index, includeLeafNodes );
}

//-------------------------------------------------------------
/*
void BVHNode::assignIndicesBreadthFirst( int index, bool includeLeafNodes )
{
    Array<BVHNode*> nodes;
    nodes.add(this);
    int head=0;

    while(head < nodes.getSize())
    {
        // pop
        BVHNode* node = nodes[head++];

        // discard
        if(node->isLeaf() && !includeLeafNodes)
            continue;

        // assign
        node->m_index = index++;

        // push children
        for(int i=0;i<node->getNumChildNodes();i++)
            nodes.add(node->getChildNode(i));
    }
}
*/
