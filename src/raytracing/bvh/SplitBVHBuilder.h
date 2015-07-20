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
//#include "bvh/BVH.hpp"
//#include "base/Timer.hpp"
#include "Util.h"
#include "Array.h"
#include <limits>
#include <stack>
#include <queue>
#include <vector>
#include <set>
#include "eavlVector3i.h"
#include "eavlVector4.h"
using namespace std;
#define NON_LEAF_SIZE 16
#define LEAF_FLAG 0xFF800000

using FW::Vec3f;
using FW::Vec3i;
using FW::AABB;

class SplitBVHBuilder
{
private:
    enum
    {
        MaxDepth        = 64,
        MaxSpatialDepth = 48,
        NumSpatialBins  = 128
    };

    struct Reference
    {
        int                 triIdx;
        AABB                bounds;

        Reference(void) : triIdx(-1) {}
    };

    struct NodeSpec
    {
        int                 numRef;
        AABB                bounds;

        NodeSpec(void) : numRef(0) {}
    };

    struct ObjectSplit
    {
        float               sah;
        int                 sortDim;
        int                 numLeft;
        AABB                leftBounds;
        AABB                rightBounds;

        ObjectSplit(void) : sah(std::numeric_limits<float>::max()), sortDim(0), numLeft(0) {}
    };

    struct SpatialSplit
    {
        float                 sah;
        int                   dim;
        float                 pos;

        SpatialSplit(void) : sah(std::numeric_limits<float>::max()), dim(0), pos(0.0f) {}
    };

    struct SpatialBin
    {
        AABB                bounds;
        int                 enter;
        int                 exit;
    };

public:
                            SplitBVHBuilder     (float *verts,int numPrimitives, const BuildParams& params, int primitveType);
                            ~SplitBVHBuilder    (void);

    BVHNode*                run                 (void);
    int                     getSAH              (BVHNode *);
    void                    bvhToFlatArray      (BVHNode * root, int &innerSize, int &leafSize, float*& innerNodes, int*& leafNodes);

private:
    static bool             sortCompare         (void* data, int idxA, int idxB);
    static void             sortSwap            (void* data, int idxA, int idxB);

    BVHNode*                buildNode           (NodeSpec spec, int level, float progressStart, float progressEnd);
    BVHNode*                createLeaf          (const NodeSpec& spec);

    ObjectSplit             findObjectSplit     (const NodeSpec& spec, float nodeSAH);
    void                    performObjectSplit  (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

    SpatialSplit            findSpatialSplit    (const NodeSpec& spec, float nodeSAH);
    void                    performSpatialSplit (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);
    void                    splitReference      (Reference& left, Reference& right, const Reference& ref, int dim, float pos);
    void                    assignParentPointers(BVHNode*);
    void                    traverseRecursive   (BVHNode* node, BVHNode* parent);
    bool                    hasReference        (int index);
private:
                            SplitBVHBuilder     (const SplitBVHBuilder&); // forbidden
    SplitBVHBuilder&        operator=           (const SplitBVHBuilder&); // forbidden

private:
    //BVH&                    m_bvh;
    Platform                m_platform;
    const BuildParams&      m_params;
    float *                 m_verts;        /* This may not be the best way of pointing, could just make it a float ptr */
    Array<Reference>        m_refStack;
    float                   m_minOverlap;
    Array<AABB>             m_rightBounds;
    int                     m_sortDim;
    SpatialBin              m_bins[3][NumSpatialBins];

    //Timer                   m_progressTimer;
    int                     m_numDuplicates;
    int                     m_numPrimitives;
    int                     m_innerNodeCount;
    int                     m_leafNodeCount;
    int                     m_maxDepth;
    int                     m_primitiveType;
    bool                    m_doSpacialSplits;
    int numR;
    int numL; //todo: delete these
    int megaCounter;
    Array<int>              m_triIndices;  //Maybe seg fault??
};

//------------------------------------------------------------------------


//------------------------------------------------------------------------
/* 
    primitives types :
                        0 = triangle
                        1 = sphere
*/
inline SplitBVHBuilder::SplitBVHBuilder(float *verts, int numPrimitives, const BuildParams& params, int primitveType)
:   //m_bvh           (bvh),
    m_primitiveType(primitveType),
    m_params        (params),
    m_minOverlap    (0),
    m_sortDim       (-1),
    m_verts         (verts),
    m_numPrimitives  (numPrimitives)

{
    //Platform* p=new Platform();
    m_platform=*(new Platform());
    m_innerNodeCount=0;
    m_leafNodeCount=0;
    m_maxDepth=0;

    if      ( m_primitiveType == 0 ) m_doSpacialSplits=true;
    else if ( m_primitiveType == 1 ) m_doSpacialSplits=false;
    else if ( m_primitiveType == 2 ) 
    {
        m_doSpacialSplits=false;
        m_platform=*(new Platform(1)); /* Only one cell per leafNode */
    }
    else if( m_primitiveType == 3) m_doSpacialSplits=false;
    //todo remove
    megaCounter=0;
}
//------------------------------------------------------------------------

inline SplitBVHBuilder::~SplitBVHBuilder(void)
{
    //delete m_platform;
}

inline int SplitBVHBuilder::getSAH(BVHNode *root)
{
    float sah=0;
    root->computeSubtreeProbabilities(m_platform,1.f,sah);
    return sah;
}
//------------------------------------------------------------------------
inline bool SplitBVHBuilder::hasReference(int index)
{
    for(int i=0; i<m_refStack.getSize();i++)
    {
        if(m_refStack[i].triIdx==index) return true;
    }
    return false;
}
inline BVHNode* SplitBVHBuilder::run(void)
{
    // Initialize reference stack and determine root bounds.


    NodeSpec rootSpec;
    rootSpec.numRef = m_numPrimitives;
    m_refStack.resize(rootSpec.numRef);
    eavlVector3 *triPtr    = (eavlVector3 *)&m_verts[0];
    eavlVector4 *spherePtr = (eavlVector4 *)&m_verts[0];
    eavlVector4 *tetPtr    = (eavlVector4 *)&m_verts[0];
    eavlVector4 *cylPtr = (eavlVector4 *)&m_verts[0];
    for (int i = 0; i < rootSpec.numRef; i++)
    {
        m_refStack[i].triIdx = i;

        /* Insert methods here for creating bounding boxes of different primitives  */
        if(m_primitiveType == 0 )
        {
            for (int j = 0; j < 3; j++) m_refStack[i].bounds.grow(Vec3f(triPtr[i*3+j].x, triPtr[i*3+j].y,triPtr[i*3+j].z));
        }
        else if ( m_primitiveType == 1 )
        {
            Vec3f temp(0,0,0);
            Vec3f center( spherePtr[i].x, spherePtr[i].y, spherePtr[i].z );
            float radius = spherePtr[i].w;
            temp.x = radius;
            temp.y = 0;
            temp.z = 0;
            m_refStack[i].bounds.grow(center+temp);
            m_refStack[i].bounds.grow(center-temp);
            temp.x = 0;
            temp.y = radius;
            temp.z = 0;
            m_refStack[i].bounds.grow(center+temp);
            m_refStack[i].bounds.grow(center-temp);
            temp.x = 0;
            temp.y = 0;
            temp.z = radius;
            m_refStack[i].bounds.grow(center+temp);
            m_refStack[i].bounds.grow(center-temp);
        }
        else if ( m_primitiveType == 2 )
        {
            for(int j = 0; j < 4; j++)
            {
                Vec3f v( tetPtr[i*4 + j].x, tetPtr[i*4 + j].y, tetPtr[i*4 + j].z );
                m_refStack[i].bounds.grow(v);
            }
            
        }
        else if( m_primitiveType == 3 )
        {
            /* insert conservative bounding capsule*/
            Vec3f temp(0,0,0);
            Vec3f base( cylPtr[i*2].x, cylPtr[i*2].y, cylPtr[i*2].z );
            float radius = cylPtr[i*2].w;
            Vec3f axis( cylPtr[i*2+1].x, cylPtr[i*2+1].y, cylPtr[i*2+1].z );
            float height = cylPtr[i*2+1].w;
            Vec3f top = base + axis * height;

            temp.x = radius;
            temp.y = 0;
            temp.z = 0;
            m_refStack[i].bounds.grow(base+temp);
            m_refStack[i].bounds.grow(base-temp);
            m_refStack[i].bounds.grow(top +temp);
            m_refStack[i].bounds.grow(top -temp);
            temp.x = 0;
            temp.y = radius;
            temp.z = 0;
            m_refStack[i].bounds.grow(base+temp);
            m_refStack[i].bounds.grow(base-temp);
            m_refStack[i].bounds.grow(top +temp);
            m_refStack[i].bounds.grow(top -temp);
            temp.x = 0;
            temp.y = 0;
            temp.z = radius;
            m_refStack[i].bounds.grow(base+temp);
            m_refStack[i].bounds.grow(base-temp);
            m_refStack[i].bounds.grow(top +temp);
            m_refStack[i].bounds.grow(top -temp);

        }
        rootSpec.bounds.grow(m_refStack[i].bounds);
    }
    

    // Initialize rest of the members.

    m_minOverlap = rootSpec.bounds.area() * m_params.splitAlpha;
    m_rightBounds.reset(max(rootSpec.numRef, (int)NumSpatialBins) - 1);
    m_numDuplicates = 0;
    //m_progressTimer.start();

    // Build recursively.

    BVHNode* root = buildNode(rootSpec, 0, 0.0f, 1.0f);
    float s=0;
    root->computeSubtreeProbabilities(m_platform,1.f,s);
    //cout<<" ------------------BVH Stats--------------------------------"<<endl;
    //cout<<"Bounds "<<rootSpec.bounds.area()<<"   SAH : "<<s<<endl;
    //cout<<"Num Primitive Refs "<<m_numPrimitives+m_numDuplicates<<" InnerNodes "<<m_innerNodeCount<<" leaf nodes "<<m_leafNodeCount<<" Max Depth "<<m_maxDepth<<endl;

    //if (m_params.enablePrints)
    //    printf("duplicates %.0f%% Spacial Splits %d\n" , (float)m_numDuplicates / (float)m_numPrimitives * 100.0f, numSpacialSplits);
    //cout<<" ------------------End BVH Stats----------------------------"<<endl;
    //m_params.stats->SAHCost           = sah;
     
    //cout<<"Leaf Count "<<root->getSubtreeSize(BVH_STAT_LEAF_COUNT)<<endl;
    //cout<<"Inner Count "<<root->getSubtreeSize(BVH_STAT_INNER_COUNT)<<endl;;
    //cout<<"Tri Count "<<root->getSubtreeSize(BVH_STAT_TRIANGLE_COUNT)<<endl;;
    //cout<<"Child Count "<<root->getSubtreeSize(BVH_STAT_CHILDNODE_COUNT)<<endl;;


    //printf("Platform Stuff\n");
    //printf("TriCost : %f\n",m_platform.getSAHTriangleCost());
   // printf("NodeCost : %f\n",m_platform.getSAHTriangleCost());
    //printf("TriBatchSize : %d\n",m_platform.getTriangleBatchSize());
    //printf("NodeBatchSize : %d\n",m_platform.getNodeBatchSize());
    //printf("MinSize : %d\n",m_platform.getMinLeafSize());
    //printf("MaxSize : %d\n",m_platform.getMaxLeafSize());

    return root;
}

//------------------------------------------------------------------------

inline bool SplitBVHBuilder::sortCompare(void* data, int idxA, int idxB)
{
    const SplitBVHBuilder* ptr = (const SplitBVHBuilder*)data;
    int dim = ptr->m_sortDim;
    const Reference& ra = ptr->m_refStack[idxA];
    const Reference& rb = ptr->m_refStack[idxB];
    float ca = ra.bounds.min()[dim] + ra.bounds.max()[dim];
    float cb = rb.bounds.min()[dim] + rb.bounds.max()[dim];
    return (ca < cb || (ca == cb && ra.triIdx < rb.triIdx));
}

//------------------------------------------------------------------------

inline void SplitBVHBuilder::sortSwap(void* data, int idxA, int idxB)
{
    SplitBVHBuilder* ptr = (SplitBVHBuilder*)data;
    swap(ptr->m_refStack[idxA], ptr->m_refStack[idxB]);
}

//------------------------------------------------------------------------

inline BVHNode* SplitBVHBuilder::buildNode(NodeSpec spec, int level, float progressStart, float progressEnd)
{
    m_maxDepth=max(level,m_maxDepth);
    // Remove degenerates.
    {
        int firstRef = m_refStack.getSize() - spec.numRef;
        for (int i = m_refStack.getSize() - 1; i >= firstRef; i--)
        {
            Vec3f size = m_refStack[i].bounds.max() - m_refStack[i].bounds.min();
            float minExtent= min(size.x,min(size.y,size.z));
            float maxExtent= max(size.x,max(size.y,size.z));
            float sum      =size.x+size.y+size.z;
            if (minExtent < 0.0f || sum == maxExtent)
            {   
                m_refStack.removeSwap(i);
            }
        }
        spec.numRef = m_refStack.getSize() - firstRef;
    }

    // Small enough or too deep => create leaf.

    if (spec.numRef <= m_platform.getMinLeafSize() || level >= MaxDepth)
        return createLeaf(spec);

    // Find split candidates.
  
    float area = spec.bounds.area(); 
    float leafSAH = area * m_platform.getTriangleCost(spec.numRef);
    float nodeSAH = area * m_platform.getNodeCost(2);
    ObjectSplit object = findObjectSplit(spec, nodeSAH);

    SpatialSplit spatial;
    if (level < MaxSpatialDepth)
    {
        AABB overlap = object.leftBounds;
        overlap.intersect(object.rightBounds);
        if (overlap.area() >= m_minOverlap && m_doSpacialSplits) //only for triangles
            spatial = findSpatialSplit(spec, nodeSAH);
    }

    // Leaf SAH is the lowest => create leaf.
   
    float minSAH = min(leafSAH, min(object.sah, spatial.sah));   
    if (minSAH == leafSAH && spec.numRef <= m_platform.getMaxLeafSize())
        return createLeaf(spec);
    // Perform split.
    NodeSpec left, right;
    if (minSAH == spatial.sah)
    {   performSpatialSplit(left, right, spec, spatial);   }
    if (!left.numRef || !right.numRef)
    {   performObjectSplit(left, right, spec, object);    }
    // Create inner node.
    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    float progressMid = lerp(progressStart, progressEnd, (float)right.numRef / (float)(left.numRef + right.numRef));
    BVHNode* rightNode = buildNode(right, level + 1, progressStart, progressMid);
    BVHNode* leftNode = buildNode(left, level + 1, progressMid, progressEnd);
    m_innerNodeCount++;
    return new InnerNode(spec.bounds, leftNode, rightNode);
}

//------------------------------------------------------------------------

inline BVHNode* SplitBVHBuilder::createLeaf(const NodeSpec& spec)
{
    m_leafNodeCount++;

    for (int i = 0; i < spec.numRef; i++)
    {
        m_triIndices.add(m_refStack.removeLast().triIdx);
    }
    return new LeafNode(spec.bounds, m_triIndices.getSize() - spec.numRef, m_triIndices.getSize());
}

//------------------------------------------------------------------------

inline SplitBVHBuilder::ObjectSplit SplitBVHBuilder::findObjectSplit(const NodeSpec& spec, float nodeSAH)
{
    ObjectSplit split;
    const Reference* refPtr = m_refStack.getPtr(m_refStack.getSize() - spec.numRef);
    float bestTieBreak = std::numeric_limits<float>::max();

    // Sort along each dimension.

    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++)
    {
        sort(this, m_refStack.getSize() - spec.numRef, m_refStack.getSize(), sortCompare, sortSwap);

        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = spec.numRef - 1; i > 0; i--)
        {
            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        for (int i = 1; i < spec.numRef; i++)
        {
            leftBounds.grow(refPtr[i - 1].bounds);
            float sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(i) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(spec.numRef - i);
            float tieBreak = ((float)i)*((float)i) + ((float)(spec.numRef - i))*((float)(spec.numRef - i));
            if (sah < split.sah || (sah == split.sah && tieBreak < bestTieBreak))
            {
                split.sah = sah;
                split.sortDim = m_sortDim;
                split.numLeft = i;
                split.leftBounds = leftBounds;
                split.rightBounds = m_rightBounds[i - 1];
                bestTieBreak = tieBreak;
            }
        }
    }
    return split;
}

//------------------------------------------------------------------------

inline void SplitBVHBuilder::performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split)
{
    m_sortDim = split.sortDim;
    sort(this, m_refStack.getSize() - spec.numRef, m_refStack.getSize(), sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
}

//------------------------------------------------------------------------

inline SplitBVHBuilder::SpatialSplit SplitBVHBuilder::findSpatialSplit(const NodeSpec& spec, float nodeSAH)
{
    // Initialize bins.

    Vec3f origin = spec.bounds.min();
    Vec3f binSize = (spec.bounds.max() - origin) * (1.0f / (float)NumSpatialBins);
    Vec3f invBinSize = 1.0f / binSize;

    for (int dim = 0; dim < 3; dim++)
    {
        for (int i = 0; i < NumSpatialBins; i++)
        {
            SpatialBin& bin = m_bins[dim][i];
            bin.bounds = AABB();
            bin.enter = 0;
            bin.exit = 0;
        }
    }

    // Chop references into bins.

    for (int refIdx = m_refStack.getSize() - spec.numRef; refIdx < m_refStack.getSize(); refIdx++)
    {
        const Reference& ref = m_refStack[refIdx];
        Vec3i firstBin = clamp(Vec3i((ref.bounds.min() - origin) * invBinSize), 0, NumSpatialBins - 1);
        Vec3i lastBin = clamp(Vec3i((ref.bounds.max() - origin) * invBinSize), firstBin, NumSpatialBins - 1);

        for (int dim = 0; dim < 3; dim++)
        {
            Reference currRef = ref;
            for (int i = firstBin[dim]; i < lastBin[dim]; i++)
            {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, dim, origin[dim] + binSize[dim] * (float)(i + 1));
                m_bins[dim][i].bounds.grow(leftRef.bounds);
                currRef = rightRef;
            }
            m_bins[dim][lastBin[dim]].bounds.grow(currRef.bounds);
            m_bins[dim][firstBin[dim]].enter++;
            m_bins[dim][lastBin[dim]].exit++;
        }
    }

    // Select best split plane.

    SpatialSplit split;
    for (int dim = 0; dim < 3; dim++)
    {
        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = NumSpatialBins - 1; i > 0; i--)
        {
            rightBounds.grow(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.numRef;

        for (int i = 1; i < NumSpatialBins; i++)
        {
            leftBounds.grow(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;

            float sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(rightNum);
            if (sah < split.sah)
            {
                split.sah = sah;
                split.dim = dim;
                split.pos = origin[dim] + binSize[dim] * (float)i;
            }
        }
    }
    return split;
}

//------------------------------------------------------------------------

inline void SplitBVHBuilder::performSpatialSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split)
{
    // Categorize references and compute bounds.
    //
    // Left-hand side:      [leftStart, leftEnd[
    // Uncategorized/split: [leftEnd, rightStart[
    // Right-hand side:     [rightStart, refs.getSize()[
    Array<Reference>& refs = m_refStack;
    int leftStart = refs.getSize() - spec.numRef;
    int leftEnd = leftStart;
    int rightStart = refs.getSize();
    left.bounds = right.bounds = AABB();

    for (int i = leftEnd; i < rightStart; i++)
    {
        // Entirely on the left-hand side?

        if (refs[i].bounds.max()[split.dim] <= split.pos)
        {
            left.bounds.grow(refs[i].bounds);
            swap(refs[i], refs[leftEnd++]);
        }

        // Entirely on the right-hand side?

        else if (refs[i].bounds.min()[split.dim] >= split.pos)
        {
            right.bounds.grow(refs[i].bounds);
            swap(refs[i--], refs[--rightStart]);
        }
    }

    // Duplicate or unsplit references intersecting both sides.

    while (leftEnd < rightStart)
    {
        // Split reference.

        Reference lref, rref;
        splitReference(lref, rref, refs[leftEnd], split.dim, split.pos);

        // Compute SAH for duplicate/unsplit candidates.

        AABB lub = left.bounds;  // Unsplit to left:     new left-hand bounds.
        AABB rub = right.bounds; // Unsplit to right:    new right-hand bounds.
        AABB ldb = left.bounds;  // Duplicate:           new left-hand bounds.
        AABB rdb = right.bounds; // Duplicate:           new right-hand bounds.
        lub.grow(refs[leftEnd].bounds);
        rub.grow(refs[leftEnd].bounds);
        ldb.grow(lref.bounds);
        rdb.grow(rref.bounds);

        float lac = m_platform.getTriangleCost(leftEnd - leftStart);
        float rac = m_platform.getTriangleCost(refs.getSize() - rightStart);
        float lbc = m_platform.getTriangleCost(leftEnd - leftStart + 1);
        float rbc = m_platform.getTriangleCost(refs.getSize() - rightStart + 1);

        float unsplitLeftSAH = lub.area() * lbc + right.bounds.area() * rac;
        float unsplitRightSAH = left.bounds.area() * lac + rub.area() * rbc;
        float duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
        float minSAH = min(unsplitLeftSAH, min(unsplitRightSAH, duplicateSAH));

        // Unsplit to left?

        if (minSAH == unsplitLeftSAH)
        {
            left.bounds = lub;
            leftEnd++;
            //cout<<"unsplit left"<<endl;
        }

        // Unsplit to right?

        else if (minSAH == unsplitRightSAH)
        {
            //cout<<" unsplit right "<<endl;
            right.bounds = rub;
            swap(refs[leftEnd], refs[--rightStart]);
        }

        // Duplicate?

        else
        {   //cout<<"Diplicate"<<endl;
            left.bounds = ldb;
            right.bounds = rdb;
            refs[leftEnd++] = lref;
            refs.add(rref);
        }
    }

    left.numRef = leftEnd - leftStart;
    right.numRef = refs.getSize() - rightStart;
}

//------------------------------------------------------------------------

inline void SplitBVHBuilder::splitReference(Reference& left, Reference& right, const Reference& ref, int dim, float pos)
{
    // Initialize references.

    left.triIdx = right.triIdx = ref.triIdx;
    left.bounds = right.bounds = AABB();
    Vec3f *verts= (Vec3f*)&m_verts[0];
    // Loop over vertices/edges.

    //const Vec3i* tris = (const Vec3i*)m_bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    //const eavlVector3* verts = (const eavlVector3*)m_bvh.getScene()->getVtxPosBuffer().getPtr();
    Vec3i inds(ref.triIdx*3, ref.triIdx*3+1, ref.triIdx*3+2);
    const Vec3f* v1 = &verts[inds.z];

    for (int i = 0; i < 3; i++)
    {
        const Vec3f* v0 = v1;
        v1 = &verts[inds[i]];
        float v0p = v0->get(dim);
        float v1p = v1->get(dim);

        // Insert vertex to the boxes it belongs to.

        if (v0p <= pos)
            left.bounds.grow(*v0);
        if (v0p >= pos)
            right.bounds.grow(*v0);

        // Edge intersects the plane => insert intersection to both boxes.

        if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
        {
            Vec3f t = lerp(*v0, *v1, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            left.bounds.grow(t);
            right.bounds.grow(t);
        }
    }

    // Intersect with original bounds.

    left.bounds.max()[dim] = pos;
    right.bounds.min()[dim] = pos;
    left.bounds.intersect(ref.bounds);
    right.bounds.intersect(ref.bounds);
}

inline void SplitBVHBuilder::traverseRecursive(BVHNode* node, BVHNode* parent)
{
    node->m_parent=parent;
    if(node->isLeaf())
    {
        return;
    }
    for(int i=0;i<node->getNumChildNodes();i++)
        traverseRecursive(node->getChildNode(i), node);
}
inline void SplitBVHBuilder::assignParentPointers(BVHNode* root)
{
    if(root!=NULL)
    {
        traverseRecursive(root, NULL);

    }
    else
    {
        cerr<<"Cannot assign parent pointers. Null root. Bailing"<<endl;
        exit(1);
    }
}

inline void SplitBVHBuilder::bvhToFlatArray(BVHNode * root, int &innerSize, int &leafSize, float*& innerNodes, int*& leafNodes)
{
    vector<float> *flat_inner_array = new vector<float>(m_innerNodeCount*16+16);// allocate some space.
    //cout<<"Inner node array size "<<m_innerNodeCount*16+1<<endl;
    vector<int> *flat_leaf_array = new vector<int>(m_leafNodeCount*(m_platform.getMaxLeafSize()*2+1));
    //cout<<"leaf array size "<<m_leafNodeCount*(m_platform.getMaxLeafSize()+1)<<endl;
    assignParentPointers(root);
   
    root->m_index = 0;
    stack<BVHNode*> tree;
    tree.push(root);
    BVHNode *current;
    int currentIndex = 0;
    int currentLeafIndex = -1; //negative values indicate this is a leaf
    while(!tree.empty())
    {
 
        current = tree.top();
        tree.pop();
        
        if(!current->isLeaf())
        {
            current->m_index = currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(0)->m_bounds.min().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(0)->m_bounds.min().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(0)->m_bounds.min().z);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(0)->m_bounds.max().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(0)->m_bounds.max().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(0)->m_bounds.max().z);
            ++currentIndex;
            //cerr<<"bbox 2"<<endl;
            flat_inner_array->at(currentIndex) = (current->getChildNode(1)->m_bounds.min().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(1)->m_bounds.min().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(1)->m_bounds.min().z);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(1)->m_bounds.max().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(1)->m_bounds.max().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex) = (current->getChildNode(1)->m_bounds.max().z);
            ++currentIndex;
            //cerr<<"other"<<endl;
            flat_inner_array->at(currentIndex) = -1;//leftchild
            ++currentIndex;
            flat_inner_array->at(currentIndex) = -1;//rightchild Index
            ++currentIndex;
            flat_inner_array->at(currentIndex) = -2;//pad
            ++currentIndex;
            flat_inner_array->at(currentIndex) = -2;//pad
            ++currentIndex;
            //cout<<" Done"<<endl;
        }
        else
        {   
            current->m_index = currentLeafIndex;
            flat_leaf_array->at(-currentLeafIndex) = (current->getNumTriangles());
            --currentLeafIndex;
            
            LeafNode* leaf = (LeafNode*)current;
            for(int i=0; i < leaf->getNumTriangles();i++)
            {
                flat_leaf_array->at(-currentLeafIndex) = (m_triIndices[leaf->m_lo+i]);
                --currentLeafIndex;
            }
        }
        
        //tell your parent where you are 
        if(current->m_index != 0) //root has no parents 
        {

            int nodeIdx=0;
            if(current->m_index > -1) nodeIdx = current->m_index/4; // this is needed since we are loading the bvh inner nodes as float4s
            else nodeIdx = current->m_index;
            
            /* Special case where leaf is only node in the tree (i.e one primitive). Must create parent node so nothing segfaults.*/
            if( current->m_parent == NULL)
            {
                currentIndex = 0;
                flat_inner_array->at(currentIndex) = (current->m_bounds.min().x);
                ++currentIndex;
                flat_inner_array->at(currentIndex) = (current->m_bounds.min().y);
                ++currentIndex;
                flat_inner_array->at(currentIndex) = (current->m_bounds.min().z);
                ++currentIndex;
                flat_inner_array->at(currentIndex) = (current->m_bounds.max().x);
                ++currentIndex;
                flat_inner_array->at(currentIndex) = (current->m_bounds.max().y);
                ++currentIndex;
                flat_inner_array->at(currentIndex) = (current->m_bounds.max().z);
                ++currentIndex;
 
                flat_inner_array->at(currentIndex) = 0; /*TODO: maybe make this some really large value */
                ++currentIndex;
                flat_inner_array->at(currentIndex) = 0;
                ++currentIndex;
                flat_inner_array->at(currentIndex) = 0;
                ++currentIndex;
                flat_inner_array->at(currentIndex) = 0;
                ++currentIndex;
                flat_inner_array->at(currentIndex) = 0;
                ++currentIndex;
                flat_inner_array->at(currentIndex) = 0;
                ++currentIndex;
                flat_inner_array->at(currentIndex)=-1;//leftchild
                ++currentIndex;
                flat_inner_array->at(currentIndex)=-1;//rightchild Index
                ++currentIndex;
                flat_inner_array->at(currentIndex)=-2;//pad
                ++currentIndex;
                flat_inner_array->at(currentIndex)=-2;//pad
                ++currentIndex;

            }
            else
            {
                if(current->m_parent->getChildNode(0) == current)
                {
                    float clf;
                    memcpy(&clf,&nodeIdx, 4);
                    flat_inner_array->at(current->m_parent->m_index+12) = clf;
                }
                else if(current->m_parent->getChildNode(1) == current)
                {
                    float clf;
                    memcpy(&clf,&nodeIdx, 4);
                    flat_inner_array->at(current->m_parent->m_index+13) = clf;
                }

            }
           
        }

        if (current->getChildNode(0)!=NULL) tree.push(current->getChildNode(0));
        if (current->getChildNode(1)!=NULL) tree.push(current->getChildNode(1));

    }

    float *innerraw = new float[currentIndex];
    int   *leafraw = new int[-currentLeafIndex];
    int numLeafVals=-currentLeafIndex;

    for (int i = 0; i < currentIndex;i++)
    {
        innerraw[i] = flat_inner_array->at(i);
    }
    cout<<endl;
    for (int i = 1; i < numLeafVals;i++)
    {
        leafraw[i - 1] = flat_leaf_array->at(i);
    }
    cout<<endl;
    innerNodes = innerraw;
    leafNodes = leafraw;
    delete flat_inner_array;
    delete flat_leaf_array;
    innerSize = currentIndex;
    leafSize = numLeafVals;

}

//---------

