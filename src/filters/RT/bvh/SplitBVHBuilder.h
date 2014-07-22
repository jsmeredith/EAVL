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
using namespace std;
#define NON_LEAF_SIZE 16
#define LEAF_FLAG 0xFF800000
int numSpacialSplits=0;
int tcounter=0;
class SplitBVHBuilder
{
private:
    enum
    {
        MaxDepth        = 64,
        MaxSpatialDepth = 48,
        NumSpatialBins  = 128,
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
    void                    bvhToFlatArray      (BVHNode * root, int &innerSize, int &leafSize, float*& innerNodes, float*& leafNodes);

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
SplitBVHBuilder::SplitBVHBuilder(float *verts, int numPrimitives, const BuildParams& params, int primitveType)
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
    //todo remove
    megaCounter=0;
}
//------------------------------------------------------------------------

SplitBVHBuilder::~SplitBVHBuilder(void)
{
}

int SplitBVHBuilder::getSAH(BVHNode *root)
{
    float sah=0;
    root->computeSubtreeProbabilities(m_platform,1.f,sah);
    return sah;
}
//------------------------------------------------------------------------
bool SplitBVHBuilder::hasReference(int index)
{
    for(int i=0; i<m_refStack.getSize();i++)
    {
        if(m_refStack[i].triIdx==index) return true;
    }
    return false;
}
BVHNode* SplitBVHBuilder::run(void)
{
    // Initialize reference stack and determine root bounds.

    //const Vec3i* tris ;//= (const Vec3i*)m_bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    //const eavlVector3* verts ;//= (const eavlVector3*)m_bvh.getScene()->getVtxPosBuffer().getPtr(); //insert here

    NodeSpec rootSpec;
    rootSpec.numRef = m_numPrimitives;
    m_refStack.resize(rootSpec.numRef);
    eavlVector3 *triPtr    = (eavlVector3 *)&m_verts[0];
    eavlVector4 *spherePtr = (eavlVector4 *)&m_verts[0];
    for (int i = 0; i < rootSpec.numRef; i++)
    {
        m_refStack[i].triIdx = i;

        /* Insert methods here for creating bounding boxes of different primitives  */
        if(m_primitiveType == 0 )
        {
            for (int j = 0; j < 3; j++) m_refStack[i].bounds.grow(triPtr[i*4+j]);
        }
        else if ( m_primitiveType == 1 )
        {
            eavlVector3 temp(0,0,0);
            eavlVector3 center( spherePtr[i].x, spherePtr[i].y, spherePtr[i].z );
            float radius = spherePtr[i].w;
            temp.x=radius;
            temp.y=0;
            temp.z=0;
            m_refStack[i].bounds.grow(center+temp);
            m_refStack[i].bounds.grow(center-temp);
            temp.x=0;
            temp.y=radius;
            temp.z=0;
            m_refStack[i].bounds.grow(center+temp);
            m_refStack[i].bounds.grow(center-temp);
            temp.x=0;
            temp.y=0;
            temp.z=radius;
            m_refStack[i].bounds.grow(center+temp);
            m_refStack[i].bounds.grow(center-temp);
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
    cout<<" ------------------BVH Stats--------------------------------"<<endl;
    cout<<"Bounds "<<rootSpec.bounds.area()<<"   SAH : "<<s<<endl;
    cout<<"Num Triangles Refs "<<m_numPrimitives+m_numDuplicates<<" InnerNodes "<<m_innerNodeCount<<" leaf nodes "<<m_leafNodeCount<<" Max Depth "<<m_maxDepth<<endl;

    if (m_params.enablePrints)
        printf("duplicates %.0f%% Spacial Splits %d\n" , (float)m_numDuplicates / (float)m_numPrimitives * 100.0f, numSpacialSplits);
    cout<<" ------------------End BVH Stats----------------------------"<<endl;
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

bool SplitBVHBuilder::sortCompare(void* data, int idxA, int idxB)
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

void SplitBVHBuilder::sortSwap(void* data, int idxA, int idxB)
{
    SplitBVHBuilder* ptr = (SplitBVHBuilder*)data;
    swap(ptr->m_refStack[idxA], ptr->m_refStack[idxB]);
}

//------------------------------------------------------------------------

BVHNode* SplitBVHBuilder::buildNode(NodeSpec spec, int level, float progressStart, float progressEnd)
{
    // Display progress.
 
        //removed
    megaCounter++;

    //todo remove me 
    numR=0;
    numL=0;

    m_maxDepth=max(level,m_maxDepth);
    // Remove degenerates.
    {
        //bool hereThen=hasReference(2);
        int firstRef = m_refStack.getSize() - spec.numRef;
        for (int i = m_refStack.getSize() - 1; i >= firstRef; i--)
        {
            eavlVector3 size = m_refStack[i].bounds.max() - m_refStack[i].bounds.min();
            float minExtent= min(size.x,min(size.y,size.z));
            float maxExtent= max(size.x,max(size.y,size.z));
            float sum      =size.x+size.y+size.z;
            if (minExtent < 0.0f || sum == maxExtent)
            {   //std::cerr<<"Degenerate : "<<minExtent<<" "<<maxExtent<<" "<<sum<<std::endl;
                m_refStack.removeSwap(i);
            }
        }
        spec.numRef = m_refStack.getSize() - firstRef;
        //bool hereNow=hasReference(2);
        //std::cerr<<hereThen<<" spec "<< hereNow<<std::endl;
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
    //cout<<"SAH for splits : split : "<<spatial.sah<<" "<<object.sah<<endl;
    //cout<<"R "<<numR<<" L "<<numL<<" "<<megaCounter<<endl;
    // Perform split.
    //printf("Counter %d\n",tcounter); 
    NodeSpec left, right;
    if (minSAH == spatial.sah)
    {   performSpatialSplit(left, right, spec, spatial);   }
    if (!left.numRef || !right.numRef)
    {   performObjectSplit(left, right, spec, object);    }
    tcounter++;
    // Create inner node.
    //if(tcounter==10) exit(0);

    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    float progressMid = lerp(progressStart, progressEnd, (float)right.numRef / (float)(left.numRef + right.numRef));
    BVHNode* rightNode = buildNode(right, level + 1, progressStart, progressMid);
    BVHNode* leftNode = buildNode(left, level + 1, progressMid, progressEnd);
    m_innerNodeCount++;
    return new InnerNode(spec.bounds, leftNode, rightNode);
}

//------------------------------------------------------------------------

BVHNode* SplitBVHBuilder::createLeaf(const NodeSpec& spec)
{
    m_leafNodeCount++;
    //Array<int>& tris = m_bvh.getTriIndices();
    
    for (int i = 0; i < spec.numRef; i++)
    {
        //if(m_refStack.getLast().triIdx==63) {cout<<"This is the leaf "<<megaCounter<<endl; exit(0);}
        m_triIndices.add(m_refStack.removeLast().triIdx);
    }
    
    /*if(hereThen && !hereNow)
    {
        std:cerr<<"lost 2"<<" spec "<< spec.numRef<<std::endl;
        for (int i = m_triIndices.getSize()-1; i > m_triIndices.getSize()-spec.numRef-1; i++)
        std::cerr<<m_triIndices[i]<<" ";
        exit(0);
    }*/
    return new LeafNode(spec.bounds, m_triIndices.getSize() - spec.numRef, m_triIndices.getSize());
}

//------------------------------------------------------------------------

SplitBVHBuilder::ObjectSplit SplitBVHBuilder::findObjectSplit(const NodeSpec& spec, float nodeSAH)
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

void SplitBVHBuilder::performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split)
{
    m_sortDim = split.sortDim;
    sort(this, m_refStack.getSize() - spec.numRef, m_refStack.getSize(), sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
}

//------------------------------------------------------------------------

SplitBVHBuilder::SpatialSplit SplitBVHBuilder::findSpatialSplit(const NodeSpec& spec, float nodeSAH)
{
    // Initialize bins.

    eavlVector3 origin = spec.bounds.min();
    eavlVector3 binSize = (spec.bounds.max() - origin) * (1.0f / (float)NumSpatialBins);
    eavlVector3 invBinSize(1.0f / binSize.x,1.0f / binSize.y,1.0f / binSize.z) ;
    //if(megaCounter==204978)
    //{
    //    cout<<"Nodespec num"<<spec.numRef<<endl;
    //    exit(0);
    //}
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
        eavlVector3i firstBin;
        firstBin.x= clamp((int)((ref.bounds.min().x - origin.x) * invBinSize.x), 0, NumSpatialBins - 1);
        firstBin.y= clamp((int)((ref.bounds.min().y - origin.y) * invBinSize.y), 0, NumSpatialBins - 1);
        firstBin.z= clamp((int)((ref.bounds.min().z - origin.z) * invBinSize.z), 0, NumSpatialBins - 1);
        eavlVector3i lastBin;
        lastBin.x = clamp((int)((ref.bounds.max().x - origin.x) * invBinSize.x), firstBin.x, NumSpatialBins - 1);
        lastBin.y = clamp((int)((ref.bounds.max().y - origin.y) * invBinSize.y), firstBin.y, NumSpatialBins - 1);
        lastBin.z = clamp((int)((ref.bounds.max().z - origin.z) * invBinSize.z), firstBin.z, NumSpatialBins - 1);

        for (int dim = 0; dim < 3; dim++)
        {
            Reference currRef = ref;

            /*int firstBinDim;
            if     (dim==0) firstBinDim=firstBin.x;
            else if(dim==1) firstBinDim=firstBin.y;
            else            firstBinDim=firstBin.z;
            int lastBinDim;
            if     (dim==0) lastBinDim=lastBin.x;
            else if(dim==1) lastBinDim=lastBin.y;
            else            lastBinDim=lastBin.z;
            */
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
        //cout<<"XXXX";
        for (int i = 1; i < NumSpatialBins; i++)
        {
            //cout<<leftBounds.min()<<endl;
            leftBounds.grow(m_bins[dim][i - 1].bounds);
            //cout<<leftBounds.min()<<endl;
            if(leftBounds.min().x<-100000)
            {
                //cout<<"poop"<<endl;
            }
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;
            //cout<<i<<" Totols refs "<< spec.numRef<<" Right " <<rightNum<<" left "<<leftNum<<endl;
            //cout<<"Bounds left : "<<leftBounds.area()<< " right bounds "<< " i "<<i<<" "<<m_rightBounds[i - 1].area()<<endl;
            
            float sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(rightNum);
            if (sah < split.sah)
            {
                split.sah = sah;
                split.dim = dim;
                split.pos = origin[dim] + binSize[dim] * (float)i;
                numR=rightNum;
                numL=leftNum;
            }
        }
    }
    return split;
}

//------------------------------------------------------------------------

void SplitBVHBuilder::performSpatialSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split)
{
    // Categorize references and compute bounds.
    //
    // Left-hand side:      [leftStart, leftEnd[
    // Uncategorized/split: [leftEnd, rightStart[
    // Right-hand side:     [rightStart, refs.getSize()[
    numSpacialSplits++;
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

void SplitBVHBuilder::splitReference(Reference& left, Reference& right, const Reference& ref, int dim, float pos)
{
    // Initialize references.

    left.triIdx = right.triIdx = ref.triIdx;
    left.bounds = right.bounds = AABB();
    eavlVector3 *triPtr= (eavlVector3*)&m_verts[0];
    // Loop over vertices/edges.

    //const Vec3i* tris = (const Vec3i*)m_bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    //const eavlVector3* verts = (const eavlVector3*)m_bvh.getScene()->getVtxPosBuffer().getPtr();
    //const Vec3i& inds = tris[ref.triIdx];
    eavlVector3* v1;// = &verts[inds.z];
    eavlVector3* v0;
    for (int i = 0; i < 3; i++)
    {
        //const eavlVector3* v0 = v1;
        //v1 = &verts[inds[i]];
        if(i==0)
        {
            
            v0=&triPtr[ref.triIdx*4];
            v1=&triPtr[ref.triIdx*4+1];
        }
        else if(i==1)
        {
            
            v0=&triPtr[ref.triIdx*4+1];
            v1=&triPtr[ref.triIdx*4+2];
        }
        else
        {   
            v0=&triPtr[ref.triIdx*4+2];
            v1=&triPtr[ref.triIdx*4];
        }

        float v0p, v1p;
        if(dim==0)
        {
            v0p=v0->x;
            v1p=v1->x;
        }
        else if (dim==1)
        {
            v0p=v0->y;
            v1p=v1->y;
        }
        else 
        {
            v0p=v0->z;
            v1p=v1->z;
        }
        // Insert vertex to the boxes it belongs to.

        if (v0p <= pos)
            left.bounds.grow(*v0);
        if (v0p >= pos)
            right.bounds.grow(*v0);

        // Edge intersects the plane => insert intersection to both boxes.

        if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
        {
            eavlVector3 t;// = lerp(*v0, *v1, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            t.x=lerp(v0->x, v1->x, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            t.y=lerp(v0->y, v1->y, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            t.z=lerp(v0->z, v1->z, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
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

void SplitBVHBuilder::traverseRecursive(BVHNode* node, BVHNode* parent)
{
    node->m_parent=parent;
    if(node->isLeaf())
    {
        return;
    }
    for(int i=0;i<node->getNumChildNodes();i++)
        traverseRecursive(node->getChildNode(i), node);
}
void SplitBVHBuilder::assignParentPointers(BVHNode* root)
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

void SplitBVHBuilder::bvhToFlatArray(BVHNode * root, int &innerSize, int &leafSize, float*& innerNodes, float*& leafNodes)
{
    vector<float> *flat_inner_array= new vector<float>(m_innerNodeCount*16+1);// allocate some space.
    vector<float> *flat_leaf_array = new vector<float>(m_leafNodeCount*(m_platform.getMaxLeafSize()+1));
    assignParentPointers(root);
   
    root->m_index=0;
    stack<BVHNode*> tree;
    tree.push(root);
    BVHNode *current;
    int currentIndex=0;
    int currentLeafIndex=-1; //negative values indicate this is a leaf
    while(!tree.empty())
    {
        //cout<<"Beg"<<endl;
        current=tree.top();
        tree.pop();
        //cout<<"In"<<endl;
        
        if(!current->isLeaf())
        {
            //cerr<<"making inner"<<currentIndex;
            current->m_index=currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(0)->m_bounds.min().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(0)->m_bounds.min().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(0)->m_bounds.min().z);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(0)->m_bounds.max().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(0)->m_bounds.max().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(0)->m_bounds.max().z);
            ++currentIndex;
            //cerr<<"bbox 2"<<endl;
            flat_inner_array->at(currentIndex)=(current->getChildNode(1)->m_bounds.min().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(1)->m_bounds.min().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(1)->m_bounds.min().z);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(1)->m_bounds.max().x);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(1)->m_bounds.max().y);
            ++currentIndex;
            flat_inner_array->at(currentIndex)=(current->getChildNode(1)->m_bounds.max().z);
            ++currentIndex;
            //cerr<<"other"<<endl;
            flat_inner_array->at(currentIndex)=-1;//leftchild
            ++currentIndex;
            flat_inner_array->at(currentIndex)=-1;//rightchild Index
            ++currentIndex;
            flat_inner_array->at(currentIndex)=-2;//pad
            ++currentIndex;
            flat_inner_array->at(currentIndex)=-2;//pad
            ++currentIndex;
            //cout<<" Done"<<endl;
        }
        else
        {   //cout<<"Making leaf "<<currentLeafIndex<<" Size "<<flat_leaf_array->size()<<endl;
            current->m_index=currentLeafIndex;
            flat_leaf_array->at(-currentLeafIndex)=(current->getNumTriangles());
            --currentLeafIndex;
            
            LeafNode* leaf=(LeafNode*)current;
            for(int i=0;i<leaf->getNumTriangles();i++)
            {
                //cout<<currentLeafIndex<<endl;
                flat_leaf_array->at(-currentLeafIndex)=(m_triIndices[leaf->m_lo+i]);
                --currentLeafIndex;
            }
            //cout<<" Done"<<endl;
        }
        
        //tell your parent where you are
        //cout<<"Where is your parent"<<endl;
        if(current->m_index!=0) //root has no parents 
        {

            int nodeIdx=0;
            if(current->m_index>-1) nodeIdx=current->m_index/4; // this is needed since we are loading the bvh inner nodes as float4s
            else nodeIdx=current->m_index;
            if(current->m_parent->getChildNode(0)==current)
            {
                //cerr<<"I am the left child updating my location"<<current->m_parent->m_index+12<<" "<<current->m_index<<" "<<flat_inner_array->size()<<endl;
                flat_inner_array->at(current->m_parent->m_index+12)=nodeIdx;


            }
            else if(current->m_parent->getChildNode(1)==current)
            {
                //cerr<<"I am the right child updating my location"<<current->m_parent->m_index+13<<" "<<current->m_index<<" "<<flat_inner_array->size()<<endl;
                flat_inner_array->at(current->m_parent->m_index+13)=nodeIdx;
            }
            else cerr<<"Node "<<current->m_index<<" is an oprhan"<<endl;
        }
        //++currentIndex;
        //cout<<"H"<<endl;
        if (current->getChildNode(0)!=NULL) tree.push(current->getChildNode(0));
        if (current->getChildNode(1)!=NULL) tree.push(current->getChildNode(1));
        //cout<<"Hi"<<endl;
    }
    //cout<<"After a while"<<endl;
    float *innerraw = new float[currentIndex];
    float *leafraw = new float[-currentLeafIndex];
    int numLeafVals=-currentLeafIndex;
    //cout<<"BVH address : "<<raw<<endl;
    //exit(0);
    cout<<"Dumping raw vals"<<endl;
    for (int i=0; i<currentIndex;i++)
    {
        innerraw[i]=flat_inner_array->at(i);
        //cout<<innerraw[i]<<" ";
    }
    cout<<"Doing leaves"<<endl;
    for (int i=0; i<numLeafVals;i++)
    {
        leafraw[i]=flat_leaf_array->at(i);
        //cout<<leafraw[i]<<" ";
    }
    innerNodes=innerraw;
    leafNodes=leafraw;
    delete flat_inner_array;
    delete flat_leaf_array;
    innerSize=currentIndex;
    leafSize=numLeafVals;
    cerr<<"Done.. Inner Size "<<innerSize<<" leaf size "<<numLeafVals<<endl;
    //for (int i=0; i< m_numPrimitives;i++) cerr<<"Root acess test "<<m_triIndices[i]<<endl;
}

//---------

