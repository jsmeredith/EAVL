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
#include <string.h>


class LeafNode;
class BVHNode;
class Platform
{
public:
    Platform() { m_name=string("Default"); m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize=1; m_maxLeafSize=8; }//m_maxLeafSize=0x7FFFFFF;
    Platform(const string& name,float nodeCost=1.f, float triCost=1.f, int nodeBatchSize=1, int triBatchSize=1) { m_name=name; m_SAHNodeCost = nodeCost; m_SAHTriangleCost = triCost; m_nodeBatchSize = nodeBatchSize; m_triBatchSize = triBatchSize; m_minLeafSize=1; m_maxLeafSize=8; }
    Platform(int nodeMax) { m_name=string("Default"); m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize = 1; m_maxLeafSize = nodeMax; }
    const string&   getName() const                     { return m_name; }

    // SAH weights
    float getSAHTriangleCost() const                    { return m_SAHTriangleCost; }
    float getSAHNodeCost() const                        { return m_SAHNodeCost; }

    // SAH costs, raw and batched
    float getCost(int numChildNodes,int numTris) const  { return getNodeCost(numChildNodes) + getTriangleCost(numTris); }
    float getTriangleCost(int n) const                  { return roundToTriangleBatchSize(n) * m_SAHTriangleCost; }
    float getNodeCost(int n) const                      { return roundToNodeBatchSize(n) * m_SAHNodeCost; }

    // batch processing (how many ops at the price of one)
    int   getTriangleBatchSize() const                  { return m_triBatchSize; }
    int   getNodeBatchSize() const                      { return m_nodeBatchSize; }
    void  setTriangleBatchSize(int triBatchSize)        { m_triBatchSize = triBatchSize; }
    void  setNodeBatchSize(int nodeBatchSize)           { m_nodeBatchSize= nodeBatchSize; }
    int   roundToTriangleBatchSize(int n) const         { return ((n+m_triBatchSize-1)/m_triBatchSize)*m_triBatchSize; }
    int   roundToNodeBatchSize(int n) const             { return ((n+m_nodeBatchSize-1)/m_nodeBatchSize)*m_nodeBatchSize; }

    // leaf preferences
    void  setLeafPreferences(int minSize,int maxSize)   { m_minLeafSize=minSize; m_maxLeafSize=maxSize; }
    int   getMinLeafSize() const                        { return m_minLeafSize; }
    int   getMaxLeafSize() const                        { return m_maxLeafSize; }

//    U32   computeHash() const                           { return hashBits(hash<String>(m_name), floatToBits(m_SAHNodeCost), floatToBits(m_SAHTriangleCost), hashBits(m_triBatchSize, m_nodeBatchSize, m_minLeafSize, m_maxLeafSize)); }

private:
    string  m_name;
    float   m_SAHNodeCost;
    float   m_SAHTriangleCost;
    int     m_triBatchSize;
    int     m_nodeBatchSize;
    int     m_minLeafSize;
    int     m_maxLeafSize;
};

