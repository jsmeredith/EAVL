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
//Adapted



#ifndef UTIL_H
#define UTIL_H
EAVL_HOSTDEVICE eavlVector3  vector3min(const eavlVector3 &r,const eavlVector3 &u)
{
    eavlVector3 result;
    result.x = (u.x > r.x) ? r.x : u.x ;
    result.y = (u.y > r.y) ? r.y : u.y ;
    result.z = (u.z > r.z) ? r.z : u.z ;
    
    return result;
}

EAVL_HOSTDEVICE eavlVector3  vector3max(const eavlVector3 &r,const eavlVector3 &u)
{
    eavlVector3 result;
    result.x = (u.x < r.x) ? r.x : u.x ;
    result.y = (u.y < r.y) ? r.y : u.y ;
    result.z = (u.z < r.z) ? r.z : u.z ;
    
    return result;
}
#include <limits>
using namespace std;
class AABB
{
public:
    EAVL_HOSTDEVICE                    		 AABB        (void) : m_mn(10000000, 10000000, 10000000), m_mx(-10000000, -10000000, -10000000) {}
    EAVL_HOSTDEVICE                    		 AABB        (const eavlVector3& mn, const eavlVector3& mx) : m_mn(mn), m_mx(mx) {}

    EAVL_HOSTDEVICE    void            		 grow        (const eavlVector3& pt)   { m_mn = vector3min(m_mn,pt); m_mx = vector3max(m_mx,pt); }
    EAVL_HOSTDEVICE    void            		 grow        (const AABB& aabb)  
    { 
        m_mn.x=(aabb.min().x< m_mn.x) ? aabb.min().x : m_mn.x;
        m_mn.y=(aabb.min().y< m_mn.y) ? aabb.min().y : m_mn.y;
        m_mn.z=(aabb.min().z< m_mn.z) ? aabb.min().z : m_mn.z;

        m_mx.x=(aabb.max().x> m_mx.x) ? aabb.max().x : m_mx.x;
        m_mx.y=(aabb.max().y> m_mx.y) ? aabb.max().y : m_mx.y;
        m_mx.z=(aabb.max().z> m_mx.z) ? aabb.max().z : m_mx.z;

    }
    EAVL_HOSTDEVICE    void            		 intersect   (const AABB& aabb)  { m_mn = vector3max(m_mn,aabb.m_mn); m_mx = vector3min(m_mx,aabb.m_mx); }
    EAVL_HOSTDEVICE    float           		 volume      (void) const        { if(!valid()) return 0.0f; return (m_mx.x-m_mn.x) * (m_mx.y-m_mn.y) * (m_mx.z-m_mn.z); }
    EAVL_HOSTDEVICE    float           		 area        (void) const        { if(!valid()) return 0.0f; eavlVector3 d = m_mx - m_mn; return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f; }
    EAVL_HOSTDEVICE    bool            		 valid       (void) const        { return m_mn.x<=m_mx.x && m_mn.y<=m_mx.y && m_mn.z<=m_mx.z; }
    EAVL_HOSTDEVICE    eavlVector3           midPoint    (void) const        { return (m_mn+m_mx)*0.5f; }
    EAVL_HOSTDEVICE    const eavlVector3&    min         (void) const        { return m_mn; }
    EAVL_HOSTDEVICE    const eavlVector3&    max         (void) const        { return m_mx; }
    EAVL_HOSTDEVICE    eavlVector3&          min         (void)              { return m_mn; }
    EAVL_HOSTDEVICE    eavlVector3&          max         (void)              { return m_mx; }

    EAVL_HOSTDEVICE    AABB            operator+   (const AABB& aabb) const { AABB u(*this); u.grow(aabb); return u; }

private:
    eavlVector3           m_mn;
    eavlVector3           m_mx;
};

struct Stats
{
    Stats()             { clear(); }
    void clear()        { memset(this, 0, sizeof(Stats)); }
    void print() const  { printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf\n", branchingFactor,numLeafNodes+numInnerNodes, numLeafNodes,numInnerNodes, SAHCost, 1.f*numChildNodes/max(numInnerNodes,1), 1.f*numTris/max(numLeafNodes,1)); }

    float   SAHCost;
    int     branchingFactor;
    int     numInnerNodes;
    int     numLeafNodes;
    int     numChildNodes;
    int     numTris;
};

struct BuildParams
{
    Stats*      stats;
    bool        enablePrints;
    float         splitAlpha;     // spatial split area threshold

    BuildParams(void)
    {
        stats           = NULL;
        enablePrints    = true;
        splitAlpha      = 1.0e-5f;
    }

    //U32 computeHash(void) const
    //{
    //    return hashBits(floatToBits(splitAlpha));
    //}
};


template <class A, class B> EAVL_HOSTDEVICE A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }

//template <class T> EAVL_HOSTDEVICE void swap(T& a, T& b) { T t = a; a = b; b = t; }
template <class T> EAVL_HOSTDEVICE T clamp(T v, T lo, T hi) { return min(max(v, lo), hi); }


typedef bool    (*SortCompareFunc) (void* data, int idxA, int idxB);    // Returns true if A should come before B.
typedef void    (*SortSwapFunc)    (void* data, int idxA, int idxB);    // Swaps A and B.

#define QSORT_STACK_SIZE    32
#define QSORT_MIN_SIZE      16

void insertionSort(int start, int size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    //FW_ASSERT(compareFunc && swapFunc);
    //FW_ASSERT(size >= 0);

    for (int i = 1; i < size; i++)
    {
        int j = start + i - 1;
        while (j >= start && compareFunc(data, j + 1, j))
        {
            swapFunc(data, j, j + 1);
            j--;
        }
    }
}
int median3(int low, int high, void* data, SortCompareFunc compareFunc)
{
    //FW_ASSERT(compareFunc);
    //FW_ASSERT(low >= 0 && high >= 2);

    int l = low;
    int c = (low + high) >> 1;
    int h = high - 2;

    if (compareFunc(data, h, l)) swap(l, h);
    if (compareFunc(data, c, l)) c = l;
    return (compareFunc(data, h, c)) ? h : c;
}
int partition(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    // Select pivot using median-3, and hide it in the highest entry.

    swapFunc(data, median3(low, high, data, compareFunc), high - 1);

    // Partition data.

    int i = low - 1;
    int j = high - 1;
    for (;;)
    {
        do
            i++;
        while (compareFunc(data, i, high - 1));
        do
            j--;
        while (compareFunc(data, high - 1, j));

        //FW_ASSERT(i >= low && j >= low && i < high && j < high);
        if (i >= j)
            break;

        swapFunc(data, i, j);
    }

    // Restore pivot.

    swapFunc(data, i, high - 1);
    return i;
}
void qsort(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{

    int stack[QSORT_STACK_SIZE];
    int sp = 0;
    stack[sp++] = high;

    while (sp)
    {
        high = stack[--sp];

        // Small enough or stack full => use insertion sort.

        if (high - low < QSORT_MIN_SIZE || sp + 2 > QSORT_STACK_SIZE)
        {
            insertionSort(low, high - low, data, compareFunc, swapFunc);
            low = high + 1;
            continue;
        }

        // Partition and sort sub-partitions.

        int i = partition(low, high, data, compareFunc, swapFunc);
        if (high - i > 2)
            stack[sp++] = high;
        if (i - low > 1)
            stack[sp++] = i;
        else
            low = i + 1;
    }
}

void sort(void* data, int start, int end, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{

    // Nothing to do => skip.

    if (end - start < 2)
        return;

    // Single-core.
    qsort(start, end, data, compareFunc, swapFunc);
}

#endif

