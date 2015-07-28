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
#ifndef UTIL_H
#define UTIL_H

#include <limits>
using namespace std;

#ifndef HAVE_CUDA
struct int3
{
    int x;
    int y;
    int z;
};
struct float3
{
    float x;
    float y;
    float z;
};
#endif
namespace FW
{
#define FW_SPECIALIZE_MINMAX(TEMPLATE, T, MIN, MAX) \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b) { return MIN; } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b) { return MAX; } \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b, T c) { return min(min(a, b), c); } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b, T c) { return max(max(a, b), c); } \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b, T c, T d) { return min(min(min(a, b), c), d); } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b, T c, T d) { return max(max(max(a, b), c), d); } \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b, T c, T d, T e) { return min(min(min(min(a, b), c), d), e); } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b, T c, T d, T e) { return max(max(max(max(a, b), c), d), e); } \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b, T c, T d, T e, T f) { return min(min(min(min(min(a, b), c), d), e), f); } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b, T c, T d, T e, T f) { return max(max(max(max(max(a, b), c), d), e), f); } \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b, T c, T d, T e, T f, T g) { return min(min(min(min(min(min(a, b), c), d), e), f), g); } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b, T c, T d, T e, T f, T g) { return max(max(max(max(max(max(a, b), c), d), e), f), g); } \
    TEMPLATE EAVL_HOSTDEVICE T min(T a, T b, T c, T d, T e, T f, T g, T h) { return min(min(min(min(min(min(min(a, b), c), d), e), f), g), h); } \
    TEMPLATE EAVL_HOSTDEVICE T max(T a, T b, T c, T d, T e, T f, T g, T h) { return max(max(max(max(max(max(max(a, b), c), d), e), f), g), h); } \
    TEMPLATE EAVL_HOSTDEVICE T clamp(T v, T lo, T hi) { return min(max(v, lo), hi); }

FW_SPECIALIZE_MINMAX(template <class T>, T&, (a < b) ? a : b, (a > b) ? a : b)
FW_SPECIALIZE_MINMAX(template <class T>, const T&, (a < b) ? a : b, (a > b) ? a : b)

EAVL_HOSTDEVICE int      abs   (int a)         { return (a >= 0) ? a : -a; }
EAVL_HOSTDEVICE float    abs   (float a)       { return ::fabsf(a); }


template <class T, int L> class Vector;

template <class T, int L, class S> class VectorBase
{
public:
    EAVL_HOSTDEVICE                    VectorBase  (void)                      {}

    EAVL_HOSTDEVICE    const T*        getPtr      (void) const                { return ((S*)this)->getPtr(); }
    EAVL_HOSTDEVICE    T*              getPtr      (void)                      { return ((S*)this)->getPtr(); }
    EAVL_HOSTDEVICE    const T&        get         (int idx) const             {  return getPtr()[idx]; }
    EAVL_HOSTDEVICE    T&              get         (int idx)                   {  return getPtr()[idx]; }
    EAVL_HOSTDEVICE    T               set         (int idx, const T& a)       { T& slot = get(idx); T old = slot; slot = a; return old; }

    EAVL_HOSTDEVICE    void            set         (const T& a)                { T* tp = getPtr(); for (int i = 0; i < L; i++) tp[i] = a; }
    EAVL_HOSTDEVICE    void            set         (const T* ptr)              { T* tp = getPtr(); for (int i = 0; i < L; i++) tp[i] = ptr[i]; }
    EAVL_HOSTDEVICE    void            setZero     (void)                      { set((T)0); }
    EAVL_HOSTDEVICE    bool            isZero      (void) const                { const T* tp = getPtr(); for (int i = 0; i < L; i++) if (tp[i] != (T)0) return false; return true; }
    EAVL_HOSTDEVICE    T               lenSqr      (void) const                { const T* tp = getPtr(); T r = (T)0; for (int i = 0; i < L; i++) r += sqr(tp[i]); return r; }
    EAVL_HOSTDEVICE    T               length      (void) const                { return sqrt(lenSqr()); }
    EAVL_HOSTDEVICE    S               normalized  (T len = (T)1) const        { return operator*(len * rcp(length())); }
    EAVL_HOSTDEVICE    void            normalize   (T len = (T)1)              { set(normalized(len)); }
    EAVL_HOSTDEVICE    T               min         (void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r = FW::min(r, tp[i]); return r; }
    EAVL_HOSTDEVICE    T               max         (void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r = FW::max(r, tp[i]); return r; }
    EAVL_HOSTDEVICE    T               sum         (void) const                { const T* tp = getPtr(); T r = tp[0]; for (int i = 1; i < L; i++) r += tp[i]; return r; }
    EAVL_HOSTDEVICE    S               abs         (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::abs(tp[i]); return r; }

    EAVL_HOSTDEVICE    Vector<T, L + 1> toHomogeneous(void) const              { const T* tp = getPtr(); Vector<T, L + 1> r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i]; rp[L] = (T)1; return r; }
    EAVL_HOSTDEVICE    Vector<T, L - 1> toCartesian(void) const                { const T* tp = getPtr(); Vector<T, L - 1> r; T* rp = r.getPtr(); T c = rcp(tp[L - 1]); for (int i = 0; i < L - 1; i++) rp[i] = tp[i] * c; return r; }

    EAVL_HOSTDEVICE    const T&        operator[]  (int idx) const             { return get(idx); }
    EAVL_HOSTDEVICE    T&              operator[]  (int idx)                   { return get(idx); }
    EAVL_HOSTDEVICE    S&              operator=   (const T& a)                { set(a); return *(S*)this; }
    EAVL_HOSTDEVICE    S               operator+   (void) const                { return *this; }
    EAVL_HOSTDEVICE    S               operator-   (void) const                { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = -tp[i]; return r; }

    EAVL_HOSTDEVICE    S               operator+   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] + a; return r; }
    EAVL_HOSTDEVICE    S               operator-   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] - a; return r; }
    EAVL_HOSTDEVICE    S               operator*   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] * a; return r; }
    EAVL_HOSTDEVICE    S               operator/   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] / a; return r; }
    EAVL_HOSTDEVICE    S               operator%   (const T& a) const          { const T* tp = getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] % a; return r; }

    template <class V> EAVL_HOSTDEVICE void    set         (const VectorBase<T, L, V>& v)          { set(v.getPtr()); }
    template <class V> EAVL_HOSTDEVICE T       dot         (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); T r = (T)0; for (int i = 0; i < L; i++) r += tp[i] * vp[i]; return r; }
    template <class V> EAVL_HOSTDEVICE S       min         (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::min(tp[i], vp[i]); return r; }
    template <class V> EAVL_HOSTDEVICE S       max         (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::max(tp[i], vp[i]); return r; }
    template <class V, class W> EAVL_HOSTDEVICE S clamp    (const VectorBase<T, L, V>& lo, const VectorBase<T, L, W>& hi) const { const T* tp = getPtr(); const T* lop = lo.getPtr(); const T* hip = hi.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = FW::clamp(tp[i], lop[i], hip[i]); return r; }

    template <class V> EAVL_HOSTDEVICE S&      operator=   (const VectorBase<T, L, V>& v)          { set(v); return *(S*)this; }


    template <class V> EAVL_HOSTDEVICE S       operator+   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] +  vp[i]; return r; }
    template <class V> EAVL_HOSTDEVICE S       operator-   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] -  vp[i]; return r; }
    template <class V> EAVL_HOSTDEVICE S       operator*   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] *  vp[i]; return r; }
    template <class V> EAVL_HOSTDEVICE S       operator/   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] /  vp[i]; return r; }
    template <class V> EAVL_HOSTDEVICE S       operator%   (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = tp[i] %  vp[i]; return r; }

    template <class V> EAVL_HOSTDEVICE bool    operator==  (const VectorBase<T, L, V>& v) const    { const T* tp = getPtr(); const T* vp = v.getPtr(); for (int i = 0; i < L; i++) if (tp[i] != vp[i]) return false; return true; }
    template <class V> EAVL_HOSTDEVICE bool    operator!=  (const VectorBase<T, L, V>& v) const    { return (!operator==(v)); }
};

//------------------------------------------------------------------------

template <class T, int L> class Vector : public VectorBase<T, L, Vector<T, L> >
{
public:
    EAVL_HOSTDEVICE                    Vector      (void)                      {  this->setZero(); }
    EAVL_HOSTDEVICE                    Vector      (T a)                       {  this->set(a); }

    EAVL_HOSTDEVICE    const T*        getPtr      (void) const                { return m_values; }
    EAVL_HOSTDEVICE    T*              getPtr      (void)                      { return m_values; }
    static EAVL_HOSTDEVICE Vector      fromPtr     (const T* ptr)              { Vector v; v.set(ptr); return v; }

    template <class V> EAVL_HOSTDEVICE Vector(const VectorBase<T, L, V>& v) { this->set(v); }
    template <class V> EAVL_HOSTDEVICE Vector& operator=(const VectorBase<T, L, V>& v) { this->set(v); return *this; }

private:
    T               m_values[L];
};


//------------------------------------------------------------------------

class Vec3i : public VectorBase<int, 3, Vec3i>, public int3
{
public:
    EAVL_HOSTDEVICE                    Vec3i       (void)                      { setZero(); }
    EAVL_HOSTDEVICE                    Vec3i       (int a)                     { set(a); }
    EAVL_HOSTDEVICE                    Vec3i       (int xx, int yy, int zz)    { x = xx; y = yy; z = zz; }
    EAVL_HOSTDEVICE    const int*      getPtr      (void) const                { return &x; }
    EAVL_HOSTDEVICE    int*            getPtr      (void)                      { return &x; }
    static EAVL_HOSTDEVICE Vec3i       fromPtr     (const int* ptr)            { return Vec3i(ptr[0], ptr[1], ptr[2]); }
    template <class V> EAVL_HOSTDEVICE Vec3i(const VectorBase<int, 3, V>& v) { set(v); }
    template <class V> EAVL_HOSTDEVICE Vec3i& operator=(const VectorBase<int, 3, V>& v) { set(v); return *this; }
};
//------------------------------------------------------------------------

class Vec3f : public VectorBase<float, 3, Vec3f>, public float3
{
public:
    EAVL_HOSTDEVICE                    Vec3f       (void)                      { setZero(); }
    EAVL_HOSTDEVICE                    Vec3f       (float a)                     { set(a); }
    EAVL_HOSTDEVICE                    Vec3f       (float xx, float yy, float zz)    { x = xx; y = yy; z = zz; }
    EAVL_HOSTDEVICE                    Vec3f       (const Vec3i& v)            { x = (float)v.x; y = (float)v.y; z = (float)v.z; }

    EAVL_HOSTDEVICE    const float*      getPtr      (void) const                { return &x; }
    EAVL_HOSTDEVICE    float*            getPtr      (void)                      { return &x; }
    static EAVL_HOSTDEVICE Vec3f       fromPtr     (const float* ptr)            { return Vec3f(ptr[0], ptr[1], ptr[2]); }

    EAVL_HOSTDEVICE    operator        Vec3i       (void) const                { return Vec3i((int)x, (int)y, (int)z); }
   

    EAVL_HOSTDEVICE    Vec3f           cross       (const Vec3f& v) const      { return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

    template <class V> EAVL_HOSTDEVICE Vec3f(const VectorBase<float, 3, V>& v) { set(v); }
    template <class V> EAVL_HOSTDEVICE Vec3f& operator=(const VectorBase<float, 3, V>& v) { set(v); return *this; }
};

//------------------------------------------------------------------------

template <class T, int L, class S> EAVL_HOSTDEVICE T lenSqr    (const VectorBase<T, L, S>& v)                  { return v.lenSqr(); }
template <class T, int L, class S> EAVL_HOSTDEVICE T length    (const VectorBase<T, L, S>& v)                  { return v.length(); }
template <class T, int L, class S> EAVL_HOSTDEVICE S normalize (const VectorBase<T, L, S>& v, T len = (T)1)    { return v.normalized(len); }
template <class T, int L, class S> EAVL_HOSTDEVICE T min       (const VectorBase<T, L, S>& v)                  { return v.min(); }
template <class T, int L, class S> EAVL_HOSTDEVICE T max       (const VectorBase<T, L, S>& v)                  { return v.max(); }
template <class T, int L, class S> EAVL_HOSTDEVICE T sum       (const VectorBase<T, L, S>& v)                  { return v.sum(); }
template <class T, int L, class S> EAVL_HOSTDEVICE S abs       (const VectorBase<T, L, S>& v)                  { return v.abs(); }

template <class T, int L, class S> EAVL_HOSTDEVICE S operator+     (const T& a, const VectorBase<T, L, S>& b)  { return b + a; }
template <class T, int L, class S> EAVL_HOSTDEVICE S operator-     (const T& a, const VectorBase<T, L, S>& b)  { return -b + a; }
template <class T, int L, class S> EAVL_HOSTDEVICE S operator*     (const T& a, const VectorBase<T, L, S>& b)  { return b * a; }
template <class T, int L, class S> EAVL_HOSTDEVICE S operator/     (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a / bp[i]; return r; }
template <class T, int L, class S> EAVL_HOSTDEVICE S operator%     (const T& a, const VectorBase<T, L, S>& b)  { const T* bp = b.getPtr(); S r; T* rp = r.getPtr(); for (int i = 0; i < L; i++) rp[i] = a % bp[i]; return r; }

template <class T, int L, class S, class V> EAVL_HOSTDEVICE T dot(const VectorBase<T, L, S>& a, const VectorBase<T, L, V>& b) { return a.dot(b); }

EAVL_HOSTDEVICE Vec3f  cross           (const Vec3f& a, const Vec3f& b)    { return a.cross(b); }

#define MINMAX(T) \
    EAVL_HOSTDEVICE T min(const T& a, const T& b)                          { return a.min(b); } \
    EAVL_HOSTDEVICE T min(T& a, T& b)                                      { return a.min(b); } \
    EAVL_HOSTDEVICE T max(const T& a, const T& b)                          { return a.max(b); } \
    EAVL_HOSTDEVICE T max(T& a, T& b)                                      { return a.max(b); } \
    EAVL_HOSTDEVICE T min(const T& a, const T& b, const T& c)              { return a.min(b).min(c); } \
    EAVL_HOSTDEVICE T min(T& a, T& b, T& c)                                { return a.min(b).min(c); } \
    EAVL_HOSTDEVICE T max(const T& a, const T& b, const T& c)              { return a.max(b).max(c); } \
    EAVL_HOSTDEVICE T max(T& a, T& b, T& c)                                { return a.max(b).max(c); } \
    EAVL_HOSTDEVICE T min(const T& a, const T& b, const T& c, const T& d)  { return a.min(b).min(c).min(d); } \
    EAVL_HOSTDEVICE T min(T& a, T& b, T& c, T& d)                          { return a.min(b).min(c).min(d); } \
    EAVL_HOSTDEVICE T max(const T& a, const T& b, const T& c, const T& d)  { return a.max(b).max(c).max(d); } \
    EAVL_HOSTDEVICE T max(T& a, T& b, T& c, T& d)                          { return a.max(b).max(c).max(d); } \
    EAVL_HOSTDEVICE T clamp(const T& v, const T& lo, const T& hi)          { return v.clamp(lo, hi); } \
    EAVL_HOSTDEVICE T clamp(T& v, T& lo, T& hi)                            { return v.clamp(lo, hi); }

MINMAX(Vec3i)
MINMAX(Vec3f) 
#undef MINMAX

class AABB
{
public:
    EAVL_HOSTDEVICE                    AABB        (void) : m_mn(10000000, 10000000, 10000000), m_mx(-10000000, -10000000, -10000000) {}
    EAVL_HOSTDEVICE                    AABB        (const Vec3f& mn, const Vec3f& mx) : m_mn(mn), m_mx(mx) {}

    EAVL_HOSTDEVICE    void            grow        (const Vec3f& pt)   { m_mn = m_mn.min(pt); m_mx = m_mx.max(pt); }
    EAVL_HOSTDEVICE    void            grow        (const AABB& aabb)  { grow(aabb.m_mn); grow(aabb.m_mx); }
    EAVL_HOSTDEVICE    void            intersect   (const AABB& aabb)  { m_mn = m_mn.max(aabb.m_mn); m_mx = m_mx.min(aabb.m_mx); }
    EAVL_HOSTDEVICE    float           volume      (void) const        { if(!valid()) return 0.0f; return (m_mx.x-m_mn.x) * (m_mx.y-m_mn.y) * (m_mx.z-m_mn.z); }
    EAVL_HOSTDEVICE    float           area        (void) const        { if(!valid()) return 0.0f; Vec3f d = m_mx - m_mn; return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f; }
    EAVL_HOSTDEVICE    bool            valid       (void) const        { return m_mn.x<=m_mx.x && m_mn.y<=m_mx.y && m_mn.z<=m_mx.z; }
    EAVL_HOSTDEVICE    Vec3f           midPoint    (void) const        { return (m_mn+m_mx)*0.5f; }
    EAVL_HOSTDEVICE    const Vec3f&    min         (void) const        { return m_mn; }
    EAVL_HOSTDEVICE    const Vec3f&    max         (void) const        { return m_mx; }
    EAVL_HOSTDEVICE    Vec3f&          min         (void)              { return m_mn; }
    EAVL_HOSTDEVICE    Vec3f&          max         (void)              { return m_mx; }

    EAVL_HOSTDEVICE    AABB            operator+   (const AABB& aabb) const { AABB u(*this); u.grow(aabb); return u; }

private:
    Vec3f           m_mn;
    Vec3f           m_mx;
};

}//namespace FW

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
};






typedef bool    (*SortCompareFunc) (void* data, int idxA, int idxB);    // Returns true if A should come before B.
typedef void    (*SortSwapFunc)    (void* data, int idxA, int idxB);    // Swaps A and B.

#define QSORT_STACK_SIZE    32
#define QSORT_MIN_SIZE      16

inline void insertionSort(int start, int size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
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
inline int median3(int low, int high, void* data, SortCompareFunc compareFunc)
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
inline int partition(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
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
inline void qsort(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
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

inline void sort(void* data, int start, int end, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    // Nothing to do => skip.
    if (end - start < 2)
        return;

    // Single-core.
    qsort(start, end, data, compareFunc, swapFunc);
}

#endif

