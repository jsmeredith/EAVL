// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_OP_DISPATCH_H
#define EAVL_OP_DISPATCH_H

#include "eavl.h"
#include "eavlTuple.h"
#include "eavlIndexable.h"
#include "eavlTupleTraits.h"

// ****************************************************************************
// Function:  eavlOpDispatch
//
// Purpose:
///   This function is used internally by iterators.  Its basic purpose is
///   to take groups of input eavlArrays and translate them to explicit
///   raw pointers -- either host pointers, or device pointers after any
///   necessary data transfers -- for calling a kernel.
///
///   If an input array is the eavlArray base class, this will pre-generate
///   code (at compile time) for both float* and int* versions of the
///   kernel and choose the proper path at runtime when the type is
///   known.  If an input array is an eavlConcrete array, it will only
///   create a version for the known-at-compile-time base type.  So
///   try to pass in concrete eavlArrays when possible to minimize the
///   multiplicity of paths needed to be generated.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************

// utilities for reversing a tuple
// (the dispatch process reverses the input arrays,
// so we want to reverse them back before handing
// them to the operation itself.)
template<class RT, class IT, class OT>
struct reverseclass
{
    static inline RT getreversed(IT &i, OT o)
    {
        typedef typename IT::resttype NEW_IT;
        typedef cons<typename IT::firsttype,OT> NEW_OT;
        return reverseclass<RT,NEW_IT,NEW_OT>::getreversed(i.rest, NEW_OT(i.first, o));
    }
};

template<class RT, class IFT, class OT>
struct reverseclass<RT, cons<IFT,nulltype>, OT>
{
    static inline RT getreversed(cons<IFT,nulltype> &i, OT o)
    {
        return cons<IFT,OT>(i.first, o);
    }
};

template<class FT, class RT>
inline typename traits<cons<FT,RT> >::reversetype reversed(cons<FT,RT> &t)
{
    return reverseclass<typename traits<cons<FT,RT> >::reversetype, cons<FT,RT>, nulltype>::getreversed(t, cnull());
}


// actual dispatch code
template <int N,
          class K,
          class S,
          class Z0F, class Z0R,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclassgetrawptr;

template <int N,
          class K,
          class S,
          class Z0F, class Z0R,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_dropfirst;

// entry point for recursion, and where we start when we cycle the args
template <int N,
          class K,
          class S,
          class Z0F, class Z0R,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_start
{
    static void go(int n, S &structure,
                   cons<Z0F,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   cons<Z3F,Z3R> &args3,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        dispatchclassgetrawptr<N, K, S, Z0F, Z0R, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, RZ0, RZ1, RZ2, RZ3, F>
            ::go(n, structure, args0, args1, args2, args3, ptrs0, ptrs1, ptrs2, ptrs3, functor);
    }
};

// base cases for final recursion: all ZFs and ZRs are nulltype
//    - single-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_start<1, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, RZ3, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        K::call(n, structure, reversed(ptrs3), functor);
    }
};

//    - double-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_start<2, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, RZ3, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        K::call(n, structure, reversed(ptrs2), reversed(ptrs3), functor);
    }
};

//    - triple-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_start<3, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, RZ3, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        K::call(n, structure, reversed(ptrs1), reversed(ptrs2), reversed(ptrs3), functor);
    }
};

//    - quadruple-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_start<4, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, RZ3, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        K::call(n, structure, reversed(ptrs0), reversed(ptrs1), reversed(ptrs2), reversed(ptrs3), functor);
    }
};

//
// recurse odd-phase: extract the head from the zip, get its raw pointer, and add to the raw ptr cons
//

// (this non-specialized version assumed it's a concrete array, not a base class)
template <int N,
          class K,
          class S,
          class Z0F, class Z0R,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclassgetrawptr
{
    static void go(int n, S &structure,
                   cons<Z0F,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   cons<Z3F,Z3R> &args3,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        typedef typename Z0F::type::type rawtype;
        rawtype *raw = (rawtype*)((K::location()==eavlArray::HOST) ? args0.first.array->GetHostArray() : args0.first.array->GetCUDAArray());
        typedef cons<eavlIndexable<rawtype>, RZ0> newp;
        dispatchclass_dropfirst<N, K, S, Z0F, Z0R, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, newp, RZ1, RZ2, RZ3, F>
            ::go(n, structure, args0, args1, args2, args3, newp(eavlIndexable<rawtype>(raw, args0.first.indexer), ptrs0), ptrs1, ptrs2, ptrs3, functor);
    }
};

// special case for recursion getrawptr: Z0F is base class eavlArray*
template <int N,
          class K,
          class S,
          class Z0R,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclassgetrawptr<N, K, S, eavlIndexable<eavlArray>, Z0R, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, RZ0, RZ1, RZ2, RZ3, F>
{
    static void go(int n, S &structure,
                   cons<eavlIndexable<eavlArray>,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   cons<Z3F,Z3R> &args3,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        eavlConcreteArray<int> *ai = dynamic_cast<eavlConcreteArray<int>*>(args0.first.array);
        eavlConcreteArray<float> *af = dynamic_cast<eavlConcreteArray<float>*>(args0.first.array);
        if (ai)
        {
            int *raw = (int*)((K::location()==eavlArray::HOST) ? ai->GetHostArray() : ai->GetCUDAArray());
            typedef cons<eavlIndexable<int>, RZ0> newp;
            dispatchclass_dropfirst<N, K, S, eavlIndexable<eavlArray>, Z0R, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, newp, RZ1, RZ2, RZ3, F>
                ::go(n, structure, args0, args1, args2, args3, newp(eavlIndexable<int>(raw,args0.first.indexer), ptrs0), ptrs1, ptrs2, ptrs3, functor);
        }
        if (af)
        {
            float *raw = (float*)((K::location()==eavlArray::HOST) ? af->GetHostArray() : af->GetCUDAArray());
            typedef cons<eavlIndexable<float>,RZ0> newp;
            dispatchclass_dropfirst<N, K, S, eavlIndexable<eavlArray>, Z0R, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, newp, RZ1, RZ2, RZ3, F>
                ::go(n, structure, args0, args1, args2, args3, newp(eavlIndexable<float>(raw,args0.first.indexer), ptrs0), ptrs1, ptrs2, ptrs3, functor);
        }
    }
};

//
// recurse even-phase: drop the head element from the input zip
//

template <int N,
          class K,
          class S,
          class Z0F, class Z0R,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_dropfirst
{
    static void go(int n, S &structure,
                   cons<Z0F,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   cons<Z3F,Z3R> &args3,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        dispatchclassgetrawptr<N, K, S, typename Z0R::firsttype, typename Z0R::resttype,
            Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, RZ0, RZ1, RZ2, RZ3, F>
            ::go(n, structure, args0.rest, args1, args2, args3, ptrs0, ptrs1, ptrs2, ptrs3, functor);
    }
};


// base case for recursion: ZR is nulltype; rotate args and continue
template <int N,
          class K,
          class S,
          class Z0F,
          class Z1F, class Z1R,
          class Z2F, class Z2R,
          class Z3F, class Z3R,
          class RZ0,
          class RZ1,
          class RZ2,
          class RZ3,
          class F>
struct dispatchclass_dropfirst<N, K, S, Z0F, nulltype, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, RZ0, RZ1, RZ2, RZ3, F>
{
    static void go(int n, S &structure,
                   cons<Z0F,nulltype> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   cons<Z3F,Z3R> &args3,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   RZ3 ptrs3,
                   F &functor)
    {
        cons<nulltype,nulltype> empty;
        dispatchclass_start<N+1, K, S, Z1F, Z1R, Z2F, Z2R, Z3F, Z3R, nulltype, nulltype, RZ1, RZ2, RZ3, RZ0, F>
            ::go(n, structure, args1, args2, args3, empty, ptrs1, ptrs2, ptrs3, ptrs0, functor);
    }
};

// entry points for dispatch
// 4-arg
template<class K, class S, class T0, class T1, class T2, class T3, class F>
void eavlOpDispatch(int n, S &structure, T0 arrays0, T1 arrays1, T2 arrays2, T3 arrays3, F functor)
{
    dispatchclassgetrawptr<0, K, S,
        typename T0::firsttype, typename T0::resttype,
        typename T1::firsttype, typename T1::resttype,
        typename T2::firsttype, typename T2::resttype,
        typename T3::firsttype, typename T3::resttype,
        nulltype,
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, arrays1, arrays2, arrays3, cnull(), cnull(), cnull(), cnull(), functor);
}

// 3-arg
template<class K, class S, class T0, class T1, class T2, class F>
void eavlOpDispatch(int n, S &structure, T0 arrays0, T1 arrays1, T2 arrays2, F functor)
{
    cons<nulltype,nulltype> empty;
    dispatchclassgetrawptr<0, K, S,
        typename T0::firsttype, typename T0::resttype,
        typename T1::firsttype, typename T1::resttype,
        typename T2::firsttype, typename T2::resttype,
        nulltype, nulltype,
        nulltype,
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, arrays1, arrays2, empty, cnull(), cnull(), cnull(), cnull(), functor);
}

// 2-arg
template<class K, class S, class T0, class T1, class F>
void eavlOpDispatch(int n, S &structure, T0 arrays0, T1 arrays1, F functor)
{
    cons<nulltype,nulltype> empty;
    dispatchclassgetrawptr<0, K, S,
        typename T0::firsttype, typename T0::resttype,
        typename T1::firsttype, typename T1::resttype,
        nulltype, nulltype,
        nulltype, nulltype,
        nulltype,
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, arrays1, empty, empty, cnull(), cnull(), cnull(), cnull(), functor);
}

// 1-arg
template<class K, class S, class T0, class F>
void eavlOpDispatch(int n, S &structure, T0 arrays0, F functor)
{
    cons<nulltype,nulltype> empty;
    dispatchclassgetrawptr<0, K, S,
        typename T0::firsttype, typename T0::resttype,
        nulltype, nulltype,
        nulltype, nulltype,
        nulltype, nulltype,
        nulltype,
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, empty, empty, empty, cnull(), cnull(), cnull(), cnull(), functor);
}

#endif
