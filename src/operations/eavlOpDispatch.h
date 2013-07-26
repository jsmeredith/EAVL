// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_OP_DISPATCH_H
#define EAVL_OP_DISPATCH_H

#include "eavl.h"
#include "eavlTuple.h"

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
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclassgetrawptr;

template <int N,
          class K,
          class S,
          class Z0F, class Z0R, 
          class Z1F, class Z1R,
          class Z2F, class Z2R, 
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_dropfirst;

// entry point for recursion, and where we start when we cycle the args
template <int N, 
          class K,
          class S,
          class Z0F, class Z0R, 
          class Z1F, class Z1R,
          class Z2F, class Z2R, 
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_start
{
    static void go(int n, S &structure,
                   cons<Z0F,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        dispatchclassgetrawptr<N, K, S, Z0F, Z0R, Z1F, Z1R, Z2F, Z2R, RZ0, RZ1, RZ2, F>
            ::go(n, structure, args0, args1, args2, ptrs0, ptrs1, ptrs2, functor);
    }
};

// base cases for final recursion: all ZFs and ZRs are nulltype
//    - single-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_start<1, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        K::call(n, reversed(ptrs2), functor);
    }
};

//    - double-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_start<2, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        K::call(n, reversed(ptrs1), reversed(ptrs2), functor);
    }
};

//    - triple-arg group
template <class K,
          class S,
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_start<3, K, S, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, RZ0, RZ1, RZ2, F>
{
    static void go(int n, S &structure,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   cons<nulltype,nulltype> &,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        K::call(n, reversed(ptrs0), reversed(ptrs1), reversed(ptrs2), functor);
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
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclassgetrawptr
{
    static void go(int n, S &structure,
                   cons<Z0F,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        typedef typename Z0F::type::type rawtype;
        rawtype *raw = (rawtype*)((K::location()==eavlArray::HOST) ? args0.first.array->GetHostArray() : args0.first.array->GetCUDAArray());
        typedef cons<eavlIndexable<rawtype>, RZ0> newp;
        dispatchclass_dropfirst<N, K, S, Z0F, Z0R, Z1F, Z1R, Z2F, Z2R, newp, RZ1, RZ2, F>
            ::go(n, structure, args0, args1, args2, newp(eavlIndexable<rawtype>(raw, args0.first.indexer), ptrs0), ptrs1, ptrs2, functor);
    }
};

// special case for recursion getrawptr: Z0F is base class eavlArray*
template <int N, 
          class K,
          class S,
          class Z0R, 
          class Z1F, class Z1R,
          class Z2F, class Z2R, 
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclassgetrawptr<N, K, S, eavlIndexable<eavlArray>, Z0R, Z1F, Z1R, Z2F, Z2R, RZ0, RZ1, RZ2, F>
{
    static void go(int n, S &structure,
                   cons<eavlIndexable<eavlArray>,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        eavlConcreteArray<int> *ai = dynamic_cast<eavlConcreteArray<int>*>(args0.first.array);
        eavlConcreteArray<float> *af = dynamic_cast<eavlConcreteArray<float>*>(args0.first.array);
        if (ai)
        {
            int *raw = (int*)((K::location()==eavlArray::HOST) ? ai->GetHostArray() : ai->GetCUDAArray());
            typedef cons<eavlIndexable<int>, RZ0> newp;
            dispatchclass_dropfirst<N, K, S, eavlIndexable<eavlArray>, Z0R, Z1F, Z1R, Z2F, Z2R, newp, RZ1, RZ2, F>
                ::go(n, structure, args0, args1, args2, newp(eavlIndexable<int>(raw,args0.first.indexer), ptrs0), ptrs1, ptrs2, functor);
        }
        if (af)
        {
            float *raw = (float*)((K::location()==eavlArray::HOST) ? af->GetHostArray() : af->GetCUDAArray());
            typedef cons<eavlIndexable<float>,RZ0> newp;
            dispatchclass_dropfirst<N, K, S, eavlIndexable<eavlArray>, Z0R, Z1F, Z1R, Z2F, Z2R, newp, RZ1, RZ2, F>
                ::go(n, structure, args0, args1, args2, newp(eavlIndexable<float>(raw,args0.first.indexer), ptrs0), ptrs1, ptrs2, functor);
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
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_dropfirst
{
    static void go(int n, S &structure,
                   cons<Z0F,Z0R> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        dispatchclassgetrawptr<N, K, S, typename Z0R::firsttype, typename Z0R::resttype,
            Z1F, Z1R, Z2F, Z2R, RZ0, RZ1, RZ2, F>
            ::go(n, structure, args0.rest, args1, args2, ptrs0, ptrs1, ptrs2, functor);
    }
};


// base case for recursion: ZR is nulltype; rotate args and continue
template <int N,
          class K,
          class S,
          class Z0F,
          class Z1F, class Z1R,
          class Z2F, class Z2R, 
          class RZ0,
          class RZ1, 
          class RZ2, 
          class F>
struct dispatchclass_dropfirst<N, K, S, Z0F, nulltype, Z1F, Z1R, Z2F, Z2R, RZ0, RZ1, RZ2, F>
{
    static void go(int n, S &structure,
                   cons<Z0F,nulltype> &args0,
                   cons<Z1F,Z1R> &args1,
                   cons<Z2F,Z2R> &args2,
                   RZ0 ptrs0,
                   RZ1 ptrs1,
                   RZ2 ptrs2,
                   F &functor)
    {
        cons<nulltype,nulltype> empty;
        dispatchclass_start<N+1, K, S, Z1F, Z1R, Z2F, Z2R, nulltype, nulltype, RZ1, RZ2, RZ0, F> 
            ::go(n, structure, args1, args2, empty, ptrs1, ptrs2, ptrs0, functor);
    }
};

// entry points for dispatch
// 3-arg
template<class K, class S, class T0, class T1, class T2, class F>
void eavlOpDispatch(int n, S &structure, T0 arrays0, T1 arrays1, T2 arrays2, F functor)
{
    dispatchclassgetrawptr<0, K, S,
        typename T0::firsttype, typename T0::resttype,
        typename T1::firsttype, typename T1::resttype,
        typename T2::firsttype, typename T2::resttype,
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, arrays1, arrays2, cnull(), cnull(), cnull(), functor);
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
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, arrays1, empty, cnull(), cnull(), cnull(), functor);
}

///\todo: this is an EXAMPLE of how we could specialize to
/// get this to work without ptrtuples as inputs.
/// we need to overload the other versions, too, if we like
/// this.
template<class K, class S, class T0, class T1, class F>
void eavlOpDispatch(int n, S &structure, T0 *arrays0, T1 *arrays1, F functor)
{
    cons<T0,nulltype> a0(arrays0);
    cons<T1,nulltype> a1(arrays1);
    cons<nulltype,nulltype> empty;
    dispatchclassgetrawptr<0, K, S,
        T0, nulltype,
        T1, nulltype,
        nulltype, nulltype,
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, a0, a1, empty, cnull(), cnull(), cnull(), functor);
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
        nulltype,
        nulltype,
        nulltype,
        F>::go(n, structure, arrays0, empty, empty, cnull(), cnull(), cnull(), functor);
}

#endif
