// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_OPERATION_H
#define EAVL_OPERATION_H

#include <climits>
#include <cfloat>
#include "eavlException.h"
#include "eavlConfig.h"
#include "eavlArray.h"
#include "eavlRegularStructure.h"
#include "eavlExplicitConnectivity.h"
#include "eavlCUDA.h"
#include "eavlIndexable.h"

#include "eavlTuple.h"
#include "eavlRefTuple.h"
#include "eavlTupleTraits.h"
#include "eavlCollect.h"


// Create a tuple of indexable arrays.  This style is called if your inputs are all eavlIndexable<> types already.
inline nulltype eavlOpArgs() { return cnull(); }

template <class A>
inline tuple<A> eavlOpArgs(const A &a) { return tuple<A>(a); }

template <class A, class B>
inline tuple<A,B> eavlOpArgs(const A &a, const B &b) { return tuple<A,B>(a,b); }

template <class A, class B, class C>
inline tuple<A,B,C> eavlOpArgs(const A &a, const B &b, const C &c) { return tuple<A,B,C>(a,b,c); }

template <class A, class B, class C, class D>
inline tuple<A,B,C,D> eavlOpArgs(const A &a, const B &b, const C &c, const D &d) { return tuple<A,B,C,D>(a,b,c,d); }

template <class A, class B, class C, class D, class E>
inline tuple<A,B,C,D,E> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e) { return tuple<A,B,C,D,E>(a,b,c,d,e); }

template <class A, class B, class C, class D, class E, class F>
inline tuple<A,B,C,D,E,F> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f) { return tuple<A,B,C,D,E,F>(a,b,c,d,e,f); }

template <class A, class B, class C, class D, class E, class F, class G>
inline tuple<A,B,C,D,E,F,G> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g) { return tuple<A,B,C,D,E,F,G>(a,b,c,d,e,f,g); }

template <class A, class B, class C, class D, class E, class F, class G, class H>
inline tuple<A,B,C,D,E,F,G,H> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h) { return tuple<A,B,C,D,E,F,G,H>(a,b,c,d,e,f,g,h); }


// Create a tuple of indexable arrays.  This style is called if your inputs are all standard arrays; it will create linear indexes for them.
template <class A>
inline tuple< eavlIndexable<A> > eavlOpArgs(A *a) { return make_tuple(eavlIndexable<A>(a)); }

template <class A, class B>
inline tuple< eavlIndexable<A> , eavlIndexable<B> > eavlOpArgs(A *a, B *b) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b)); }

template <class A, class B, class C>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> > eavlOpArgs(A *a, B *b, C *c) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c)); }

template <class A, class B, class C, class D>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> > eavlOpArgs(A *a, B *b, C *c, D *d) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d)); }

template <class A, class B, class C, class D, class E>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e)); }

template <class A, class B, class C, class D, class E, class F>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f)); }

template <class A, class B, class C, class D, class E, class F, class G>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g)); }

template <class A, class B, class C, class D, class E, class F, class G, class H>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h)); }


///\todo: switch to the new eavlIndexable and remove this
struct eavlArrayWithLinearIndex
{
    eavlArray *array;
    int div;
    int mod;
    int mul;
    int add;
    eavlArrayWithLinearIndex() : array(NULL)
    {
    }
    eavlArrayWithLinearIndex(const eavlArrayWithLinearIndex &awli)
        : array(awli.array),
          div(awli.div),
          mod(awli.mod),
          mul(awli.mul),
          add(awli.add)
    {
    }
    void operator=(const eavlArrayWithLinearIndex &awli)
    {
        array = awli.array;
        div = awli.div;
        mod = awli.mod;
        mul = awli.mul;
        add = awli.add;
    }
    eavlArrayWithLinearIndex(eavlArray *arr)
        : array(arr),
          div(1),
          mod(INT_MAX),
          mul(1),
          add(0)
    {
        if (array->GetNumberOfComponents() != 1)
            THROW(eavlException,"Must specify a component for a multi-component array");
    }
    eavlArrayWithLinearIndex(eavlArray *arr, int component)
        : array(arr),
          div(1),
          mod(INT_MAX),
          mul(arr->GetNumberOfComponents()),
          add(component)
    {
    }
    ///\todo: we're assuming a logical dimension must be a NODE array, and
    ///       we're probably not just making that assumption right here, either!
    eavlArrayWithLinearIndex(eavlArray *arr,
                           eavlRegularStructure reg, int logicaldim)
        : array(arr),
          div(reg.CalculateNodeIndexDivForDimension(logicaldim)),
          mod(reg.CalculateNodeIndexModForDimension(logicaldim)),
          mul(1),
          add(0)
    {
        if (array->GetNumberOfComponents() != 1)
            THROW(eavlException,"Must specify a component for a multi-component array");
    }
    ///\todo: we're assuming a logical dimension must be a NODE array, and
    ///       we're probably not just making that assumption right here, either!
    eavlArrayWithLinearIndex(eavlArray *arr, int component,
                           eavlRegularStructure reg, int logicaldim)
        : array(arr),
          div(reg.CalculateNodeIndexDivForDimension(logicaldim)),
          mod(reg.CalculateNodeIndexModForDimension(logicaldim)),
          mul(arr->GetNumberOfComponents()),
          add(component)
    {
    }
    void Print(ostream &out)
    {
        cout << "array="<<array<<" div="<<div<<" mod="<<mod<<" mul="<<mul<<" add="<<add<<endl;
    }
};


// ****************************************************************************
// Class:  eavlOperation
//
// Purpose:
///
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    September 2, 2011
//
// ****************************************************************************

template<class T>
struct eavlAddFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a + b; }
    T identity() { return 0; }
};


template<class T>
struct eavlSubFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a - b; }
    T identity() { return 0; }
};

template<class T>
struct eavlMulFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a * b; }
    T identity() { return 1; }
};

template<class T>
struct eavlDivFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a / b; }
    T identity() { return 1; }
};

template<class T>
struct eavlMaxFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return (a > b) ? a : b; }
    T identity();
};


template<class T>
struct eavlMinFunctor
{
  public:
    EAVL_FUNCTOR T operator()(T a, T b) { return (a < b) ? a : b; }
    T identity();
};

template<class T>
struct eavlLessThanConstFunctor
{
  private:
    T target;
  public:
    eavlLessThanConstFunctor(const T &t) : target(t) { }
    EAVL_FUNCTOR bool operator()(float x) { return x < target; }
};

template<class T>
struct eavlNotEqualFunctor
{
    EAVL_FUNCTOR bool operator()(T a, T b) { return a != b; }
};

struct DummyFunctor
{
    void operator()(void) { }
};


class eavlOperation 
{
    friend class eavlExecutor;
  public:
    virtual ~eavlOperation() { }
  protected:
    virtual void GoCPU() = 0;
    virtual void GoGPU() = 0;
};

#endif
