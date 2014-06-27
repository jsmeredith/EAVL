// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
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

template <class A, class B, class C, class D, class E, class F, class G, class H, class I>
inline tuple<A,B,C,D,E,F,G,H,I> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i) { return tuple<A,B,C,D,E,F,G,H,I>(a,b,c,d,e,f,g,h,i); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J>
inline tuple<A,B,C,D,E,F,G,H,I,J> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j) { return tuple<A,B,C,D,E,F,G,H,I,J>(a,b,c,d,e,f,g,h,i,j); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K>
inline tuple<A,B,C,D,E,F,G,H,I,J,K> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j, const K &k) { return tuple<A,B,C,D,E,F,G,H,I,J,K>(a,b,c,d,e,f,g,h,i,j,k); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L>
inline tuple<A,B,C,D,E,F,G,H,I,J,K,L> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j, const K &k, const L &l) { return tuple<A,B,C,D,E,F,G,H,I,J,K,L>(a,b,c,d,e,f,g,h,i,j,k,l); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M>
inline tuple<A,B,C,D,E,F,G,H,I,J,K,L,M> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j, const K &k, const L &l, const M &m) { return tuple<A,B,C,D,E,F,G,H,I,J,K,L,M>(a,b,c,d,e,f,g,h,i,j,k,l,m); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M, class N>
inline tuple<A,B,C,D,E,F,G,H,I,J,K,L,M,N> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j, const K &k, const L &l, const M &m, const N &n) { return tuple<A,B,C,D,E,F,G,H,I,J,K,L,M,N>(a,b,c,d,e,f,g,h,i,j,k,l,m,n); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M, class N, class O>
inline tuple<A,B,C,D,E,F,G,H,I,J,K,L,M,N,O> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j, const K &k, const L &l, const M &m, const N &n, const O &o) { return tuple<A,B,C,D,E,F,G,H,I,J,K,L,M,N,O>(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M, class N, class O, class P>
inline tuple<A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P> eavlOpArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h, const I &i, const J &j, const K &k, const L &l, const M &m, const N &n, const O &o, const P &p) { return tuple<A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P>(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p); }


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

template <class A, class B, class C, class D, class E, class F, class G, class H, class I>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J>, eavlIndexable<K> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j, K *k) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j), eavlIndexable<K>(k)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J>, eavlIndexable<K>, eavlIndexable<L> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j, K *k, L *l) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j), eavlIndexable<K>(k), eavlIndexable<L>(l)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J>, eavlIndexable<K>, eavlIndexable<L>, eavlIndexable<M> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j, K *k, L *l, M *m) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j), eavlIndexable<K>(k), eavlIndexable<L>(l), eavlIndexable<M>(m)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M, class N>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J>, eavlIndexable<K>, eavlIndexable<L>, eavlIndexable<M>, eavlIndexable<N> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j, K *k, L *l, M *m, N *n) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j), eavlIndexable<K>(k), eavlIndexable<L>(l), eavlIndexable<M>(m), eavlIndexable<N>(n)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M, class N, class O>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J>, eavlIndexable<K>, eavlIndexable<L>, eavlIndexable<M>, eavlIndexable<N>, eavlIndexable<O> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j, K *k, L *l, M *m, N *n, O *o) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j), eavlIndexable<K>(k), eavlIndexable<L>(l), eavlIndexable<M>(m), eavlIndexable<N>(n), eavlIndexable<O>(o)); }

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K, class L, class M, class N, class O, class P>
inline tuple< eavlIndexable<A> , eavlIndexable<B> , eavlIndexable<C> , eavlIndexable<D> , eavlIndexable<E> , eavlIndexable<F> , eavlIndexable<G> , eavlIndexable<H>, eavlIndexable<I>, eavlIndexable<J>, eavlIndexable<K>, eavlIndexable<L>, eavlIndexable<M>, eavlIndexable<N>, eavlIndexable<O>, eavlIndexable<P> > eavlOpArgs(A *a, B *b, C *c, D *d, E *e, F *f, G *g, H *h, I *i, J *j, K *k, L *l, M *m, N *n, O *o, P *p) { return make_tuple(eavlIndexable<A>(a), eavlIndexable<B>(b), eavlIndexable<C>(c), eavlIndexable<D>(d), eavlIndexable<E>(e), eavlIndexable<F>(f), eavlIndexable<G>(g), eavlIndexable<H>(h), eavlIndexable<I>(i), eavlIndexable<J>(j), eavlIndexable<K>(k), eavlIndexable<L>(l), eavlIndexable<M>(m), eavlIndexable<N>(n), eavlIndexable<O>(o), eavlIndexable<P>(p)); }
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

// Note:
//    These functors are more complex than they need to be for most usage.
//    However, as these are provided for arbitrary use, they need to be
//    flexible.  For instance, they need to support use with either a
//    pair of arguments or a tuple.  And in the case of a tuple, we want
//    to support a tuple of references to the original arrays (refcons)
//    or a standard tuple of some known type.  The associative/commutative
//    ones -- like addition or min and max -- are worse, because you might
//    potentially want to collapse any number of items with these functors,
//    and so those must use template-recursion to walk through all items.
//    When you are writing your own functor, you know how it will be used,
//    and so you can often get away with a single operator() taking 
//    either a tuple or primitive arguments.  So while the functors below
//    can provide an example of how to write one which will perform well
//    and accurately in any situation, you can generally write a much
//    simpler functor without having to follow these patterns.

template<class T>
struct eavlAddFunctor
{
    // this works on two inputs (each of arbitrary length) by calling the single-arg versions below
    template<class A, class B>
    EAVL_FUNCTOR T operator()(const A &args0, const B &args1) { return operator()(args0) + operator()(args1); }

    // this works on one primitive type
    template<class A>
    EAVL_FUNCTOR T operator()(const A &arg) { return arg; }

    // this works on one input (of arbitrary length) via template recursion; cons/tuple version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const cons<HT,TT> &args) { return T(args.first) + operator()(args.rest); }
    template<class HT>
    EAVL_FUNCTOR T operator()(const cons<HT,nulltype> &args) { return T(args.first); }

    // this works on one input (of arbitrary length) via template recursion; refcons version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const refcons<HT,TT> &args) { return T(args.first) + operator()(args.rest); }
    template<class HT>
    EAVL_FUNCTOR T operator()(const refcons<HT,nulltype> &args) { return T(args.first); }

    T identity() { return 0; }
};

template<class T>
struct eavlSubFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a - b; }
    template <class T1, class T2>
    EAVL_FUNCTOR T operator()(const cons<T1, const cons<T2, nulltype> > &args) { return T(args.first) - T(args.rest.first); }
    template <class T1, class T2>
    EAVL_FUNCTOR T operator()(const refcons<T1, const refcons<T2, nulltype> > &args) { return T(args.first) - T(args.rest.first); }
    T identity() { return 0; }
};

template<class T>
struct eavlMulFunctor
{
    // this works on two inputs (each of arbitrary length) by calling the single-arg versions below
    template<class A, class B>
    EAVL_FUNCTOR T operator()(const A &args0, const B &args1) { return operator()(args0) * operator()(args1); }

    // this works on one primitive type
    template<class A>
    EAVL_FUNCTOR T operator()(const A &arg) { return arg; }

    // this works on one input (of arbitrary length) via template recursion; cons/tuple version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const cons<HT,TT> &args) { return T(args.first) * operator()(args.rest); }
    template<class HT>
    EAVL_FUNCTOR T operator()(const cons<HT,nulltype> &args) { return T(args.first); }

    // this works on one input (of arbitrary length) via template recursion; refcons version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const refcons<HT,TT> &args) { return T(args.first) * operator()(args.rest); }
    template<class HT>
    EAVL_FUNCTOR T operator()(const refcons<HT,nulltype> &args) { return T(args.first); }

    T identity() { return 0; }
};

template<class T>
struct eavlDivFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a / b; }
    template <class T1, class T2>
    EAVL_FUNCTOR T operator()(const cons<T1, const cons<T2, nulltype> > &args) { return T(args.first) / T(args.rest.first); }
    template <class T1, class T2>
    EAVL_FUNCTOR T operator()(const refcons<T1, const refcons<T2, nulltype> > &args) { return T(args.first) / T(args.rest.first); }
    T identity() { return 0; }
};

template<class T>
struct eavlMaxFunctor
{
    // this works on two inputs (each of arbitrary length) by calling the single-arg versions below
    template<class A, class B>
    EAVL_FUNCTOR T operator()(const A &args0, const B &args1) { T a0 = operator()(args0); T a1 = operator()(args1); return a0>a1 ? a0:a1; }

    // this works on one primitive type
    template<class A>
    EAVL_FUNCTOR T operator()(const A &arg) { return arg; }

    // this works on one input (of arbitrary length) via template recursion; cons/tuple version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const cons<HT,TT> &args) { T a0 = args.first;  T a1 = operator()(args.rest); return a0>a1 ? a0:a1; }
    template<class HT>
    EAVL_FUNCTOR T operator()(const cons<HT,nulltype> &args) { return T(args.first); }

    // this works on one input (of arbitrary length) via template recursion; refcons version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const refcons<HT,TT> &args) { T a0 = args.first;  T a1 = operator()(args.rest); return a0>a1 ? a0:a1; }
    template<class HT>
    EAVL_FUNCTOR T operator()(const refcons<HT,nulltype> &args) { return T(args.first); }

    T identity();
};

template<class T>
struct eavlMinFunctor
{
    // this works on two inputs (each of arbitrary length) by calling the single-arg versions below
    template<class A, class B>
    EAVL_FUNCTOR T operator()(const A &args0, const B &args1) { T a0 = operator()(args0); T a1 = operator()(args1); return a0<a1 ? a0:a1; }

    // this works on one primitive type
    template<class A>
    EAVL_FUNCTOR T operator()(const A &arg) { return arg; }

    // this works on one input (of arbitrary length) via template recursion; cons/tuple version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const cons<HT,TT> &args) { T a0 = args.first;  T a1 = operator()(args.rest); return a0<a1 ? a0:a1; }
    template<class HT>
    EAVL_FUNCTOR T operator()(const cons<HT,nulltype> &args) { return T(args.first); }

    // this works on one input (of arbitrary length) via template recursion; refcons version
    template<class HT, class TT>
    EAVL_FUNCTOR T operator()(const refcons<HT,TT> &args) { T a0 = args.first;  T a1 = operator()(args.rest); return a0<a1 ? a0:a1; }
    template<class HT>
    EAVL_FUNCTOR T operator()(const refcons<HT,nulltype> &args) { return T(args.first); }

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

