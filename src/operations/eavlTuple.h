// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TUPLE_H
#define EAVL_TUPLE_H

#include "eavl.h"

template <class FT, class RT> struct refcons;

// recursive (first+rest) template container
template <class FT, class RT>
struct cons
{
    typedef FT firsttype;
    typedef RT resttype;
    FT first;
    RT rest;

    EAVL_HOSTDEVICE cons() : first(), rest()
    {
    }

    template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
    EAVL_HOSTDEVICE cons(const T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4, const T5 &t5, const T6 &t6, const T7 &t7)
        : first(t0), rest(t1,t2,t3,t4,t5,t6,t7, cnull())
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE cons(const T0 &t0, const RT &rest) : first(t0), rest(rest)
    {
    }

    template <class FT2,class RT2>
    EAVL_HOSTDEVICE cons(const refcons<FT2,RT2> &rc);
};

// base case template recursion
template <class FT>
struct cons<FT, nulltype>
{
    typedef FT       firsttype;
    typedef nulltype resttype;
    FT first;

    EAVL_HOSTDEVICE cons() : first()
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE cons(const T0 &t0, nulltype=cnull()) : first(t0)
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE cons(const T0 &t0, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&)
        : first(t0)
    {
    }

    EAVL_HOSTDEVICE operator FT() const
    {
        return first;
    }

    template <class FT2>
    EAVL_HOSTDEVICE cons(const refcons<FT2,nulltype> &rc);
};

// recursive (first+rest) typing structure
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct constype
{
    typedef cons<T0, typename constype<T1,T2,T3,T4,T5,T6,T7,nulltype>::type> type;
};

// base case for type recursion
template <>
struct constype<nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype>
{
    typedef nulltype type;
};

template <class T0=nulltype, class T1=nulltype, class T2=nulltype, class T3=nulltype, class T4=nulltype, class T5=nulltype, class T6=nulltype, class T7=nulltype>
class tuple;

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
class tuple : public constype<T0,T1,T2,T3,T4,T5,T6,T7>::type
{
  public:
    typedef typename constype<T0,T1,T2,T3,T4,T5,T6,T7>::type base;
    typedef typename base::firsttype firsttype;
    typedef typename base::resttype resttype;
  public:
    EAVL_HOSTDEVICE tuple() : base()
    {
    }
    // these are const so we can initialize from a temporary
    EAVL_HOSTDEVICE tuple(const T0 &t0) : base(t0, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1) : base(t0, t1, cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1, const T2 &t2) : base(t0, t1, t2, cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3) : base(t0, t1, t2, t3, cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4) : base(t0, t1, t2, t3, t4, cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4, const T5 &t5) : base(t0, t1, t2, t3, t4, t5, cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4, const T5 &t5, const T6 &t6) : base(t0, t1, t2, t3, t4, t5, t6, cnull())
    {
    }
    EAVL_HOSTDEVICE tuple(const T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4, const T5 &t5, const T6 &t6, const T7 &t7) : base(t0, t1, t2, t3, t4, t5, t6, t7)
    {
    }
    template <class FT, class RT>
    EAVL_HOSTDEVICE tuple(const cons<FT,RT> &c) : base(c)
    {
    }
    template <class FT, class RT>
    EAVL_HOSTDEVICE tuple(const refcons<FT,RT> &rc);
};


template <class A>
inline tuple<A> make_tuple(const A &a) { return tuple<A>(a); }

template <class A, class B>
inline tuple<A,B> make_tuple(const A &a, const B &b) { return tuple<A,B>(a,b); }

template <class A, class B, class C>
inline tuple<A,B,C> make_tuple(const A &a, const B &b, const C &c) { return tuple<A,B,C>(a,b,c); }

template <class A, class B, class C, class D>
inline tuple<A,B,C,D> make_tuple(const A &a, const B &b, const C &c, const D &d) { return tuple<A,B,C,D>(a,b,c,d); }

template <class A, class B, class C, class D, class E>
inline tuple<A,B,C,D,E> make_tuple(const A &a, const B &b, const C &c, const D &d, const E &e) { return tuple<A,B,C,D,E>(a,b,c,d,e); }

template <class A, class B, class C, class D, class E, class F>
inline tuple<A,B,C,D,E,F> make_tuple(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f) { return tuple<A,B,C,D,E,F>(a,b,c,d,e,f); }

template <class A, class B, class C, class D, class E, class F, class G>
inline tuple<A,B,C,D,E,F,G> make_tuple(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g) { return tuple<A,B,C,D,E,F,G>(a,b,c,d,e,f,g); }

template <class A, class B, class C, class D, class E, class F, class G, class H>
inline tuple<A,B,C,D,E,F,G,H> make_tuple(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f, const G &g, const H &h) { return tuple<A,B,C,D,E,F,G,H>(a,b,c,d,e,f,g,h); }

#include "eavlRefTuple.h"

// create a standard cons from a refcons
// (recursion case)
template <class FT, class RT>
template <class FT2,class RT2>
cons<FT,RT>::cons(const refcons<FT2,RT2> &rc) : first(rc.first), rest(rc.rest)
{
}

// (base case)
template <class FT>
template <class FT2>
cons<FT,nulltype>::cons(const refcons<FT2,nulltype> &rc) : first(rc.first)
{
}

// create a standard tuple from a refcons, too (pass the work off to the base class, though)
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
template <class FT, class RT>
tuple<T0,T1,T2,T3,T4,T5,T6,T7>::tuple(const refcons<FT,RT> &rc) : base(rc)
{
}

// -- get<N> extracts the Nth value from a tuple

// element
template<int N, class T>
struct elementtype
{
    typedef typename T::resttype resttype;
    typedef typename elementtype<N-1, resttype>::type type;
};

// base case
template<class T>
struct elementtype<0, T>
{
    typedef typename T::firsttype type;
};

// use a class so we can specialize for the base case
template <int N>
struct getclass
{
    template<class T, class HT, class TT>
    EAVL_HOSTDEVICE static T &get(cons<HT,TT> &c)
    {
        return getclass<N-1>::template get<T>(c.rest);
    }
};

template <>
struct getclass<0>
{
    template<class T, class HT, class TT>
    EAVL_HOSTDEVICE static T &get(cons<HT,TT> &c)
    {
        return c.first;
    }
};


// now get the function just redirects to the class version
template<int N, class HT, class TT>
EAVL_HOSTDEVICE typename elementtype<N, cons<HT,TT> >::type &get(cons<HT,TT> &c)
{
    return getclass<N>::template get<typename elementtype<N, cons<HT,TT> >::type,HT,TT>(c);
}

#endif
