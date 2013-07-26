// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_REF_TUPLE_H
#define EAVL_REF_TUPLE_H

#include "eavl.h"

template <class FT, class RT> struct cons;

// recursive (first+rest) template container
template <class FT, class RT>
struct refcons
{
    typedef FT firsttype;
    typedef RT resttype;
    FT &first;
    RT rest;

    //refcons() : first(), rest()
    //{
    //}

    EAVL_HOSTDEVICE refcons(const refcons &rc) : first(rc.first), rest(rc.rest)
    {
    }
    EAVL_HOSTDEVICE refcons(refcons &rc) : first(rc.first), rest(rc.rest)
    {
    }

    
    template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
    EAVL_HOSTDEVICE refcons(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7)
        : first(t0), rest(t1,t2,t3,t4,t5,t6,t7, cnull())
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE refcons(T0 &t0, const RT &rest) : first(t0), rest(rest)
    {
    }

    template <class FT2,class RT2>
    EAVL_HOSTDEVICE void operator=(const cons<FT2,RT2> &c);

    template <class FT2,class RT2>
    EAVL_HOSTDEVICE refcons(cons<FT2,RT2> &c);
};

// base case for firstrest container template recursion
template <class FT>
struct refcons<FT, nulltype>
{
    typedef FT       firsttype;
    typedef nulltype resttype;
    FT &first;

    //refcons() : first(cnull())
    //{
    //}

    EAVL_HOSTDEVICE refcons(const refcons &rc) : first(rc.first)
    {
    }
    EAVL_HOSTDEVICE refcons(refcons &rc) : first(rc.first)
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE refcons(T0 &t0, nulltype=cnull()) : first(t0)
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE refcons(T0 &t0, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&)
        : first(t0)
    {
    }

    EAVL_HOSTDEVICE void operator=(const FT &t0)
    {
        first = t0;
    }

    template <class FT2>
    EAVL_HOSTDEVICE void operator=(const cons<FT2,nulltype> &c);

    template <class FT2>
    EAVL_HOSTDEVICE refcons(cons<FT2,nulltype> &c);

    EAVL_HOSTDEVICE operator FT()
    {
        return first;
    }

};

// recursive (first+rest) typing structure
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct refconstype
{
    typedef refcons<T0, typename refconstype<T1,T2,T3,T4,T5,T6,T7,nulltype>::type> type;
};

// base case for type recursion
template <>
struct refconstype<nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype>
{
    typedef nulltype type;
};

template <class T0=nulltype, class T1=nulltype, class T2=nulltype, class T3=nulltype, class T4=nulltype, class T5=nulltype, class T6=nulltype, class T7=nulltype>
class reftuple;

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
class reftuple : public refconstype<T0,T1,T2,T3,T4,T5,T6,T7>::type
{
  public:
    typedef typename refconstype<T0,T1,T2,T3,T4,T5,T6,T7>::type base;
    typedef typename base::firsttype firsttype;
    typedef typename base::resttype resttype;
  public:
    EAVL_HOSTDEVICE reftuple() : base()
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0) : base(t0, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1) : base(t0, t1, cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2) : base(t0, t1, t2, cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3) : base(t0, t1, t2, t3, cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4) : base(t0, t1, t2, t3, t4, cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5) : base(t0, t1, t2, t3, t4, t5, cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6) : base(t0, t1, t2, t3, t4, t5, t6, cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7) : base(t0, t1, t2, t3, t4, t5, t6, t7)
    {
    }
    template <class FT, class RT>
    EAVL_HOSTDEVICE reftuple(const refcons<FT,RT> &c) : base(c)
    {
    }
};


#include "eavlTuple.h"

// copy values over the ones in our references via assignment from a standard cons
// (recursion case)
template <class FT, class RT>
template <class FT2,class RT2>
void refcons<FT,RT>::operator=(const cons<FT2,RT2> &c)
{
    first = c.first;
    rest = c.rest;
}

// (base case)
template <class FT>
template <class FT2>
void refcons<FT,nulltype>::operator=(const cons<FT2,nulltype> &c)
{
    first = c.first;
}

// create a refcons with references to our standard cons values
// (recursion case)
template <class FT, class RT>
template <class FT2,class RT2>
refcons<FT,RT>::refcons(cons<FT2,RT2> &c) : first(c.first), rest(c.rest)
{
}

// (base case)
template <class FT>
template <class FT2>
refcons<FT,nulltype>::refcons(cons<FT2,nulltype> &c) : first(c.first)
{
}

// -- get<N> extracts the Nth value from a tuple

// element
template<int N, class T>
struct refelementtype
{
    typedef typename T::resttype resttype;
    typedef typename refelementtype<N-1, resttype>::type type;
};

// base case
template<class T>
struct refelementtype<0, T>
{
    typedef typename T::firsttype type;
};

// use a class so we can specialize for the base case
template <int N>
struct refgetclass
{
    template<class T, class HT, class TT>
    EAVL_HOSTDEVICE static T &get(refcons<HT,TT> &rc)
    {
        return refgetclass<N-1>::template get<T>(rc.rest);
    }
};

template <>
struct refgetclass<0>
{
    template<class T, class HT, class TT>
    EAVL_HOSTDEVICE static T &get(refcons<HT,TT> &rc)
    {
        return rc.first;
    }
};


// now get the function just redirects to the class version
template<int N, class HT, class TT>
EAVL_HOSTDEVICE typename refelementtype<N, refcons<HT,TT> >::type &get(refcons<HT,TT> &rc)
{
    return refgetclass<N>::template get<typename refelementtype<N, refcons<HT,TT> >::type,HT,TT>(rc);
}

// -- length() returns the number of elements in a tuple

template<class HT, class TT>
struct reflengthclass
{
    EAVL_HOSTDEVICE static int length(const refcons<HT,TT> &c)
    {
        return 1 + reflengthclass<typename TT::firsttype,typename TT::resttype>::length(c.rest);
    }
};

template <class HT>
struct reflengthclass<HT, nulltype>
{
    EAVL_HOSTDEVICE static int length(const refcons<HT,nulltype> &c)
    {
        return 1;
    }
};


// now get the function just redirects to the class version
template<class HT, class TT>
EAVL_HOSTDEVICE int length(const refcons<HT,TT> &c)
{
    return reflengthclass<HT,TT>::length(c);
}

#endif
