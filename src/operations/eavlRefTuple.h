// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    
    template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15>
    EAVL_HOSTDEVICE refcons(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10, T11 &t11, T12 &t12, T13 &t13, T14 &t14, T15 &t15)
        : first(t0), rest(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15, cnull())
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE refcons(T0 &t0, const RT &rest) : first(t0), rest(rest)
    {
    }

    EAVL_HOSTDEVICE void CopyFrom(const refcons &rc)
    {
        first = rc.first;
        rest.CopyFrom(rc.rest);
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

    template <class T0>
    EAVL_HOSTDEVICE refcons(T0 &t0, nulltype=cnull()) : first(t0)
    {
    }

    template <class T0>
    EAVL_HOSTDEVICE refcons(T0 &t0, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&,const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&, const nulltype&)
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

    EAVL_HOSTDEVICE operator FT() const
    {
        return first;
    }

    EAVL_HOSTDEVICE void CopyFrom(const FT &v)
    {
        first = v;
    }

  private:
    EAVL_HOSTDEVICE void operator=(const refcons &rc); // = delete
};

// recursive (first+rest) typing structure
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15>
struct refconstype
{
    typedef refcons<T0, typename refconstype<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,nulltype>::type> type;
};

// base case for type recursion
template <>
struct refconstype<nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype,nulltype, nulltype, nulltype, nulltype, nulltype, nulltype, nulltype>
{
    typedef nulltype type;
};

template <class T0=nulltype, class T1=nulltype, class T2=nulltype, class T3=nulltype, class T4=nulltype, class T5=nulltype, class T6=nulltype, class T7=nulltype, class T8=nulltype, class T9=nulltype, class T10=nulltype, class T11=nulltype, class T12=nulltype, class T13=nulltype, class T14=nulltype, class T15=nulltype>
class reftuple;

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15>
class reftuple : public refconstype<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>::type
{
  public:
    typedef typename refconstype<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>::type base;
    typedef typename base::firsttype firsttype;
    typedef typename base::resttype resttype;
  public:
    EAVL_HOSTDEVICE reftuple() : base()
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0) : base(t0, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1) : base(t0, t1, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(),cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2) : base(t0, t1, t2, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3) : base(t0, t1, t2, t3, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4) : base(t0, t1, t2, t3, t4, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5) : base(t0, t1, t2, t3, t4, t5, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6) : base(t0, t1, t2, t3, t4, t5, t6, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7) : base(t0, t1, t2, t3, t4, t5, t6, t7, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, cnull(), cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, cnull(), cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, cnull(), cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10, T11 &t11) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, cnull(), cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10, T11 &t11, T12 &t12) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, cnull(), cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10, T11 &t11, T12 &t12, T13 &t13) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, cnull(), cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10, T11 &t11, T12 &t12, T13 &t13, T14 &t14) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, cnull())
    {
    }
    EAVL_HOSTDEVICE reftuple(T0 &t0,T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9, T10 &t10, T11 &t11, T12 &t12, T13 &t13, T14 &t14, T15 &t15) : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15)
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
    template<class T, class HT, class TT>
    EAVL_HOSTDEVICE static const T &get(const refcons<HT,TT> &rc)
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
    template<class T, class HT, class TT>
    EAVL_HOSTDEVICE static const T &get(const refcons<HT,TT> &rc)
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

template<int N, class HT, class TT>
EAVL_HOSTDEVICE const typename refelementtype<N, refcons<HT,TT> >::type &get(const refcons<HT,TT> &rc)
{
    return refgetclass<N>::template get<typename refelementtype<N, const refcons<HT,TT> >::type,HT,TT>(rc);
}

#endif

