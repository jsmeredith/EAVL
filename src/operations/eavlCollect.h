// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLLECT_H
#define EAVL_COLLECT_H

#include "eavlRefTuple.h"
#include "eavlTuple.h"

template <class T>
struct collecttype
{
    typedef typename T::firsttype first;
    typedef typename T::resttype rest;
    typedef refcons<typename first::type, typename collecttype<rest>::type> type;
};

template <>
struct collecttype<nulltype>
{
    typedef nulltype type;
};


template <class T>
struct collectclass
{
    EAVL_HOSTDEVICE static typename collecttype<T>::type get(int i, const T &t)
    {
        //typedef typename collecttype<T::resttype> TT;
        typename collecttype<T>::type
            cc(t.first.array[t.first.indexer.index(i)],
               collectclass<typename T::resttype>::get(i, t.rest));
        return cc;
    }
};

template <class FT>
struct collectclass< cons<FT, nulltype> >
{
    EAVL_HOSTDEVICE static typename collecttype<cons<FT,nulltype> >::type get(int i, const cons<FT, nulltype> &t)
    {
        typename collecttype< cons<FT,nulltype> >::type cc(t.first.array[t.first.indexer.index(i)],
                                                                   cnull());
        //cc.r = cnull();
        return cc;
    }
};

// T is the cons of raw pointers
// i is the index
template<class FT, class RT>
EAVL_HOSTDEVICE typename collecttype< cons<FT, RT> >::type collect(int i, const cons<FT, RT> &t)
{
    return collectclass< cons<FT,RT> >::get(i, t);
}

#endif
