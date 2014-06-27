// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    typedef const refcons<const typename first::type, typename collecttype<rest>::const_type> const_type;
};

template <>
struct collecttype<nulltype>
{
    typedef nulltype type;
    typedef nulltype const_type;
};


// methods to actually create the collected data
template <class T>
struct collectclass
{
    EAVL_HOSTDEVICE static typename collecttype<T>::type get(int i, T &t)
    {
        return typename collecttype<T>::type(t.first.array[t.first.indexer.index(i)],
                                             collectclass<typename T::resttype>::get(i, t.rest));
    }
};

template <class FT>
struct collectclass< cons<FT, nulltype> >
{
    EAVL_HOSTDEVICE static typename collecttype<cons<FT,nulltype> >::type get(int i, cons<FT, nulltype> &t)
    {
        return typename collecttype< cons<FT,nulltype> >::type(t.first.array[t.first.indexer.index(i)],
                                                               cnull());
    }
};

// const version of methods to actually create the collected data
template <class T>
struct const_collectclass
{
    EAVL_HOSTDEVICE static typename collecttype<T>::const_type get(int i, const T &t)
    {
        return typename collecttype<const T>::const_type(t.first.array[t.first.indexer.index(i)],
                                                   const_collectclass<const typename T::resttype>::get(i, t.rest));
    }
};

template <class FT>
struct const_collectclass< const cons<FT, nulltype> >
{
    EAVL_HOSTDEVICE static typename collecttype<const cons<FT,nulltype> >::const_type get(int i, const cons<FT, nulltype> &t)
    {
        return typename collecttype< const cons<FT,nulltype> >::const_type(t.first.array[t.first.indexer.index(i)],
                                                                     cnull());
    }
};

// collect, using the index, one value from each array in the input, and return as references
template<class FT, class RT>
EAVL_HOSTDEVICE typename collecttype< const cons<FT, RT> >::const_type collect(int i, const cons<FT, RT> &t)
{
    return const_collectclass< const cons<FT,RT> >::get(i, t);
}

template<class FT, class RT>
EAVL_HOSTDEVICE typename collecttype< cons<FT, RT> >::type collect(int i, cons<FT, RT> &t)
{
    return collectclass< cons<FT,RT> >::get(i, t);
}

#endif
