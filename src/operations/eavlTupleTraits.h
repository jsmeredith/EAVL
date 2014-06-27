// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TUPLE_TRAITS_H
#define EAVL_TUPLE_TRAITS_H

#include "eavlTuple.h"
#include "eavlRefTuple.h"

template<class T>
struct traits
{
    typedef typename T::firsttype first;
    typedef typename T::resttype  rest;

    typedef typename traits<rest>::lasttype                              lasttype;
    typedef cons<first, typename traits<rest>::allbutlasttype>           allbutlasttype;
    typedef cons<lasttype, typename traits<allbutlasttype>::reversetype> reversetype;

    enum values { length = 1 + traits<rest>::length };
};

template<class T>
struct traits< cons<T,nulltype> >
{
    typedef T                lasttype;
    typedef nulltype         allbutlasttype;
    typedef cons<T,nulltype> reversetype;

    enum values { length = 1 };
};


///\todo: we're not handling ref types well here; we just make cons out of them
template<class T>
struct traits< refcons<T,nulltype> >
{
    typedef T                lasttype;
    typedef nulltype         allbutlasttype;
    typedef cons<T,nulltype> reversetype;

    enum values { length = 1 };
};

#endif
