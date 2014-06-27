// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlOperation.h"


template<> int eavlMaxFunctor<int>::identity() { return INT_MIN; }
template<> byte eavlMaxFunctor<byte>::identity() { return 0; }
template<> float eavlMaxFunctor<float>::identity() { return -FLT_MAX; }


template<> int eavlMinFunctor<int>::identity() { return INT_MAX; }
template<> byte eavlMinFunctor<byte>::identity() { return UCHAR_MAX; }
template<> float eavlMinFunctor<float>::identity() { return FLT_MAX; }
