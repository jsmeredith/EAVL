// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlArray.h"

template<> const char *eavlConcreteArray<int>::GetBasicType() { return "int"; }
template<> const char *eavlConcreteArray<byte>::GetBasicType() { return "byte"; }
template<> const char *eavlConcreteArray<float>::GetBasicType() { return "float"; }
