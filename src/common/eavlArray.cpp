// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlArray.h"

template<> const char *eavlConcreteArray<int>::GetBasicType() const { return "int"; }
template<> const char *eavlConcreteArray<byte>::GetBasicType() const { return "byte"; }
template<> const char *eavlConcreteArray<float>::GetBasicType() const { return "float"; }

eavlArray *
eavlArray::CreateObjFromName(const string &nm)
{
    if (nm == "eavlConcreteArray<float>")
	return new eavlConcreteArray<float>("");
    else if (nm == "eavlConcreteArray<byte>")
	return new eavlConcreteArray<byte>("");
    else if (nm == "eavlConcreteArray<int>")
	return new eavlConcreteArray<int>("");
    else
	throw;
}


