// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.

#include "eavlFlatArray.h"

template<> const char *eavlFlatArray<int>::GetBasicType() const {return "int";}

template <class T> eavlFlatArray<T> *
eavlFlatArray<T>::CreateObjFromName(const string &nm)
{
    if (nm == "eavlFlatArray<int>")
	return new eavlFlatArray<int>();
    if (nm == "eavlFlatArray<float>")
	return new eavlFlatArray<float>();
    if (nm == "eavlFlatArray<bool>")
	return new eavlFlatArray<bool>();
    else
	throw;
}



