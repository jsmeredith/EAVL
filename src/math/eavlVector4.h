// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VECTOR4_H
#define EAVL_VECTOR4_H

#include "STL.h"
#include <math.h>
#include "eavlVector3.h"
#include "eavlUtility.h"

// ****************************************************************************
// Class:  eavlVector4
//
// Purpose:
///   A 4-component vector.
//
// Programmer:  Jeremy Meredith
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlVector4
{
  public:
    float x, y, z, w;
  public:
    eavlVector4();
    eavlVector4(const eavlVector3&);
};

EAVL_HOSTDEVICE eavlVector4::eavlVector4()
{
    x = y = z = w = 0;
}

EAVL_HOSTDEVICE eavlVector4::eavlVector4(const eavlVector3 &r)
{
    x = r.x;
    y = r.y;
    z = r.z;
    w = 0;
}

#endif
