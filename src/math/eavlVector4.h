// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    float v[4];
  public:
    eavlVector4();
    eavlVector4(const eavlVector3&);

    float *vals() { return v; }

    const float &x() const { return v[0]; }
    const float &y() const { return v[1]; }
    const float &z() const { return v[2]; }
    const float &w() const { return v[3]; }

};

inline eavlVector4::eavlVector4()
{
    v[0] = v[1] = v[2] = v[3] = 0;
}

inline eavlVector4::eavlVector4(const eavlVector3 &r)
{
    v[0] = r.x;
    v[1] = r.y;
    v[2] = r.z;
    v[3] = 0;
}

#endif
