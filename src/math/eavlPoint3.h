// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_POINT3_H
#define EAVL_POINT3_H

#include "STL.h"
#include <math.h>
#include "eavlVector3.h"
#include "eavlUtility.h"

// ****************************************************************************
// Class:  eavlPoint3
//
// Purpose:
///   A 3-component point.
//
// Programmer:  Jeremy Meredith
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlPoint3
{
  public:
    float x, y, z;
  public:
    EAVL_HOSTDEVICE eavlPoint3();
    EAVL_HOSTDEVICE eavlPoint3(const float*);
    EAVL_HOSTDEVICE eavlPoint3(const eavlPoint3&);
    EAVL_HOSTDEVICE eavlPoint3(float,float,float);

    // equality operator
    EAVL_HOSTDEVICE bool        operator==(const eavlPoint3&) const;

    // assignment operator
    EAVL_HOSTDEVICE void        operator=(const eavlPoint3&);

    // point subtraction results in vector
    EAVL_HOSTDEVICE eavlVector3 operator-(const eavlPoint3&) const;

    // addition/subtraction of a vector results in a point
    EAVL_HOSTDEVICE eavlPoint3  operator-(const eavlVector3&) const;
    EAVL_HOSTDEVICE eavlPoint3  operator+(const eavlVector3&) const;
    EAVL_HOSTDEVICE void        operator-=(const eavlVector3&);
    EAVL_HOSTDEVICE void        operator+=(const eavlVector3&);

    // dot product with vector
    EAVL_HOSTDEVICE float       operator*(const eavlVector3&) const;

    // unary negation
    EAVL_HOSTDEVICE eavlPoint3  operator-() const;

    // scalar multiplication/division
    EAVL_HOSTDEVICE eavlPoint3  operator*(const float&) const;
    EAVL_HOSTDEVICE void        operator*=(const float&);
    EAVL_HOSTDEVICE eavlPoint3  operator/(const float&) const;
    EAVL_HOSTDEVICE void        operator/=(const float&);

    EAVL_HOSTDEVICE const float &operator[](int i) const {return (&x)[i];}
    EAVL_HOSTDEVICE float &operator[](int i) {return (&x)[i];}

  private:
    // friends
    friend ostream& operator<<(ostream&,const eavlPoint3&);
};



EAVL_HOSTDEVICE 
eavlPoint3::eavlPoint3()
{
    x=y=z=0;
}

EAVL_HOSTDEVICE 
eavlPoint3::eavlPoint3(const float *f)
{
    x=f[0];
    y=f[1];
    z=f[2];
}

EAVL_HOSTDEVICE 
eavlPoint3::eavlPoint3(float x_,float y_,float z_)
{
    x=x_;
    y=y_;
    z=z_;
}

EAVL_HOSTDEVICE 
eavlPoint3::eavlPoint3(const eavlPoint3 &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

EAVL_HOSTDEVICE void
eavlPoint3::operator=(const eavlPoint3 &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

EAVL_HOSTDEVICE bool
eavlPoint3::operator==(const eavlPoint3 &r) const
{
    return (x == r.x &&
            y == r.y &&
            z == r.z);
}

// point subtraction results in vector
EAVL_HOSTDEVICE eavlVector3
eavlPoint3::operator-(const eavlPoint3 &r) const
{
    return eavlVector3(x-r.x, y-r.y, z-r.z);
}

// addition/subtraction of a vector results in a point
EAVL_HOSTDEVICE eavlPoint3
eavlPoint3::operator-(const eavlVector3 &r) const
{
    return eavlPoint3(x-r.x, y-r.y, z-r.z);
}

EAVL_HOSTDEVICE eavlPoint3
eavlPoint3::operator+(const eavlVector3 &r) const
{
    return eavlPoint3(x+r.x, y+r.y, z+r.z);
}

EAVL_HOSTDEVICE void
eavlPoint3::operator-=(const eavlVector3 &r)
{
    x -= r.x;
    y -= r.y;
    z -= r.z;
}

EAVL_HOSTDEVICE void
eavlPoint3::operator+=(const eavlVector3 &r)
{
    x += r.x;
    y += r.y;
    z += r.z;
}

// dot product
EAVL_HOSTDEVICE float
eavlPoint3::operator*(const eavlVector3 &r) const
{
    return x*r.x + y*r.y + z*r.z;
}

EAVL_HOSTDEVICE float
operator*(const eavlVector3 &l, const eavlPoint3 &r)
{
    return l.x*r.x + l.y*r.y + l.z*r.z;
}


// unary negation

EAVL_HOSTDEVICE eavlPoint3
eavlPoint3::operator-() const
{
    return eavlPoint3(-x, -y, -z);
}


// scalar multiplication/division

EAVL_HOSTDEVICE eavlPoint3
operator*(const float &s, eavlPoint3 &p)
{
    return eavlPoint3(p.x*s, p.y*s, p.z*s);
}

EAVL_HOSTDEVICE eavlPoint3
eavlPoint3::operator*(const float &s) const
{
    return eavlPoint3(x*s, y*s, z*s);
}

EAVL_HOSTDEVICE void
eavlPoint3::operator*=(const float &s)
{
    x *= s;
    y *= s;
    z *= s;
}

EAVL_HOSTDEVICE eavlPoint3
eavlPoint3::operator/(const float &s) const
{
    return eavlPoint3(x/s, y/s, z/s);
}

EAVL_HOSTDEVICE void
eavlPoint3::operator/=(const float &s)
{
    x /= s;
    y /= s;
    z /= s;
}

#endif
