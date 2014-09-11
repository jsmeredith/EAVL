// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VECTOR3_H
#define EAVL_VECTOR3_H

#include "STL.h"
#include "eavlUtility.h"
#include <math.h>

// ****************************************************************************
// Class:  eavlVector3
//
// Purpose:
///   A 3-component vector.
//
// Programmer:  Jeremy Meredith
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlVector3
{
  public:
    float x, y, z;
  public:
    EAVL_HOSTDEVICE eavlVector3();
    EAVL_HOSTDEVICE eavlVector3(const float*);
    EAVL_HOSTDEVICE eavlVector3(const eavlVector3&);
    EAVL_HOSTDEVICE eavlVector3(float,float,float);

    // equality operator
    EAVL_HOSTDEVICE bool        operator==(const eavlVector3&) const;

    // assignment operator
    EAVL_HOSTDEVICE void        operator=(const eavlVector3&);

    // vector addition/subtraction
    EAVL_HOSTDEVICE eavlVector3 operator+(const eavlVector3&) const;
    EAVL_HOSTDEVICE void        operator+=(const eavlVector3&);
    EAVL_HOSTDEVICE eavlVector3 operator-(const eavlVector3&) const;
    EAVL_HOSTDEVICE void        operator-=(const eavlVector3&);

    // unary negation
    EAVL_HOSTDEVICE eavlVector3 operator-() const;

    // scalar multiplication/division
    EAVL_HOSTDEVICE eavlVector3 operator*(const float&) const;
    EAVL_HOSTDEVICE void        operator*=(const float&);
    EAVL_HOSTDEVICE eavlVector3 operator/(const float&) const;
    EAVL_HOSTDEVICE void        operator/=(const float&);

    // cross product
    EAVL_HOSTDEVICE eavlVector3 operator%(const eavlVector3&) const;

    // dot product
    EAVL_HOSTDEVICE float       operator*(const eavlVector3&) const;

    // 2-norm
    EAVL_HOSTDEVICE float       norm() const;
    // normalize
    EAVL_HOSTDEVICE void        normalize();
    EAVL_HOSTDEVICE eavlVector3 normalized() const;

    // project another vector onto this vector
    EAVL_HOSTDEVICE eavlVector3 project(const eavlVector3 &) const;

    EAVL_HOSTDEVICE const float &operator[](int i) const {return (&x)[i];}
    EAVL_HOSTDEVICE float &operator[](int i) {return (&x)[i];}

  private:
    // friends
    friend ostream& operator<<(ostream&,const eavlVector3&);
};

EAVL_HOSTDEVICE 
eavlVector3::eavlVector3()
{
    x=y=z=0;
}

EAVL_HOSTDEVICE 
eavlVector3::eavlVector3(const float *f)
{
    x=f[0];
    y=f[1];
    z=f[2];
}

EAVL_HOSTDEVICE 
eavlVector3::eavlVector3(float x_,float y_,float z_)
{
    x=x_;
    y=y_;
    z=z_;
}

EAVL_HOSTDEVICE 
eavlVector3::eavlVector3(const eavlVector3 &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

EAVL_HOSTDEVICE void
eavlVector3::operator=(const eavlVector3 &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

EAVL_HOSTDEVICE bool
eavlVector3::operator==(const eavlVector3 &r) const
{
    return (x == r.x &&
            y == r.y &&
            z == r.z);
}

// vector addition/subtraction

EAVL_HOSTDEVICE eavlVector3
eavlVector3::operator+(const eavlVector3 &r) const
{
    return eavlVector3(x+r.x, y+r.y, z+r.z);
}

EAVL_HOSTDEVICE void
eavlVector3::operator+=(const eavlVector3 &r)
{
    x += r.x;
    y += r.y;
    z += r.z;
}

EAVL_HOSTDEVICE eavlVector3
eavlVector3::operator-(const eavlVector3 &r) const
{
    return eavlVector3(x-r.x, y-r.y, z-r.z);
}

EAVL_HOSTDEVICE void
eavlVector3::operator-=(const eavlVector3 &r)
{
    x -= r.x;
    y -= r.y;
    z -= r.z;
}


// unary negation

EAVL_HOSTDEVICE eavlVector3
eavlVector3::operator-() const
{
    return eavlVector3(-x, -y, -z);
}


// scalar multiplication/division

EAVL_HOSTDEVICE eavlVector3
operator*(const float &s, eavlVector3 &p)
{
    return eavlVector3(p.x*s, p.y*s, p.z*s);
}

EAVL_HOSTDEVICE eavlVector3
eavlVector3::operator*(const float &s) const
{
    return eavlVector3(x*s, y*s, z*s);
}

EAVL_HOSTDEVICE void
eavlVector3::operator*=(const float &s)
{
    x *= s;
    y *= s;
    z *= s;
}

EAVL_HOSTDEVICE eavlVector3
eavlVector3::operator/(const float &s) const
{
    return eavlVector3(x/s, y/s, z/s);
}

EAVL_HOSTDEVICE void
eavlVector3::operator/=(const float &s)
{
    x /= s;
    y /= s;
    z /= s;
}



// cross product
EAVL_HOSTDEVICE eavlVector3
eavlVector3::operator%(const eavlVector3 &r) const
{
    eavlVector3 v;
    v.x = y*r.z - z*r.y;
    v.y = z*r.x - x*r.z;
    v.z = x*r.y - y*r.x;
    return v;
}


// dot product
EAVL_HOSTDEVICE float
eavlVector3::operator*(const eavlVector3 &r) const
{
    return x*r.x + y*r.y + z*r.z;
}

// 2-norm
EAVL_HOSTDEVICE float
eavlVector3::norm() const
{
    float n = (x*x + y*y + z*z);
    if (n>0)
        n = sqrt(n);
    return n;
}

// normalize
EAVL_HOSTDEVICE void
eavlVector3::normalize()
{
    float n = (x*x + y*y + z*z);
    if (n>0)
    {
        n = 1./sqrt(n);
        x *= n;
        y *= n;
        z *= n;
    }
}

EAVL_HOSTDEVICE eavlVector3
eavlVector3::normalized() const
{
    float n = (x*x + y*y + z*z);
    if (n==0)
        return *this;

    n = 1./sqrt(n);
    return eavlVector3(x*n, y*n, z*n);
}

EAVL_HOSTDEVICE eavlVector3
eavlVector3::project(const eavlVector3 &a) const
{
    float n = (x*x + y*y + z*z);
    if (n==0)
        return *this;
    return eavlVector3( (*this) * ((a * (*this) )  / n ) );
}

#endif
