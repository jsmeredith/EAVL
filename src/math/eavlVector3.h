// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VECTOR3_H
#define EAVL_VECTOR3_H

#include "STL.h"
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
    float v[3];
    float &x, &y, &z; ///<\todo: this may be dumb....
  public:
    eavlVector3();
    eavlVector3(const float*);
    eavlVector3(const eavlVector3&);
    eavlVector3(float,float,float);

    // assignment operator
    void        operator=(const eavlVector3&);

    // vector addition/subtraction
    eavlVector3 operator+(const eavlVector3&) const;
    void        operator+=(const eavlVector3&);
    eavlVector3 operator-(const eavlVector3&) const;
    void        operator-=(const eavlVector3&);

    // unary negation
    eavlVector3 operator-() const;

    // scalar multiplication/division
    eavlVector3 operator*(const float&) const;
    void        operator*=(const float&);
    eavlVector3 operator/(const float&) const;
    void        operator/=(const float&);

    // cross product
    eavlVector3 operator%(const eavlVector3&) const;

    // dot product
    float       operator*(const eavlVector3&) const;

    // 2-norm
    float       norm() const;
    // normalize
    void        normalize();
    eavlVector3 normalized() const;

    const float &operator[](int i) const {return (&x)[i];}
    float &operator[](int i) {return (&x)[i];}

  private:
    // friends
    friend ostream& operator<<(ostream&,const eavlVector3&);
};

inline 
eavlVector3::eavlVector3() : x(v[0]), y(v[1]), z(v[2])
{
    x=y=z=0;
}

inline 
eavlVector3::eavlVector3(const float *f) : x(v[0]), y(v[1]), z(v[2])
{
    x=f[0];
    y=f[1];
    z=f[2];
}

inline 
eavlVector3::eavlVector3(float x_,float y_,float z_) : x(v[0]), y(v[1]), z(v[2])
{
    x=x_;
    y=y_;
    z=z_;
}

inline 
eavlVector3::eavlVector3(const eavlVector3 &r) : x(v[0]), y(v[1]), z(v[2])
{
    x=r.x;
    y=r.y;
    z=r.z;
}

inline void
eavlVector3::operator=(const eavlVector3 &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

// vector addition/subtraction

inline eavlVector3
eavlVector3::operator+(const eavlVector3 &r) const
{
    return eavlVector3(x+r.x, y+r.y, z+r.z);
}

inline void
eavlVector3::operator+=(const eavlVector3 &r)
{
    x += r.x;
    y += r.y;
    z += r.z;
}

inline eavlVector3
eavlVector3::operator-(const eavlVector3 &r) const
{
    return eavlVector3(x-r.x, y-r.y, z-r.z);
}

inline void
eavlVector3::operator-=(const eavlVector3 &r)
{
    x -= r.x;
    y -= r.y;
    z -= r.z;
}


// unary negation

inline eavlVector3
eavlVector3::operator-() const
{
    return eavlVector3(-x, -y, -z);
}


// scalar multiplication/division

inline eavlVector3
eavlVector3::operator*(const float &s) const
{
    return eavlVector3(x*s, y*s, z*s);
}

inline void
eavlVector3::operator*=(const float &s)
{
    x *= s;
    y *= s;
    z *= s;
}

inline eavlVector3
eavlVector3::operator/(const float &s) const
{
    return eavlVector3(x/s, y/s, z/s);
}

inline void
eavlVector3::operator/=(const float &s)
{
    x /= s;
    y /= s;
    z /= s;
}



// cross product
inline eavlVector3
eavlVector3::operator%(const eavlVector3 &r) const
{
    eavlVector3 v;
    v.x = y*r.z - z*r.y;
    v.y = z*r.x - x*r.z;
    v.z = x*r.y - y*r.x;
    return v;
}


// dot product
inline float
eavlVector3::operator*(const eavlVector3 &r) const
{
    return x*r.x + y*r.y + z*r.z;
}

// 2-norm
inline float
eavlVector3::norm() const
{
    float n = (x*x + y*y + z*z);
    if (n>0)
        n = sqrt(n);
    return n;
}

// normalize
inline void
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

inline eavlVector3
eavlVector3::normalized() const
{
    float n = (x*x + y*y + z*z);
    if (n==0)
        return *this;

    n = 1./sqrt(n);
    return eavlVector3(x*n, y*n, z*n);
}


#endif
