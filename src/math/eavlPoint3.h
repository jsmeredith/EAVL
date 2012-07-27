// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_POINT3_H
#define EAVL_POINT3_H

#include "STL.h"
#include <math.h>
#include "eavlVector3.h"

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
    float v[3];
    float &x, &y, &z; ///<\todo: this may be dumb....
  public:
    virtual const char *GetType() {return "eavlPoint3";}

    eavlPoint3();
    eavlPoint3(const float*);
    eavlPoint3(const eavlPoint3&);
    eavlPoint3(float,float,float);

    // assignment operator
    void        operator=(const eavlPoint3&);

    // point subtraction results in vector
    eavlVector3 operator-(const eavlPoint3&) const;

    // addition/subtraction of a vector results in a point
    eavlPoint3  operator-(const eavlVector3&) const;
    eavlPoint3  operator+(const eavlVector3&) const;
    void        operator-=(const eavlVector3&) const;
    void        operator+=(const eavlVector3&) const;

    // dot product with vector
    float       operator*(const eavlVector3&) const;

    // unary negation
    eavlPoint3  operator-() const;

    // scalar multiplication/division
    eavlPoint3  operator*(const float&) const;
    void        operator*=(const float&);
    eavlPoint3  operator/(const float&) const;
    void        operator/=(const float&);

    const float &operator[](int i) const {return (&x)[i];}
    float &operator[](int i) {return (&x)[i];}

  private:
    // friends
    friend ostream& operator<<(ostream&,const eavlPoint3&);
};



inline 
eavlPoint3::eavlPoint3() : x(v[0]), y(v[1]), z(v[2])
{
    x=y=z=0;
}

inline 
eavlPoint3::eavlPoint3(const float *f) : x(v[0]), y(v[1]), z(v[2])
{
    x=f[0];
    y=f[1];
    z=f[2];
}

inline 
eavlPoint3::eavlPoint3(float x_,float y_,float z_) : x(v[0]), y(v[1]), z(v[2])
{
    x=x_;
    y=y_;
    z=z_;
}

inline 
eavlPoint3::eavlPoint3(const eavlPoint3 &r) : x(v[0]), y(v[1]), z(v[2])
{
    x=r.x;
    y=r.y;
    z=r.z;
}

inline void
eavlPoint3::operator=(const eavlPoint3 &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

// point subtraction results in vector
inline eavlVector3
eavlPoint3::operator-(const eavlPoint3 &r) const
{
    return eavlVector3(x-r.x, y-r.y, z-r.z);
}

// addition/subtraction of a vector results in a point
inline eavlPoint3
eavlPoint3::operator-(const eavlVector3 &r) const
{
    return eavlPoint3(x-r.x, y-r.y, z-r.z);
}

inline eavlPoint3
eavlPoint3::operator+(const eavlVector3 &r) const
{
    return eavlPoint3(x+r.x, y+r.y, z+r.z);
}

inline void
eavlPoint3::operator-=(const eavlVector3 &r) const
{
    x -= r.x;
    y -= r.y;
    z -= r.z;
}

inline void
eavlPoint3::operator+=(const eavlVector3 &r) const
{
    x += r.x;
    y += r.y;
    z += r.z;
}

// dot product
inline float
eavlPoint3::operator*(const eavlVector3 &r) const
{
    return x*r.x + y*r.y + z*r.z;
}

inline float
operator*(const eavlVector3 &l, const eavlPoint3 &r)
{
    return l.x*r.x + l.y*r.y + l.z*r.z;
}


// unary negation

inline eavlPoint3
eavlPoint3::operator-() const
{
    return eavlPoint3(-x, -y, -z);
}


// scalar multiplication/division

inline eavlPoint3
eavlPoint3::operator*(const float &s) const
{
    return eavlPoint3(x*s, y*s, z*s);
}

inline void
eavlPoint3::operator*=(const float &s)
{
    x *= s;
    y *= s;
    z *= s;
}

inline eavlPoint3
eavlPoint3::operator/(const float &s) const
{
    return eavlPoint3(x/s, y/s, z/s);
}

inline void
eavlPoint3::operator/=(const float &s)
{
    x /= s;
    y /= s;
    z /= s;
}

#endif
