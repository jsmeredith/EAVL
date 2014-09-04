#ifndef EAVL_VECTOR3_I_H
#define EAVL_VECTOR3_I_H

#include "STL.h"
#include "eavlUtility.h"
#include <math.h>
class eavlVector3i
{
  public:
    int x, y, z;
  public:
    EAVL_HOSTDEVICE eavlVector3i();
    EAVL_HOSTDEVICE eavlVector3i(const int*);
    EAVL_HOSTDEVICE eavlVector3i(const eavlVector3i&);
    EAVL_HOSTDEVICE eavlVector3i(int,int,int);

    // assignment operator
    EAVL_HOSTDEVICE void        operator=(const eavlVector3i&);

    // vector addition/subtraction
    EAVL_HOSTDEVICE eavlVector3i operator+(const eavlVector3i&) const;
    EAVL_HOSTDEVICE void        operator+=(const eavlVector3i&);
    EAVL_HOSTDEVICE eavlVector3i operator-(const eavlVector3i&) const;
    EAVL_HOSTDEVICE void        operator-=(const eavlVector3i&);

    // unary negation
    EAVL_HOSTDEVICE eavlVector3i operator-() const;

    // scalar multiplication/division
    EAVL_HOSTDEVICE eavlVector3i operator*(const int&) const;
    EAVL_HOSTDEVICE void        operator*=(const int&);
    EAVL_HOSTDEVICE eavlVector3i operator/(const int&) const;
    EAVL_HOSTDEVICE void        operator/=(const int&);

    // cross product
    EAVL_HOSTDEVICE eavlVector3i operator%(const eavlVector3i&) const;

    // dot product
    EAVL_HOSTDEVICE int       operator*(const eavlVector3i&) const;

    EAVL_HOSTDEVICE eavlVector3i  min(const eavlVector3i &r) const;
    EAVL_HOSTDEVICE eavlVector3i  max(const eavlVector3i &r) const;

    // 2-norm
    EAVL_HOSTDEVICE int       norm() const;

    // normalize
    EAVL_HOSTDEVICE void        normalize();
    EAVL_HOSTDEVICE eavlVector3i normalized() const;

    EAVL_HOSTDEVICE const int &operator[](int i) const {return (&x)[i];}
    EAVL_HOSTDEVICE int &operator[](int i) {return (&x)[i];}

  private:
    // friends
    friend ostream& operator<<(ostream&,const eavlVector3i&);
};

EAVL_HOSTDEVICE 
eavlVector3i::eavlVector3i()
{
    x=y=z=0;
}

EAVL_HOSTDEVICE 
eavlVector3i::eavlVector3i(const int *f)
{
    x=f[0];
    y=f[1];
    z=f[2];
}

EAVL_HOSTDEVICE 
eavlVector3i::eavlVector3i(int x_,int y_,int z_)
{
    x=x_;
    y=y_;
    z=z_;
}

EAVL_HOSTDEVICE 
eavlVector3i::eavlVector3i(const eavlVector3i &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

EAVL_HOSTDEVICE void
eavlVector3i::operator=(const eavlVector3i &r)
{
    x=r.x;
    y=r.y;
    z=r.z;
}

// vector addition/subtraction

EAVL_HOSTDEVICE eavlVector3i
eavlVector3i::operator+(const eavlVector3i &r) const
{
    return eavlVector3i(x+r.x, y+r.y, z+r.z);
}

EAVL_HOSTDEVICE void
eavlVector3i::operator+=(const eavlVector3i &r)
{
    x += r.x;
    y += r.y;
    z += r.z;
}

EAVL_HOSTDEVICE eavlVector3i
eavlVector3i::operator-(const eavlVector3i &r) const
{
    return eavlVector3i(x-r.x, y-r.y, z-r.z);
}

EAVL_HOSTDEVICE void
eavlVector3i::operator-=(const eavlVector3i &r)
{
    x -= r.x;
    y -= r.y;
    z -= r.z;
}


// unary negation

EAVL_HOSTDEVICE eavlVector3i
eavlVector3i::operator-() const
{
    return eavlVector3i(-x, -y, -z);
}


// scalar multiplication/division

EAVL_HOSTDEVICE eavlVector3i
eavlVector3i::operator*(const int &s) const
{
    return eavlVector3i(x*s, y*s, z*s);
}

EAVL_HOSTDEVICE void
eavlVector3i::operator*=(const int &s)
{
    x *= s;
    y *= s;
    z *= s;
}

EAVL_HOSTDEVICE eavlVector3i
eavlVector3i::operator/(const int &s) const
{
    return eavlVector3i(x/s, y/s, z/s);
}

EAVL_HOSTDEVICE void
eavlVector3i::operator/=(const int &s)
{
    x /= s;
    y /= s;
    z /= s;
}



// cross product
EAVL_HOSTDEVICE eavlVector3i
eavlVector3i::operator%(const eavlVector3i &r) const
{
    eavlVector3i v;
    v.x = y*r.z - z*r.y;
    v.y = z*r.x - x*r.z;
    v.z = x*r.y - y*r.x;
    return v;
}


// dot product
EAVL_HOSTDEVICE int
eavlVector3i::operator*(const eavlVector3i &r) const
{
    return x*r.x + y*r.y + z*r.z;
}

EAVL_HOSTDEVICE eavlVector3i  eavlVector3i::min(const eavlVector3i &r) const
{
    eavlVector3i result;
    result.x = (x > r.x) ? r.x : x ;
    result.y = (y > r.y) ? r.y : y ;
    result.z = (z > r.z) ? r.z : z ;
    
    return result;
}

EAVL_HOSTDEVICE eavlVector3i  eavlVector3i::max(const eavlVector3i &r) const
{
    eavlVector3i result;
    result.x = (x < r.x) ? r.x : x ;
    result.y = (y < r.y) ? r.y : y ;
    result.z = (z < r.z) ? r.z : z ;
    
    return result;
}

#endif