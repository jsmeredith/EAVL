// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef MATRIX4X4_H
#define MATRIX4X4_H

class eavlVector3;
class eavlPoint3;
class eavlVector4;
#include "STL.h"

// ****************************************************************************
// Class:  eavlMatrix4x4
//
// Purpose:
///   A 4x4 matrix.
//
// Programmer:  Jeremy Meredith
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlMatrix4x4
{
  public:
    float m[4][4];
    float openglm[16];
  public:
    eavlMatrix4x4();
    eavlMatrix4x4(const eavlMatrix4x4&);

    // assignment operator
    void          operator=(const eavlMatrix4x4&);

    // multiply matrix*matrix
    eavlMatrix4x4 operator*(const eavlMatrix4x4&) const;

    // transform point
    eavlPoint3    operator*(const eavlPoint3&) const;
    // transform vector
    eavlVector3   operator*(const eavlVector3&) const;
    // transform point/vector
    eavlVector3   operator*(const eavlVector4&) const;

    void   Invert();
    void   Transpose();

    const float &operator()(int r, int c) const {return m[r][c];}
    float &operator()(int r, int c) {return m[r][c];}

    // utility
    void   CreateIdentity();
    void   CreateZero();
    void   CreateTrackball(float,float, float,float);
    void   CreateTranslate(float, float, float);
    void   CreateTranslate(const eavlVector3&);
    void   CreateTranslate(const eavlPoint3&);
    void   CreateRBT(const eavlPoint3&, const eavlPoint3&, const eavlVector3&);
    void   CreateScale(float,float,float);
    void   CreateScale(float);
    void   CreatePerspectiveProjection(float,float, float, float);
    void   CreateOrthographicProjection(float, float,float, float);
    void   CreateView(const eavlPoint3&, const eavlPoint3&, const eavlVector3&);
    void   CreateRotateX(double radians);
    void   CreateRotateY(double radians);
    void   CreateRotateZ(double radians);

    float* GetOpenGLMatrix4x4();

    // friends
    friend ostream& operator<<(ostream&,const eavlMatrix4x4&);
};

#endif
