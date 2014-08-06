// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef MATRIX4X4_H
#define MATRIX4X4_H

class eavlVector3;
class eavlPoint3;
class eavlVector4;
#include "STL.h"
#include "eavlUtility.h"

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
    EAVL_HOSTDEVICE eavlMatrix4x4();
    EAVL_HOSTDEVICE eavlMatrix4x4(float,float,float,float,
                                  float,float,float,float,
                                  float,float,float,float,
                                  float,float,float,float);
    EAVL_HOSTDEVICE eavlMatrix4x4(const eavlMatrix4x4&);

    // assignment operator
    EAVL_HOSTDEVICE void          operator=(const eavlMatrix4x4&);

    // multiply matrix*matrix
    EAVL_HOSTDEVICE eavlMatrix4x4 operator*(const eavlMatrix4x4&) const;

    // transform point
    EAVL_HOSTDEVICE eavlPoint3    operator*(const eavlPoint3&) const;
    // transform vector
    EAVL_HOSTDEVICE eavlVector3   operator*(const eavlVector3&) const;
    // transform point/vector
    EAVL_HOSTDEVICE eavlVector3   operator*(const eavlVector4&) const;

    EAVL_HOSTDEVICE void   Invert();
    EAVL_HOSTDEVICE void   Transpose();

    EAVL_HOSTDEVICE const float &operator()(int r, int c) const {return m[r][c];}
    EAVL_HOSTDEVICE float &operator()(int r, int c) {return m[r][c];}

    // utility
    EAVL_HOSTDEVICE void   CreateIdentity();
    EAVL_HOSTDEVICE void   CreateZero();
    EAVL_HOSTDEVICE void   CreateTrackball(float,float, float,float);
    EAVL_HOSTDEVICE void   CreateTranslate(float, float, float);
    EAVL_HOSTDEVICE void   CreateTranslate(const eavlVector3&);
    EAVL_HOSTDEVICE void   CreateTranslate(const eavlPoint3&);
    EAVL_HOSTDEVICE void   CreateRBT(const eavlPoint3&, const eavlPoint3&, const eavlVector3&);
    EAVL_HOSTDEVICE void   CreateWorld(const eavlPoint3&, const eavlVector3&, const eavlVector3&, const eavlVector3&);
    EAVL_HOSTDEVICE void   CreateScale(float,float,float);
    EAVL_HOSTDEVICE void   CreateScale(float);
    EAVL_HOSTDEVICE void   CreatePerspectiveProjection(float,float, float, float);
    EAVL_HOSTDEVICE void   CreateOrthographicProjection(float, float,float, float);
    EAVL_HOSTDEVICE void   CreateView(const eavlPoint3&, const eavlPoint3&, const eavlVector3&);
    EAVL_HOSTDEVICE void   CreateRotateX(double radians);
    EAVL_HOSTDEVICE void   CreateRotateY(double radians);
    EAVL_HOSTDEVICE void   CreateRotateZ(double radians);

    EAVL_HOSTDEVICE float  Determinant() const;

    EAVL_HOSTDEVICE float* GetOpenGLMatrix4x4();

    // friends
    friend ostream& operator<<(ostream&,const eavlMatrix4x4&);
};

#include "eavlVector3.h"
#include "eavlVector4.h"
#include "eavlPoint3.h"
#include <math.h>

EAVL_HOSTDEVICE eavlMatrix4x4::eavlMatrix4x4()
{
    CreateIdentity();
}

EAVL_HOSTDEVICE eavlMatrix4x4::eavlMatrix4x4(float m00,float m01,float m02,float m03,
                                             float m10,float m11,float m12,float m13,
                                             float m20,float m21,float m22,float m23,
                                             float m30,float m31,float m32,float m33)
{
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[0][3] = m03;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[1][3] = m13;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    m[2][3] = m23;
    m[3][0] = m30;
    m[3][1] = m31;
    m[3][2] = m32;
    m[3][3] = m33;
}

EAVL_HOSTDEVICE eavlMatrix4x4::eavlMatrix4x4(const eavlMatrix4x4 &R)
{
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
            m[r][c] = R.m[r][c];
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::operator=(const eavlMatrix4x4 &R)
{
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
            m[r][c] = R.m[r][c];
}


EAVL_HOSTDEVICE eavlMatrix4x4
eavlMatrix4x4::operator*(const eavlMatrix4x4 &R) const
{
    eavlMatrix4x4 C;
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
            C.m[r][c] = m[r][0] * R.m[0][c] +
                        m[r][1] * R.m[1][c] +
                        m[r][2] * R.m[2][c] +
                        m[r][3] * R.m[3][c];
    return C;
}

EAVL_HOSTDEVICE eavlPoint3
eavlMatrix4x4::operator*(const eavlPoint3 &r) const
{
    float x,y,z,w;

    x = m[0][0] * r.x +
        m[0][1] * r.y +
        m[0][2] * r.z +
        m[0][3];
    y = m[1][0] * r.x +
        m[1][1] * r.y +
        m[1][2] * r.z +
        m[1][3];
    z = m[2][0] * r.x +
        m[2][1] * r.y +
        m[2][2] * r.z +
        m[2][3];
    w = m[3][0] * r.x +
        m[3][1] * r.y +
        m[3][2] * r.z +
        m[3][3];

    float iw = 1. / w;
    x *= iw;
    y *= iw;
    z *= iw;

    return eavlPoint3(x,y,z);
}

EAVL_HOSTDEVICE eavlVector3
eavlMatrix4x4::operator*(const eavlVector4 &r) const
{
    float x,y,z,w;

    x = m[0][0] * r.x +
        m[0][1] * r.y +
        m[0][2] * r.z +
        m[0][3] * r.w;
    y = m[1][0] * r.x +
        m[1][1] * r.y +
        m[1][2] * r.z +
        m[1][3] * r.w;
    z = m[2][0] * r.x +
        m[2][1] * r.y +
        m[2][2] * r.z +
        m[2][3] * r.w;
    w = m[3][0] * r.x +
        m[3][1] * r.y +
        m[3][2] * r.z +
        m[3][3] * r.w;

    float iw = 1. / w;
    x *= iw;
    y *= iw;
    z *= iw;

    return eavlVector3(x,y,z);
}

EAVL_HOSTDEVICE eavlVector3
eavlMatrix4x4::operator*(const eavlVector3 &r) const
{
    float x,y,z;

    x = m[0][0] * r.x +
        m[0][1] * r.y +
        m[0][2] * r.z;
    y = m[1][0] * r.x +
        m[1][1] * r.y +
        m[1][2] * r.z;
    z = m[2][0] * r.x +
        m[2][1] * r.y +
        m[2][2] * r.z;

    return eavlVector3(x,y,z);
}

EAVL_HOSTDEVICE float *
eavlMatrix4x4::GetOpenGLMatrix4x4()
{
    // inexplicably, opengl expects column-major format for its matrices
    openglm[ 0] = m[0][0];
    openglm[ 1] = m[1][0];
    openglm[ 2] = m[2][0];
    openglm[ 3] = m[3][0];
    openglm[ 4] = m[0][1];
    openglm[ 5] = m[1][1];
    openglm[ 6] = m[2][1];
    openglm[ 7] = m[3][1];
    openglm[ 8] = m[0][2];
    openglm[ 9] = m[1][2];
    openglm[10] = m[2][2];
    openglm[11] = m[3][2];
    openglm[12] = m[0][3];
    openglm[13] = m[1][3];
    openglm[14] = m[2][3];
    openglm[15] = m[3][3];
    return openglm;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateIdentity()
{
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
            m[r][c] = (r==c) ? 1 : 0;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateZero()
{
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
            m[r][c] = 0;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateTranslate(float x,float y,float z)
{
    CreateIdentity();
    m[0][3] = x;
    m[1][3] = y;
    m[2][3] = z;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateTranslate(const eavlVector3 &v)
{
    CreateIdentity();
    m[0][3] = v.x;
    m[1][3] = v.y;
    m[2][3] = v.z;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateTranslate(const eavlPoint3 &p)
{
    CreateIdentity();
    m[0][3] = p.x;
    m[1][3] = p.y;
    m[2][3] = p.z;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateScale(float s)
{
    CreateIdentity();
    m[0][0] = s;
    m[1][1] = s;
    m[2][2] = s;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateScale(float x,float y,float z)
{
    CreateIdentity();
    m[0][0] = x;
    m[1][1] = y;
    m[2][2] = z;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateView(const eavlPoint3 &from,
                      const eavlPoint3 &at,
                      const eavlVector3 &world_up)
{
    eavlVector3 up, right, view_dir;

#ifdef LEFTHANDED
    view_dir = (at - from).normalized();
#else
    view_dir = (from - at).normalized();
#endif
    right    = (world_up % view_dir).normalized();
    up       = (view_dir % right).normalized();

    CreateIdentity();

    m[0][0] = right.x;
    m[0][1] = right.y;
    m[0][2] = right.z;
    m[1][0] = up.x;
    m[1][1] = up.y;
    m[1][2] = up.z;
    m[2][0] = view_dir.x;
    m[2][1] = view_dir.y;
    m[2][2] = view_dir.z;

    m[0][3] = -(right*from);
    m[1][3] = -(up*from);
    m[2][3] = -(view_dir*from);
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateRBT(const eavlPoint3 &from,
                  const eavlPoint3 &at,
                  const eavlVector3 &world_up)
{
    eavlVector3 up, right, view_dir;

#ifdef LEFTHANDED
    view_dir = (at - from).normalized();
#else
    view_dir = (from - at).normalized();
#endif
    right    = (world_up % view_dir).normalized();
    up       = (view_dir % right).normalized();

    CreateIdentity();

    m[0][0] = right.x;
    m[0][1] = up.x;
    m[0][2] = view_dir.x;
    m[1][0] = right.y;
    m[1][1] = up.y;
    m[1][2] = view_dir.y;
    m[2][0] = right.z;
    m[2][1] = up.z;
    m[2][2] = view_dir.z;

    m[0][3] = from.x;
    m[1][3] = from.y;
    m[2][3] = from.z;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateWorld(const eavlPoint3 &neworigin,
                           const eavlVector3 &newx,
                           const eavlVector3 &newy,
                           const eavlVector3 &newz)
{
    CreateIdentity();

    m[0][0] = newx.x;
    m[0][1] = newy.x;
    m[0][2] = newz.x;

    m[1][0] = newx.y;
    m[1][1] = newy.y;
    m[1][2] = newz.y;

    m[2][0] = newx.z;
    m[2][1] = newy.z;
    m[2][2] = newz.z;

    m[0][3] = neworigin.x;
    m[1][3] = neworigin.y;
    m[2][3] = neworigin.z;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreatePerspectiveProjection(float near,
                                           float far,
                                           float fov,
                                           float aspect)
{
    CreateZero();

    ///\todo: unify left-handed code into right-handed code;
    /// I think only the RH path gets correct Z buffer values
#ifdef LEFTHANDED // OLD CODE, LEFT-HANDED
    float  c, s, Q;

    c = cos(fov*0.5);
    s = sin(fov*0.5);
    Q = s/(1.0 - near/far);

    m[0][0] = c/aspect;
    m[1][1] = c;
    m[2][2] = Q;
    m[2][3] = -Q*near;
    m[3][2] = s;
#else // NEW CODE, RIGHT-HANDED
    // derived from formulas at:
    //  http://db-in.com/blog/2011/04/cameras-on-opengl-es-2-x/
    float size = near * tan(fov * 0.5);
    float left = -size*aspect, right = size*aspect, bottom = -size, top=size;
    m[0][0] = 2. * near / (right-left);

    m[1][1] = 2. * near / (top-bottom);

    m[0][2] = (right+left) / (right-left); // 0 when centered
    m[1][2] = (top+bottom) / (top-bottom); // 0 when centered
    m[2][2] = -(far+near)  / (far-near);
    m[3][2] = -1;

    m[2][3] = -(2.*far*near) / (far-near);
#endif
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateOrthographicProjection(float size,
                                            float near,
                                            float far,
                                            float aspect)
{
    CreateZero();

    ///\todo: unify left-handed code into right-handed code
    /// I think only the RH path gets correct Z buffer values
#ifdef LEFTHANDED // OLD CODE, LEFT-HANDED
    float d;
    d = far - near;

    CreateIdentity();
    m[0][0] = 2./(size*aspect);
    m[1][1] = 2./size;
    m[2][2] = 1./d;
    m[2][3] = -near/d;
    m[3][3] = 1;

#else // NEW CODE, RIGHT-HANDED
    // derived from formulas at:
    //  http://db-in.com/blog/2011/04/cameras-on-opengl-es-2-x/
    float left = -size/2. * aspect;
    float right = size/2. * aspect;
    float bottom = -size/2.;
    float top    =  size/2.;

    m[0][0] = 2. / (right - left);
    m[1][1] = 2. / (top - bottom);
    m[2][2] = -2. / (far - near);
    m[0][3] = -(right + left) / (right - left);
    m[1][3] = -(top + bottom) / (top - bottom);
    m[2][3] = -(far + near) / (far - near);
    m[3][3] = 1.;
#endif
}


EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateTrackball(float p1x,float p1y,  float p2x, float p2y)
{
#define RADIUS       0.8        /* z value at x = y = 0.0  */
#define COMPRESSION  3.5        /* multipliers for x and y */
#define AR3 (RADIUS*RADIUS*RADIUS)

    float   q[4];   // quaternion
    eavlVector3  p1, p2; // pointer locations on trackball
    eavlVector3  axis;   // axis of rotation
    double  phi;    // rotation angle (radians)
    double  t;
    eavlMatrix4x4  M;

    // Check for zero mouse movement
    if (p1x==p2x && p1y==p2y)
    {
        CreateIdentity();
        return;
    }


    // Compute z-coordinates for projection of P1 and P2 onto
    // the trackball.
    p1 = eavlVector3(p1x, p1y, AR3/((p1x*p1x+p1y*p1y)*COMPRESSION+AR3));
    p2 = eavlVector3(p2x, p2y, AR3/((p2x*p2x+p2y*p2y)*COMPRESSION+AR3));

    // Compute the axis of rotation and temporarily store it
    // in the quaternion.
    axis = (p2 % p1).normalized();

    // Figure how much to rotate around that axis.
    t = (p2 - p1).norm();
    if (t < -1.0) t = -1.0;
    if (t > +1.0) t = +1.0;
    phi = -2.0*asin(t/(2.0*RADIUS));

    axis *= sin(phi/2.0);
    q[0]  = axis.x;
    q[1]  = axis.y;
    q[2]  = axis.z;
    q[3]  = cos(phi/2.0);

    // normalize quaternion to unit magnitude
    t =  1.0 / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    q[0] *= t;
    q[1] *= t;
    q[2] *= t;
    q[3] *= t;

    //q[2]*=-1;   // swap left/right handed coordinate system

    // create the rotation matrix from the quaternion
    CreateIdentity();

    m[0][0] = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2]);
    m[0][1] = 2.0 * (q[0]*q[1] + q[2]*q[3]);
    m[0][2] = (2.0 * (q[2]*q[0] - q[1]*q[3]) );

    m[1][0] = 2.0 * (q[0]*q[1] - q[2]*q[3]);
    m[1][1] = 1.0 - 2.0 * (q[2]*q[2] + q[0]*q[0]);
    m[1][2] = (2.0 * (q[1]*q[2] + q[0]*q[3]) );

    m[2][0] = (2.0 * (q[2]*q[0] + q[1]*q[3]) );
    m[2][1] = (2.0 * (q[1]*q[2] - q[0]*q[3]) );
    m[2][2] = (1.0 - 2.0 * (q[1]*q[1] + q[0]*q[0]) );
}


EAVL_HOSTDEVICE void eavlMatrix4x4::Transpose()
{
    float t;
    for (int r=0; r<4; r++)
        for (int c=0; c<r; c++)
        {
            t       = m[r][c];
            m[r][c] = m[c][r];
            m[c][r] = t;
        }

}


static void
EAVL_HOSTDEVICE lubksb(eavlMatrix4x4 *a, int *indx, float *b)
{
    int		i, j, ii=-1, ip;
    float	sum;

    for (i=0; i<4; i++) {
        ip = indx[i];
        sum = b[ip];
        b[ip] = b[i];
        if (ii>=0) {
            for (j=ii; j<=i-1; j++) {
                sum -= a->m[i][j] * b[j];
            }
        } else if (sum != 0.0) {
            ii = i;
        }
        b[i] = sum;
    }
    for (i=3; i>=0; i--) {
        sum = b[i];
        for (j=i+1; j<4; j++) {
            sum -= a->m[i][ j] * b[j];
        }
        b[i] = sum/a->m[i][ i];
    }
}

static int
EAVL_HOSTDEVICE ludcmp(eavlMatrix4x4 *a, int *indx, float *d)
{
    float	vv[4];               // implicit scale for each row
    float	big, dum, sum, tmp;
    int		i, imax, j, k;

    *d = 1.0f;
    for (i=0; i<4; i++) {
        big = 0.0f;
        for (j=0; j<4; j++) {
            if ((tmp = (float) fabs(a->m[i][ j])) > big) {
                big = tmp;
            }
        }

        if (big == 0.0f) {
            return 1;
        }

        vv[i] = 1.0f/big;
    }
    for (j=0; j<4; j++) {
        for (i=0; i<j; i++) {
            sum = a->m[i][ j];
            for (k=0; k<i; k++) {
                sum -= a->m[i][ k] * a->m[k][ j];
            }
            a->m[i][ j] = sum;
        }
        big = 0.0f;
        for (i=j; i<4; i++) {
            sum = a->m[i][ j];
            for (k=0; k<j; k++) {
                sum -= a->m[i][ k]*a->m[k][ j];
            }
            a->m[i][ j] = sum;
            if ((dum = vv[i] * (float)fabs(sum)) >= big) {
                big = dum;
                imax = i;
            }
        }
        if (j != imax) {
            for (k=0; k<4; k++) {
                dum = a->m[imax][ k];
                a->m[imax][ k] = a->m[j][ k];
                a->m[j][ k] = dum;
            }
            *d = -(*d);
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if (a->m[j][ j] == 0.0f) {
            a->m[j][ j] = 1.0e-20f;      // can be 0.0 also...
        }
        if (j != 3) {
            dum = 1.0f/a->m[j][ j];
            for (i=j+1; i<4; i++) {
                a->m[i][ j] *= dum;
            }
        }
    }
    return 0;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::Invert()
{
    eavlMatrix4x4 n, y;
    int			i, j, indx[4];
    float		d, col[4];

    n=*this;
    if (ludcmp(&n, indx, &d)) {
        CreateIdentity();
        return;
    }

    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            col[i] = 0.0f;
        }
        col[j] = 1.0f;
        lubksb(&n, indx, col);
        for (i=0; i<4; i++) {
            y.m[i][j] = col[i];
        }
    }
    *this = y;
    return;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateRotateX(double angleRadians)
{
    float *r = &m[0][0];
    r[0]  = 1.;
    r[1]  = 0.;
    r[2]  = 0.;
    r[3]  = 0.;
    r[4]  = 0.;
    r[5]  = cos(angleRadians);
    r[6]  = - sin(angleRadians);
    r[7]  = 0.;
    r[8]  = 0.;
    r[9]  = sin(angleRadians);
    r[10] = cos(angleRadians);
    r[11] = 0.;
    r[12] = 0.;
    r[13] = 0.;
    r[14] = 0.;
    r[15] = 1.;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateRotateY(double angleRadians)
{
    float *r = &m[0][0];
    r[0]  = cos(angleRadians);
    r[1]  = 0.;
    r[2]  = sin(angleRadians);
    r[3]  = 0.;
    r[4]  = 0.;
    r[5]  = 1.;
    r[6]  = 0.;
    r[7]  = 0.;
    r[8]  = - sin(angleRadians);
    r[9]  = 0.;
    r[10] = cos(angleRadians);
    r[11] = 0.;
    r[12] = 0.;
    r[13] = 0.;
    r[14] = 0.;
    r[15] = 1.;
}

EAVL_HOSTDEVICE void
eavlMatrix4x4::CreateRotateZ(double angleRadians)
{
    float *r = &m[0][0];
    r[0]  = cos(angleRadians);
    r[1]  = - sin(angleRadians);
    r[2]  = 0.;
    r[3]  = 0.;
    r[4]  = sin(angleRadians);
    r[5]  = cos(angleRadians);
    r[6]  = 0.;
    r[7]  = 0.;
    r[8]  = 0.;
    r[9]  = 0.;
    r[10]  = 1.;
    r[11] = 0.;
    r[12] = 0.;
    r[13] = 0.;
    r[14] = 0.;
    r[15] = 1.;
}

#define D22(a,b,c,d) (a*d - b*c)
#define D33(e,f,g, h,i,j, k,l,n) (e*D22(i,j,l,n) - f*D22(h,j,k,n) + g*D22(h,i,k,l))

EAVL_HOSTDEVICE float
eavlMatrix4x4::Determinant() const
{
    float a = m[0][0] * D33(m[1][1],m[1][2],m[1][3],
                            m[2][1],m[2][2],m[2][3],
                            m[3][1],m[3][2],m[3][3]);
    float b = m[0][1] * D33(m[1][0],m[1][2],m[1][3],
                            m[2][0],m[2][2],m[2][3],
                            m[3][0],m[3][2],m[3][3]);
    float c = m[0][2] * D33(m[1][0],m[1][1],m[1][3],
                            m[2][0],m[2][1],m[2][3],
                            m[3][0],m[3][1],m[3][3]);
    float d = m[0][3] * D33(m[1][0],m[1][1],m[1][2],
                            m[2][0],m[2][1],m[2][2],
                            m[3][0],m[3][1],m[3][2]);
    return a - b + c - d;
}

#undef D22
#undef D33

#endif
