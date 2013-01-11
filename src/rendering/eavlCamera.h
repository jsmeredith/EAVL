// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CAMERA_H
#define EAVL_CAMERA_H

#include "eavlVector3.h"
#include "eavlPoint3.h"
#include "eavlMatrix4x4.h"

// ****************************************************************************
// Class:  eavlCamera
//
// Purpose:
///   3D Camera.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlCamera
{
  public:
    // common parameters
    float        aspect;

    // 3d parameters
    eavlPoint3   from;
    eavlPoint3   at;
    eavlVector3  up;
    float        nearplane;
    float        farplane;

    bool         perspective;
    float        fov; // perspective only
    float        size; // ortho only

    // 2d parameters
    bool         twod;
    float        l,r,t,b;
    eavlCamera()
    {
        aspect = 1;
        size = 1;
        fov = 0.5;
        perspective = true;
        twod = false;
    }
    
    void UpdateProjectionMatrix()
    {
        if (twod)
            P.CreateOrthographicProjection(fabs(b-t), +1, -1, aspect);
        else if (perspective)
            P.CreatePerspectiveProjection(nearplane, farplane, fov, aspect);
        else
            P.CreateOrthographicProjection(size, nearplane, farplane, aspect);
    }
    void UpdateViewMatrix()
    {
        if (twod)
        {
            at = eavlPoint3((l+r)/2., (t+b)/2., 0);
            from = at + eavlVector3(0,0,1);
            up = eavlVector3(0,1,0);
        }
        V.CreateView(from,at,up);
    }

    // Matrices
    eavlMatrix4x4  P;  ///< Projection matrix
    eavlMatrix4x4  V;  ///< View matrix
};

#endif
