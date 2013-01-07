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
    eavlPoint3   from;
    eavlPoint3   at;
    eavlVector3  up;
    float        fov;
    float        aspect;
    float        nearplane;
    float        farplane;

    void UpdateProjectionMatrix()
    {
        P.CreatePerspectiveProjection(nearplane, farplane, fov, aspect);
    }
    void UpdateViewMatrix()
    {
        V.CreateView(from,at,up);
    }

    // Matrices
    eavlMatrix4x4  P;  ///< Projection matrix
    eavlMatrix4x4  V;  ///< View matrix
};

#endif
