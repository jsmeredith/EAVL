// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_H
#define EAVL_SCENE_RENDERER_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"

// ****************************************************************************
// Class:  eavlRenderer
//
// Purpose:
///   Base class for renderers.
//
// Programmer:  Jeremy Meredith
// Creation:    July 18, 2012
//
// Modifications:
//   Jeremy Meredith, Mon Mar  4 15:44:23 EST 2013
//   Big refactoring; more consistent internal code with less
//   duplication and cleaner external API.
//
// ****************************************************************************
class eavlSceneRenderer
{
  public:
    eavlSceneRenderer()
    {
        
    }
    virtual ~eavlSceneRenderer()
    {
    }
    /*
    void Render()
    {
    }
    */
    virtual void StartScene() { }
    virtual void EndScene() { }
    virtual void RenderTriangle(double x0, double y0, double z0,
                                double x1, double y1, double z1,
                                double x2, double y2, double z2,
                                double u0, double v0, double w0,
                                double u1, double v1, double w1,
                                double u2, double v2, double w2,
                                double s0, double s1, double s2)
    {
    }
    virtual void RenderPoints(int npts, double *pts,
                              eavlField *f,
                              double vmin, double vmax,
                              eavlColorTable *ct) { }
    virtual void RenderCells0D(eavlCellSet *cs,
                               int npts, double *pts,
                               eavlField *f,
                               double vmin, double vmax,
                               eavlColorTable *ct) { }
    virtual void RenderCells1D(eavlCellSet *cs,
                               int npts, double *pts,
                               eavlField *f,
                               double vmin, double vmax,
                               eavlColorTable *ct) { }
    virtual void RenderCells2D(eavlCellSet *cs,
                               int npts, double *pts,
                               eavlField *f,
                               double vmin, double vmax,
                               eavlColorTable *ct,
                               eavlField *normals) { }
    virtual void RenderCells3D(eavlCellSet *cs,
                               int npts, double *pts,
                               eavlField *f,
                               double vmin, double vmax,
                               eavlColorTable *ct) { }
};


#endif
