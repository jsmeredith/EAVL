// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_PS_H
#define EAVL_SCENE_RENDERER_PS_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"

// ****************************************************************************
// Class:  eavlSceneRendererPS
//
// Purpose:
///   Write to vector postscript files.
//
// Programmer:  Jeremy Meredith
// Creation:    August  6, 2014
//
// Modifications:
//
// ****************************************************************************
class eavlSceneRendererPS : public eavlSceneRenderer
{
  public:
    eavlSceneRendererPS()
    {
    }
    virtual ~eavlSceneRendererPS()
    {
    }

    virtual void StartScene()
    {
        eavlSceneRenderer::StartScene();
    }

    virtual void EndScene()
    {
        eavlSceneRenderer::EndScene();
    }

    // we're not caching anything; always say we need it
    virtual bool NeedsGeometryForPlot(int)
    {
        return true;
    }

    virtual void SetActiveColor(eavlColor c)
    {
    }
    virtual void SetActiveColorTable(string ctname)
    {
    }


    // ------------------------------------------------------------------------
    virtual void StartTriangles()
    {
    }

    virtual void EndTriangles()
    {
    }

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
    }


    // ------------------------------------------------------------------------

    virtual void StartPoints()
    {
    }

    virtual void EndPoints()
    {
    }

    virtual void AddPointVs(double x, double y, double z, double r, double s)
    {
    }

    // ------------------------------------------------------------------------

    virtual void StartLines()
    {
    }

    virtual void EndLines()
    {
    }

    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
    {
    }

    virtual void Render()
    {
    }

};


#endif
