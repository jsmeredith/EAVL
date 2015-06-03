// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_PS_H
#define EAVL_SCENE_RENDERER_PS_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlSceneRendererPS.h"

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
  protected:
    eavlMatrix4x4 T,S;
    eavlMatrix4x4 W;
    int trianglecount;
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
        T.CreateTranslate(1,1,0);
        S.CreateScale(0.5 * view.w, 0.5*view.h, 1);
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

    // ------------------------------------------------------------------------
    virtual void StartTriangles()
    {
        eavlRenderSurfacePS *surf = dynamic_cast<eavlRenderSurfacePS*>(surface);
        if (!surf)
            return;

        surf->ps << "gsave" << endl;
        surf->ps << "<<" << endl;
        surf->ps << "/ShadingType 4" << endl;
        surf->ps << "/ColorSpace [/DeviceRGB]" << endl;
        surf->ps << "/DataSource" << endl;
        surf->ps << "[" << endl;

        trianglecount = 0;
    }

    virtual void EndTriangles()
    {
        eavlRenderSurfacePS *surf = dynamic_cast<eavlRenderSurfacePS*>(surface);
        if (!surf)
            return;

        surf->ps << "]" << endl;
        surf->ps << ">>" << endl;
        surf->ps << "shfill" << endl;
        surf->ps << "grestore" << endl;
    }

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
        eavlRenderSurfacePS *surf = dynamic_cast<eavlRenderSurfacePS*>(surface);
        if (!surf)
            return;


        if (trianglecount == 500)
        {
            surf->ps << "]" << endl;
            surf->ps << ">>" << endl;
            surf->ps << "shfill" << endl;
            surf->ps << "<<" << endl;
            surf->ps << "/ShadingType 4" << endl;
            surf->ps << "/ColorSpace [/DeviceRGB]" << endl;
            surf->ps << "/DataSource" << endl;
            surf->ps << "[" << endl;
            trianglecount = 0;
        }
        trianglecount++;

        eavlPoint3 p0(x0,y0,z0);
        eavlPoint3 p1(x1,y1,z1);
        eavlPoint3 p2(x2,y2,z2);

        p0 = S*T * view.P * view.V * p0;
        p1 = S*T * view.P * view.V * p1;
        p2 = S*T * view.P * view.V * p2;

        int ci0 = float(ncolors-1) * s0;
        eavlColor c0(colors[ci0*3+0], colors[ci0*3+1], colors[ci0*3+2]);
        int ci1 = float(ncolors-1) * s1;
        eavlColor c1(colors[ci1*3+0], colors[ci1*3+1], colors[ci1*3+2]);
        int ci2 = float(ncolors-1) * s2;
        eavlColor c2(colors[ci2*3+0], colors[ci2*3+1], colors[ci2*3+2]);

        surf->ps << "0" << endl; // edge flag
        surf->ps << p0.x << " " << p0.y << "     ";
        surf->ps << c0.c[0] << " " << c0.c[1] << " " << c0.c[2] << endl;

        surf->ps << "0" << endl; // edge flag
        surf->ps << p1.x << " " << p1.y << "     ";
        surf->ps << c1.c[0] << " " << c1.c[1] << " " << c1.c[2] << endl;

        surf->ps << "0" << endl; // edge flag
        surf->ps << p2.x << " " << p2.y << "     ";
        surf->ps << c2.c[0] << " " << c2.c[1] << " " << c2.c[2] << endl;
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
        eavlPoint3 p(x,y,z);

        p = S*T * view.P * view.V * p;

        int ci = float(ncolors-1) * s;
        eavlColor c(colors[ci*3+0], colors[ci*3+1], colors[ci*3+2]);

        eavlRenderSurfacePS *surf = dynamic_cast<eavlRenderSurfacePS*>(surface);
        if (!surf)
            return;

        surf->ps << "newpath" << endl;
        surf->ps << c.c[0] << " " << c.c[1] << " " << c.c[2] << " setrgbcolor" << endl;
        surf->ps << p.x << " " << p.y << " 6 0 360 arc" << endl;
        surf->ps << "fill" << endl;

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
        eavlPoint3 p0(x0,y0,z0);
        eavlPoint3 p1(x1,y1,z1);

        p0 = S*T * view.P * view.V * p0;
        p1 = S*T * view.P * view.V * p1;

        eavlRenderSurfacePS *surf = dynamic_cast<eavlRenderSurfacePS*>(surface);
        if (!surf)
            return;

        int ci0 = float(ncolors-1) * s0;
        eavlColor c0(colors[ci0*3+0], colors[ci0*3+1], colors[ci0*3+2]);
        int ci1 = float(ncolors-1) * s1;
        eavlColor c1(colors[ci1*3+0], colors[ci1*3+1], colors[ci1*3+2]);

        ///\todo: line width and color interpolation
        surf->ps << "newpath" << endl;
        surf->ps << c0.c[0] << " " << c0.c[1] << " " << c0.c[2] << " setrgbcolor" << endl;
        surf->ps << p0.x << " " << p0.y << " moveto" << endl;
        surf->ps << p1.x << " " << p1.y << " lineto" << endl;
        surf->ps << 2.0 << " setlinewidth" << endl;
        surf->ps << "stroke" << endl;        
    }

    virtual void Render()
    {
    }

};


#endif
