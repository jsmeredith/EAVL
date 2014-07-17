// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_RT_H
#define EAVL_SCENE_RENDERER_RT_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlRayTracerMutator.h"
// ****************************************************************************
// Class:  eavlSceneRendererRT
//
// Purpose:
///   
///   
//
// Programmer:  Matt Larsen
// Creation:    July 17, 2014
//
// Modifications:
//
// ****************************************************************************


class eavlSceneRendererRT : public eavlSceneRenderer
{
  private:
    eavlRayTracerMutator* tracer;
  public:
    eavlSceneRendererRT()
    {
        tracer = new eavlRayTracerMutator();
        tracer->setDepth(1);
        tracer->setVerbose(true);
        tracer->setAOMax(4);
        tracer->setOccSamples(4);
        tracer->setAO(false);
    }
    ~eavlSceneRendererRT()
    {
        delete tracer;
    }
    virtual void SetActiveColor(eavlColor c)
    {
        glColor3fv(c.c);
        glDisable(GL_TEXTURE_1D);
    }
    virtual void SetActiveColorTable(string ct)
    {
        glColor3fv(eavlColor::white.c);
        glEnable(GL_TEXTURE_1D);
    }


    // ------------------------------------------------------------------------
    virtual void StartTriangles()
    {
        glBegin(GL_TRIANGLES);
    }

    virtual void EndTriangles()
    {
        glEnd();
    }

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
        /*glNormal3d(u0,v0,w0);
        glTexCoord1f(s0);
        glVertex3d(x0,y0,z0);

        glNormal3d(u1,v1,w1);
        glTexCoord1f(s1);
        glVertex3d(x1,y1,z1);

        glNormal3d(u2,v2,w2);
        glTexCoord1f(s2);
        glVertex3d(x2,y2,z2);*/

        tracer->scene->addTriangle(eavlVector3(x0,y0,z0) , eavlVector3(x1,y1,z1), eavlVector3(x2,y2,z2),
                                   eavlVector3(u0,v0,w0) , eavlVector3(u1,v1,w1), eavlVector3(u2,v2,w2),
                                   s0,s1,s2,  "default");
    }


    // ------------------------------------------------------------------------

    virtual void StartPoints()
    {
        glDisable(GL_LIGHTING);
        glPointSize(3);
        glBegin(GL_POINTS);
    }

    virtual void EndPoints()
    {
        glEnd();
    }

    virtual void AddPointVs(double x, double y, double z, double r, double s)
    {
        glTexCoord1f(s);
        glVertex3d(x,y,z);
    }

    // ------------------------------------------------------------------------

    virtual void StartLines()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2);
        glBegin(GL_LINES);
    }

    virtual void EndLines()
    {
        glEnd();
    }

    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
    {
        glTexCoord1f(s0);
        glVertex3d(x0,y0,z0);

        glTexCoord1f(s1);
        glVertex3d(x1,y1,z1);
    }


    // ------------------------------------------------------------------------
    virtual void Render(eavlView v)
    {
        glFinish();
    }

};


#endif
