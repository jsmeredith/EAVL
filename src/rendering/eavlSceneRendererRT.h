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
    bool canRender;
  public:
    eavlSceneRendererRT()
    {
        tracer = new eavlRayTracerMutator();
        tracer->setDepth(1);
        tracer->setVerbose(true);
        tracer->setAOMax(3);
        tracer->setOccSamples(4);
        tracer->setAO(true);
        tracer->setBVHCacheName(""); // don't use cache

        canRender=false;
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
        canRender=true;
    }


    // ------------------------------------------------------------------------

    virtual void StartPoints()
    {
        //glDisable(GL_LIGHTING);
        //glPointSize(3);
        //glBegin(GL_POINTS);
    }

    virtual void EndPoints()
    {
        //glEnd();
    }

    virtual void AddPointVs(double x, double y, double z, double r, double s)
    {
        //glTexCoord1f(s);
        //glVertex3d(x,y,z);
    }

    // ------------------------------------------------------------------------

    virtual void StartLines()
    {
        //glDisable(GL_LIGHTING);
        //glLineWidth(2);
        //glBegin(GL_LINES);
    }

    virtual void EndLines()
    {
        //glEnd();
    }

    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
    {
        //glTexCoord1f(s0);
        //glVertex3d(x0,y0,z0);

        //glTexCoord1f(s1);
        //glVertex3d(x1,y1,z1);
    }


    // ------------------------------------------------------------------------
    virtual void Render(eavlView v)
    {
        if(!canRender) return;
        tracer->setResolution(v.h,v.w);
        tracer->setLightParams(v.view3d.from.x,v.view3d.from.y,v.view3d.from.z, 1, 1, 0, 0);
        float aspect=v.w/v.h;
        tracer->setFOVx(45);
        tracer->setFOVy(30);//Todo fix this
        eavlVector3 lookdir = (v.view3d.at - v.view3d.from).normalized();
        tracer->lookAtPos(v.view3d.at.x,v.view3d.at.y,v.view3d.at.z);
        tracer->setCameraPos(v.view3d.from.x,v.view3d.from.y,v.view3d.from.z);
        eavlVector3 right = (lookdir % v.view3d.up).normalized();
        eavlVector3 up = (right % lookdir).normalized();
        tracer->setUp(up.x,up.y,up.z);
        

        tracer->Execute();

        glColor3f(1,1,1);
        glDisable(GL_BLEND);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        const float* rgb=tracer->getFrameBuffer()->GetTuple(0);
        const float* depth=tracer->getDepthBuffer()->GetTuple(0);
        // draw the pixel colors
        glDrawPixels(v.w, v.h, GL_RGB, GL_FLOAT, &rgb[0]);

        // drawing the Z buffer will overwrite the pixel colors
        // unless you actively prevent it....
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_TRUE);
        // For some bizarre reason, you need GL_DEPTH_TEST enabled for
        // it to write into the Z buffer. 
        glEnable(GL_DEPTH_TEST);

        // draw the z buffer
        glDrawPixels(v.w, v.h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]);

        // set the various masks back to "normal"
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        canRender=false;
    }

};


#endif
