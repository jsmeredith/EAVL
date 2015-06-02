// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_TEST_VR_H
#define EAVL_SCENE_RENDERER_TEST_VR_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlTexture.h"
#include "eavlTimer.h"
#include "eavlVolumeRendererMutator.h"

// ****************************************************************************
// Class:  eavlSceneRendererSimpleVR
//
// Purpose:
///   A very simple volume renderer.
//
// Programmer:  Jeremy Meredith
// Creation:    July 28, 2014
//
// Modifications:
//
// ****************************************************************************
class eavlSceneRendererTestVR : public eavlSceneRenderer
{
    eavlVolumeRendererMutator* caster;
    bool doOnce;
    int numSamples;
    string ctName;
  public:
    eavlSceneRendererTestVR()
    {
        numSamples = 250;
        caster = new eavlVolumeRendererMutator();
        caster->setGPU(true);
        doOnce = true;
        ctName = "";
        
    }
    virtual ~eavlSceneRendererTestVR()
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

    virtual void SetActiveColorTable(eavlColorTable ct)
    {
        eavlSceneRenderer::SetActiveColorTable(ct);
        if(ct.GetName(0!=ctName)
        {
            ctName=ct.GetName();
            caster->setColorMap3f(colors,ncolors);
            //cout<<"Setting Active Color Table"<<endl;
        }
        
    }

    // ------------------------------------------------------------------------

    virtual void StartTetrahedra()
    {
    }

    virtual void EndTetrahedra()
    {
    }

    

    virtual void AddTetrahedronVs(double x0, double y0, double z0, //A
                                  double x1, double y1, double z1, //B
                                  double x2, double y2, double z2, //C
                                  double x3, double y3, double z3, //D
                                  double s0, double s1, double s2, double s3)
    {
       if(doOnce)
       {
        cout<<"tets being sent"<<endl;
        doOnce=false;
       }
       caster->scene.addTet(eavlVector3(x0,y0,z0), eavlVector3(x1,y1,z1), eavlVector3(x2,y2,z2), eavlVector3(x3,y3,z3),
                            s0, s1, s2, s3);
       
    }

    // ------------------------------------------------------------------------

    virtual void AddTriangleVnVs(double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double)
    {
    }


    // ------------------------------------------------------------------------

    virtual void AddPointVs(double, double, double, double, double)
    {
    }

    // ------------------------------------------------------------------------

    virtual void AddLineVs(double, double, double,
                           double, double, double,
                           double, double)
    {
    }


    // ------------------------------------------------------------------------
    virtual void Render()
    {
        cout<<"Starting to render"<<endl;
        if(caster->scene.getNumTets() == 0) return;
        
        int tframe = eavlTimer::Start();
        caster->setResolution(view.h,view.w);
        float magnitude=caster->scene.getSceneMagnitude();
        float step = magnitude/(float)numSamples;
        cout<<"STEP SIZE "<<step<<endl;
        caster->setSampleDelta(step);
        /*Set up field of view: caster takes the half FOV in degrees*/
        float fovx= 2.f*atan(tan(view.view3d.fov/2.f)*view.w/view.h);
        fovx*=180.f/M_PI;
        caster->setFOVy((view.view3d.fov*(180.f/M_PI))/2.f);
        caster->setFOVx( fovx/2.f );

        caster->setZoom(view.view3d.zoom);
        //caster->setZoom(5.f);
        //cout<<"ZOOM : "<<view.view3d.zoom<<endl;
        eavlVector3 lookdir = (view.view3d.at - view.view3d.from).normalized();
        eavlVector3 right = (lookdir % view.view3d.up).normalized();
        /* caster is a lefty, so this is flip so down is not up */
        eavlVector3 up = ( lookdir % right).normalized();  

        caster->setLookAtPos(view.view3d.at.x,view.view3d.at.y,view.view3d.at.z);
        caster->setCameraPos(view.view3d.from.x,view.view3d.from.y,view.view3d.from.z);
        //caster->setCameraPos(20,20,20);
        caster->setUp(up.x,up.y,up.z);
        
        

        caster->Execute();

        glColor3f(1,1,1);
        glDisable(GL_BLEND);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        const float* rgb   = caster->getFrameBuffer()->GetTuple(0);
        //const float* depth = tracer->getDepthBuffer()->GetTuple(0);
        // draw the pixel colors
        glDrawPixels(view.w, view.h, GL_RGBA, GL_FLOAT, &rgb[0]);

        // drawing the Z buffer will overwrite the pixel colors
        // unless you actively prevent it....
        //glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        //glDepthMask(GL_TRUE);
        // For some bizarre reason, you need GL_DEPTH_TEST enabled for
        // it to write into the Z buffer. 
        //glEnable(GL_DEPTH_TEST);

        // draw the z buffer
        //glDrawPixels(view.w, view.h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]);

        // set the various masks back to "normal"
        //glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        //glDepthMask(GL_TRUE);
        
        cerr<<"\nTotal Frame Time   : "<<eavlTimer::Stop(tframe,"")<<endl;
        doOnce =  true;
    }

    


};


#endif
