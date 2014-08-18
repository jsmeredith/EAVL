// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLEP_VR_H
#define EAVL_SCENE_RENDERER_SIMPLEP_VR_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlTimer.h"
#include "eavlSimpleVRMutator.h"

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
class eavlSceneRendererSimplePVR : public eavlSceneRenderer
{
    eavlSimpleVRMutator* vr;
    bool doOnce;
    int numSamples;
    string ctName;
  public:
    eavlSceneRendererSimplePVR()
    {
        numSamples = 300;
        vr = new eavlSimpleVRMutator();
        vr->setGPU(true);
        vr->setVerbose(true);
        doOnce = true;
        ctName = "";
        
    }
    virtual ~eavlSceneRendererSimplePVR()
    {
        delete vr;
    }

    virtual void StartScene()
    {
        eavlSceneRenderer::StartScene();
        vr->clear();
        
    }

    virtual void EndScene()
    {
        eavlSceneRenderer::EndScene();

    }

    virtual void SetActiveColorTable(string ct)
    {

        eavlSceneRenderer::SetActiveColorTable(ct);
        if(ct!=ctName)
        {
            ctName=ct;
            vr->setColorMap3f(colors,ncolors);
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
       vr->scene->addTet(eavlVector3(x0,y0,z0), eavlVector3(x1,y1,z1), eavlVector3(x2,y2,z2), eavlVector3(x3,y3,z3),
                            s0, s1, s2, s3);
       //cout<<"Scalars "<<s0<<" "<<s1<<" "<<s2<<" "<<s3<<endl;
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
        if(vr->scene->getNumTets() == 0) return;
        
        int tframe = eavlTimer::Start();
        vr->setView(view);
        vr->Execute();

        /*glColor3f(1,1,1);
        glDisable(GL_BLEND);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        const float* rgba   = vr->getFrameBuffer()->GetTuple(0);
        const float* depth = vr->getDepthBuffer()->GetTuple(0);
        // draw the pixel colors
        glDrawPixels(view.w, view.h, GL_RGBA, GL_FLOAT, &rgba[0]);
        
        // drawing the Z buffer will overwrite the pixel colors
        // unless you actively prevent it....
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_TRUE);
        // For some bizarre reason, you need GL_DEPTH_TEST enabled for
        // it to write into the Z buffer. 
        glEnable(GL_DEPTH_TEST);

        // draw the z buffer
        glDrawPixels(view.w, view.h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]);

        // set the various masks back to "normal"
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        
        
        doOnce =  true;
        */
        //int* tmp = (int*) vr->getFrameBuffer()->GetHostArray();
        //for (int i=0 ; i<1000; i++) cout<<" "<<tmp[i];
        //cout<<"Byte "<<(int)b<<endl;

        cerr<<"\nTotal Frame Time   : "<<eavlTimer::Stop(tframe,"")<<endl;
    }

    virtual unsigned char *GetRGBAPixels()
    {
        return (unsigned char *) vr->getFrameBuffer()->GetHostArray();
    }

    virtual float *GetDepthPixels()
    {    
        return (float *) vr->getDepthBuffer()->GetHostArray();
    }


};




#endif
