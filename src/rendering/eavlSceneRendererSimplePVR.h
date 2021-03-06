// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLEP_PVR_H
#define EAVL_SCENE_RENDERER_SIMPLEP_PVR_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlTimer.h"
#include "eavlSimpleVRMutator.h"

// ****************************************************************************
// Class:  eavlSceneRendererSimplePVR
//
// Purpose:
///   A parallel verion of the very simple volume renderer.
//
// Programmer:  Jeremy Meredith
// Creation:    July 28, 2014
//
// Modifications: Matt Larsen - Parallel version
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

    virtual void SetActiveColorTable(eavlColorTable ct)
    {

        eavlSceneRenderer::SetActiveColorTable(ct);
        if(ct.GetName()!=ctName)
        {
            ctName=ct.GetName();
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
        int tframe = eavlTimer::Start();
        vr->setView(view);
        vr->Execute();
        cerr<<"\nTotal Frame Time   : "<<eavlTimer::Stop(tframe,"")<<endl;
    }

    virtual unsigned char *GetRGBAPixels()
    {
        return (unsigned char *) vr->getFrameBuffer()->GetHostArray();
    }

    virtual float *GetDepthPixels()
    {    
        float proj22=view.P(2,2);
        float proj23=view.P(2,3);
        float proj32=view.P(3,2);

        return (float *) vr->getDepthBuffer(proj22,proj23,proj32)->GetHostArray();
    }


};




#endif
