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
#include "eavlRTUtil.h"
#include "eavlTransferFunction.h"

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
    eavlTransferFunction tf;
    float                tfVals[4*1024];
    bool doOnce;
    int numSamples;
    string ctName;
    bool customColorTable;
  public:
    eavlSceneRendererSimplePVR()
    {
        numSamples = 500;
        vr = new eavlSimpleVRMutator();
        vr->setVerbose(false);
        vr->setNumSamples(numSamples);
        vr->setNumPasses(2);
        doOnce = true;
        ctName = "";
        customColorTable = false;
        
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
        if(ct.GetName()!=ctName && !customColorTable)
        {
            //
            // Eavl creates a "uniqiue" name by adding 
            // two numbers to the front of the name
            //
            ctName=ct.GetName().substr(2);      
            tf.Clear();
            tf.SetByColorTableName(ctName); 
            tf.CreateDefaultAlpha();
            tf.GetTransferFunction(ncolors, tfVals);
            vr->setColorMap4f(tfVals,ncolors);
        }
        
    }
    
    eavlTransferFunction * GetTransferFunction()
    {
        return &tf;
    }
    
    void SampleTransferFunction()
    {   ncolors = 1024;
        customColorTable = true; //keep the custom table from being overwritten
        tf.GetTransferFunction(ncolors, tfVals);
        vr->setColorMap4f(tfVals,ncolors);
    }
    
    void SetTransparentBG(bool on)
    {
        vr->setTransparentBG(on);
    }
    

    // ------------------------------------------------------------------------

    virtual void StartTetrahedra()
    {
        vr->clear();
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

        vr->setBGColor(surface->bgColor);
        vr->setView(view);
        
        //int tframe = eavlTimer::Start();
        vr->Execute();
        //cerr<<"\nTotal Frame Time   : "<<eavlTimer::Stop(tframe,"")<<endl;
    }
    void getImageSubsetDims(int *dims)
    {
      vr->getImageSubsetDims(dims);
    }
    virtual unsigned char *GetRGBAPixels()
    {
        //cout<<"Getting pixels"<<endl;
        unsigned char * pixels = (unsigned char *) vr->getFrameBuffer()->GetHostArray();
        //writeFrameBufferBMP(500, 500, vr->getFrameBuffer(), "output.bmp");
        return pixels;
    }

    virtual float *GetDepthPixels()
    {    
        float proj22=view.P(2,2);
        float proj23=view.P(2,3);
        float proj32=view.P(3,2);
        float *zBuffer = (float *) vr->getDepthBuffer(proj22,proj23,proj32)->GetHostArray();
        return zBuffer;
    }


};




#endif
