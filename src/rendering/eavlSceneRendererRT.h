// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_RT_H
#define EAVL_SCENE_RENDERER_RT_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlRayTracer.h"
#include "eavlTimer.h"
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
    eavlRayTracer* tracer;
    bool canRender;
    bool setLight;
    string ctName;
    float pointRadius;
    float lineWidth;
    bool imageSubsetEnabled;
  public:
    eavlSceneRendererRT()
    {
        imageSubsetEnabled = false;
        tracer = new eavlRayTracer();
        //tracer->setDepth(1);
        // /tracer->setVerbose(true);
        //tracer->setAOMax(5);
        //tracer->setOccSamples(4);
        ////tracer->setAO(true);
        //tracer->setBVHCache(false); // don't use cache
        //tracer->setCompactOp(false);
        tracer->setShadowsOn(false);
        setLight = true;
        ctName = "";
        //tracer->setDefaultMaterial(Ka,Kd,Ks);
        pointRadius = .1f;
        lineWidth = .05f;

    }
    ~eavlSceneRendererRT()
    {
        delete tracer;
    }

    void SetPointRadius(float r)
    {
        if(r > 0) pointRadius = r;
    }

    void SetBackGroundColor(float r, float g, float b)
    {
        tracer->setBackgroundColor(r,g,b);
    }

    virtual void SetActiveColor(eavlColor c)
    {
        glColor3fv(c.c);
        glDisable(GL_TEXTURE_1D);
        //cout<<"Setting Active Color"<<endl;
    }
    virtual void SetActiveColorTable(eavlColorTable ct)
    {
        eavlSceneRenderer::SetActiveColorTable(ct);
        if(ct.GetName()!=ctName)
        {
            ctName=ct.GetName();
            tracer->setColorMap3f(colors,ncolors);
            //cout<<"Setting Active Color Table"<<endl;
        }
        
    }

    virtual void StartTetrahedra()
    {
       tracer->startScene();
    }

    virtual void EndTetrahedra()
    {
     
    }


    virtual void AddTetrahedronVs(double x0, double y0, double z0,
                                  double x1, double y1, double z1,
                                  double x2, double y2, double z2,
                                  double x3, double y3, double z3,
                                  double s0, double s1, double s2, double s3)
    {
        AddTriangleVs(x1,y1,z1,
                      x0,y0,z0,
                      x2,y2,z2,
                      s1,s0,s2);
        AddTriangleVs(x0,y0,z0,
                      x1,y1,z1,
                      x3,y3,z3,
                      s0,s1,s3);
        AddTriangleVs(x1,y1,z1,
                      x2,y2,z2,
                      x3,y3,z3,
                      s1,s2,s3);
        AddTriangleVs(x2,y2,z2,
                      x0,y0,z0,
                      x3,y3,z3,
                      s2,s0,s3);
    }
    virtual void StartScene() 
    {
        //cout<<"Calling Start scene"<<endl;
        //tracer->startScene();
    }

    // ------------------------------------------------------------------------
    virtual void StartTriangles()
    {
        //cout<<"Calling Start Tris"<<endl;
        tracer->startScene();

    }

    virtual void EndTriangles()
    {
        //cout<<"Calling End Tris"<<endl;
    }

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
        tracer->scene->addTriangle(eavlVector3(x0,y0,z0) , eavlVector3(x1,y1,z1), eavlVector3(x2,y2,z2),
                                   eavlVector3(u0,v0,w0) , eavlVector3(u1,v1,w1), eavlVector3(u2,v2,w2),
                                   s0,s1,s2);
        //cout<<"Adding T "<<eavlVector3(x0,y0,z0) <<" "<<s0<<" "<< eavlVector3(x1,y1,z1)<<" "<<s1<<" "<< eavlVector3(x2,y2,z2)<<" "<<s2<<endl;
    }


    // ------------------------------------------------------------------------

    virtual void StartPoints()
    {
        tracer->startScene();
    }

    virtual void EndPoints()
    {
        //glEnd();
    }

    virtual void AddPointVs(double x, double y, double z, double r, double s)
    {
        
        //tracer->scene->addSphere(pointRadius,x,y,z,s,"default");
    }

    // ------------------------------------------------------------------------

    virtual void StartLines()
    {
        tracer->startScene();
    }

    virtual void EndLines()
    {
        //glEnd();
    }

    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
    {
        //cout<<"ADDING LINE"<<endl;
        //tracer->scene->addLine(lineWidth, eavlVector3(x0,y0,z0),s0, eavlVector3(x1,y1,z1),s1 );
    }
    void enableImageSubset(bool on)
    {
      imageSubsetEnabled = on;
    }
    void getImageSubsetDims(int *dims)
    {
      if(imageSubsetEnabled) tracer->getImageSubsetDims(dims);
    }
    // ------------------------------------------------------------------------
    virtual void Render()
    {        
    
        if(imageSubsetEnabled) tracer->enableImageSubset(view);
        //int tframe = eavlTimer::Start();
        //cout<<"Triangles "<<tracer->scene->getNumTriangles()<<endl;
        eavlRayCamera * camera = tracer->camera;
        //tracer->setDefaultMaterial(Ka,Kd,Ks);
        camera->setWidth(view.w);
        camera->setHeight(view.h);
        tracer->setBackgroundColor(surface->bgColor.c[0],surface->bgColor.c[1],surface->bgColor.c[2]);
        float magnitude=tracer->scene->getSceneExtentMagnitude();
        //tracer->setOcclusionSamples(4);
        //tracer->setOcclusionDistance(magnitude*.2f);
		//tracer->setOcclusionOn(false);
		    
        /*Set up field of view: tracer takes the half FOV in degrees*/
        float fovx= 2.f*atan(tan(view.view3d.fov/2.f)*view.w/view.h);
        fovx*=180.f/M_PI;
        camera->setFOVY((view.view3d.fov*(180.f/M_PI))/2.f);
        camera->setFOVX( fovx/2.f );

        camera->setCameraZoom(view.view3d.zoom);

        eavlVector3 lookdir = (view.view3d.at - view.view3d.from).normalized();
        eavlVector3 right = (lookdir % view.view3d.up).normalized();
        /* Tracer is a lefty, so this is flip so down is not up */
        eavlVector3 up = ( lookdir % right).normalized();  
        camera->setCameraPosition(view.view3d.from.x,view.view3d.from.y,view.view3d.from.z);
        camera->lookAtPosition(view.view3d.at.x,view.view3d.at.y,view.view3d.at.z);
        camera->setCameraUp(up.x,up.y,up.z);
		eyeLight = true;
        /*Otherwise the light will move with the camera*/
        if(eyeLight)//setLight)
        {
          eavlVector3 minersLight(view.view3d.from.x,view.view3d.from.y,view.view3d.from.z);
          minersLight = minersLight+ up*magnitude*.3f;
          //tracer->setLightParams(minersLight.x,minersLight.y,minersLight.z, 1.f, 1.f, 0.f, 0.f);  /*Light params: intensity, constant, linear and exponential coefficeints*/
          tracer->lightPosition.x = minersLight.x;
          tracer->lightPosition.y = minersLight.y;
          tracer->lightPosition.z = minersLight.z;
        } 
        else
        {
          //tracer->setLightParams(Lx, Ly, Lz, 1.f, 1.f , 0.f , 0.f);
          tracer->lightPosition.x = Lx;
          tracer->lightPosition.y = Ly;
          tracer->lightPosition.z = Lz;
        }

        tracer->render();
        
        //cerr<<"\nTotal Frame Time   : "<<eavlTimer::Stop(tframe,"")<<endl;
    }

    virtual unsigned char *GetRGBAPixels()
    {
        return (unsigned char *) tracer->getFrameBuffer()->GetHostArray();
    }

    virtual float *GetDepthPixels()
    {    
        float proj22=view.P(2,2);
        float proj23=view.P(2,3);
        float proj32=view.P(3,2);
		float *zBuffer = (float *) tracer->getDepthBuffer(proj22,proj23,proj32)->GetHostArray();
        return zBuffer;
    }

};


#endif
