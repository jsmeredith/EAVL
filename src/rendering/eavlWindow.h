// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"
#include "eavlPlot.h"
#include "eavlTexture.h"

// ****************************************************************************
// Class:  eavlWindow
//
// Purpose:
///   Encapsulate an output window (e.g. 3D).
///   \todo: at the moment this only encapsulates textures; we need
///   to pull more of the eavlab infrastructure into this.
//
// Programmer:  Jeremy Meredith
// Creation:    January 23, 2013
//
// Modifications:
// ****************************************************************************
class eavlWindow
{
  public:
    eavlView &view;
    std::map<std::string,eavlTexture*> textures;

  public:
    eavlWindow(eavlView &v) : view(v) { }
    /*
    virtual void ResetView() = 0;
    virtual void Initialize() = 0;
    virtual void Resize(int w, int h) = 0;
    virtual void Paint() = 0;
    */
    eavlTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlTexture *tex)
    {
        textures[s] = tex;
    }

  protected:
  public:
    //
    // Set up ONLY the viewport for world/screen space
    //
    void SetupViewportForWorld()
    {
        float vl, vr, vt, vb;
        view.GetRealViewport(vl,vr,vb,vt);
        glViewport(float(view.w)*(1.+vl)/2.,
                   float(view.h)*(1.+vb)/2.,
                   float(view.w)*(vr-vl)/2.,
                   float(view.h)*(vt-vb)/2.);
    }
    void SetupViewportForScreen()
    {
        glViewport(0, 0, view.w, view.h);
    }


    //
    // Set up ONLY the matrices for world/screen space
    //
    void SetupMatricesForWorld()
    {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());
    }
    void SetupMatricesForScreen()
    {
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(-1,1, -1,1, -1,1);

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
    }


    //
    // Set up BOTH the matrices and viewport for world/screen space
    //
    void SetupForWorldSpace()
    {
        SetupMatricesForWorld();
        SetupViewportForWorld();
    }
    void SetupForScreenSpace()
    {
        SetupMatricesForScreen();
        SetupViewportForScreen();
    }

};

#endif

