// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ANNOTATION_H
#define EAVL_ANNOTATION_H

class eavlWindow;
#include "eavlView.h"

// ****************************************************************************
// Class:  eavlAnnotation
//
// Purpose:
///   Base class for all GL annotations.
//
// Programmer:  Jeremy Meredith
// Creation:    January 11, 2013
//
// Modifications:
// ****************************************************************************

///\todo: Rename to "eavlGLDrawable" or something like that....
/// Also implies the various annotations should be renamed to include "GL"?
class eavlAnnotation
{
  protected:
    eavlWindow *win;
  public:
    eavlAnnotation(eavlWindow *w)
        : win(w)
    {
    }


    virtual void Render(eavlView &view) = 0;

  protected:
    //
    // Set up ONLY the viewport for world/screen space
    //
    void SetupViewportForWorld(eavlView &view)
    {
        float vl, vr, vt, vb;
        view.GetRealViewport(vl,vr,vb,vt);
        glViewport(float(view.w)*(1.+vl)/2.,
                   float(view.h)*(1.+vb)/2.,
                   float(view.w)*(vr-vl)/2.,
                   float(view.h)*(vt-vb)/2.);
    }
    void SetupViewportForScreen(eavlView &view)
    {
        glViewport(0, 0, view.w, view.h);
    }


    //
    // Set up ONLY the matrices for world/screen space
    //
    void SetupMatricesForWorld(eavlView &view)
    {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());
    }
    void SetupMatricesForScreen(eavlView &view)
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
    void SetupForWorldSpace(eavlView &view)
    {
        SetupMatricesForWorld(view);
        SetupViewportForWorld(view);
    }
    void SetupForScreenSpace(eavlView &view)
    {
        SetupMatricesForScreen(view);
        SetupViewportForScreen(view);
    }
};

#endif
