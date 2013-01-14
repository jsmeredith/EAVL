// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ANNOTATION_H
#define EAVL_ANNOTATION_H

class eavlWindow;

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
class eavlAnnotation
{
  protected:
    eavlWindow *win;
  public:
    eavlAnnotation(eavlWindow *w)
        : win(w)
    {
    }
    virtual void Setup(eavlView &view) = 0;
    virtual void Render() = 0;
};

// ****************************************************************************
// Class:  eavlWorldSpaceAnnotation
//
// Purpose:
///   Base class for annotations in world space.
//
// Programmer:  Jeremy Meredith
// Creation:    January 11, 2013
//
// Modifications:
// ****************************************************************************
class eavlWorldSpaceAnnotation : public eavlAnnotation
{
  protected:
  public:
    eavlWorldSpaceAnnotation(eavlWindow *w)
        : eavlAnnotation(w)
    {
    }
    virtual void Setup(eavlView &view)
    {
        view.SetMatricesForViewport();

        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());
    }
};

// ****************************************************************************
// Class:  eavlScreenSpaceAnnotation
//
// Purpose:
///   Base class for annotations in screen space.
//
// Programmer:  Jeremy Meredith
// Creation:    January 11, 2013
//
// Modifications:
// ****************************************************************************
class eavlScreenSpaceAnnotation : public eavlAnnotation
{
  protected:
  public:
    eavlScreenSpaceAnnotation(eavlWindow *w)
        : eavlAnnotation(w)
    {
    }
    virtual void Setup(eavlView &view)
    {
        view.SetMatricesForScreen();

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(-1,1, -1,1, -1,1);

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
    }
};

#endif
