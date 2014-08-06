// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_FRAME_ANNOTATION_H
#define EAVL_2D_FRAME_ANNOTATION_H

// ****************************************************************************
// Class:  eavl2DFrameAnnotation
//
// Purpose:
///   A 2D frame.
//
// Programmer:  Jeremy Meredith
// Creation:    January 16, 2013
//
// Modifications:
// ****************************************************************************
class eavl2DFrameAnnotation : public eavlAnnotation
{
  protected:
    double dmin[2], dmax[2];
    eavlColor color;
  public:
    eavl2DFrameAnnotation(eavlWindow *win) : eavlAnnotation(win)
    {
        dmin[0] = dmin[1] = -1;
        dmax[0] = dmax[1] = +1;
        color = eavlColor(.5,.5,.5);
    }
    virtual ~eavl2DFrameAnnotation()
    {
    }
    void SetExtents(double xmin, double xmax,
                    double ymin, double ymax)
    {
        dmin[0] = xmin;
        dmax[0] = xmax;
        dmin[1] = ymin;
        dmax[1] = ymax;
    }
    void SetColor(eavlColor c)
    {
        color = c;
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForScreenSpace();

        win->surface->AddLine(dmin[0],dmin[1],
                              dmin[0],dmax[1],
                              1.0, color);

        win->surface->AddLine(dmax[0],dmin[1],
                              dmax[0],dmax[1],
                              1.0, color);

        win->surface->AddLine(dmin[0],dmin[1],
                              dmax[0],dmin[1],
                              1.0, color);

        win->surface->AddLine(dmin[0],dmax[1],
                              dmax[0],dmax[1],
                              1.0, color);
    }    
};


#endif
