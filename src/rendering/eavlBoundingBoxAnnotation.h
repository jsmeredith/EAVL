// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BOUNDING_BOX_ANNOTATION_H
#define EAVL_BOUNDING_BOX_ANNOTATION_H

#include "eavlAnnotation.h"

// ****************************************************************************
// Class:  eavlBoundingBoxAnnotation
//
// Purpose:
///   A 3D bounding box.
//
// Programmer:  Jeremy Meredith
// Creation:    January 11, 2013
//
// Modifications:
// ****************************************************************************
class eavlBoundingBoxAnnotation : public eavlAnnotation
{
  protected:
    double dmin[3], dmax[3];
    eavlColor color;
  public:
    eavlBoundingBoxAnnotation(eavlWindow *win) :
        eavlAnnotation(win)
    {
        dmin[0] = dmin[1] = dmin[2] = -1;
        dmax[0] = dmax[1] = dmax[2] = +1;
        color = eavlColor(.5,.5,.5);
    }
    virtual ~eavlBoundingBoxAnnotation()
    {
    }
    void SetExtents(double xmin, double xmax,
                    double ymin, double ymax,
                    double zmin, double zmax)
    {
        dmin[0] = xmin;
        dmax[0] = xmax;
        dmin[1] = ymin;
        dmax[1] = ymax;
        dmin[2] = zmin;
        dmax[2] = zmax;
    }
    void SetColor(eavlColor c)
    {
        color = c;
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForWorldSpace();

        float linewidth = 1.0;
        win->worldannotator->AddLine(dmin[0],dmin[1],dmin[2],
                                     dmin[0],dmin[1],dmax[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmin[0],dmax[1],dmin[2],
                                     dmin[0],dmax[1],dmax[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmax[0],dmin[1],dmin[2],
                                     dmax[0],dmin[1],dmax[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmax[0],dmax[1],dmin[2],
                                     dmax[0],dmax[1],dmax[2],
                                     linewidth, color);

        win->worldannotator->AddLine(dmin[0],dmin[1],dmin[2],
                                     dmin[0],dmax[1],dmin[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmin[0],dmin[1],dmax[2],
                                     dmin[0],dmax[1],dmax[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmax[0],dmin[1],dmin[2],
                                     dmax[0],dmax[1],dmin[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmax[0],dmin[1],dmax[2],
                                     dmax[0],dmax[1],dmax[2],
                                     linewidth, color);

        win->worldannotator->AddLine(dmin[0],dmin[1],dmin[2],
                                     dmax[0],dmin[1],dmin[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmin[0],dmin[1],dmax[2],
                                     dmax[0],dmin[1],dmax[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmin[0],dmax[1],dmin[2],
                                     dmax[0],dmax[1],dmin[2],
                                     linewidth, color);
        win->worldannotator->AddLine(dmin[0],dmax[1],dmax[2],
                                     dmax[0],dmax[1],dmax[2],
                                     linewidth, color);
    }    
};


#endif
