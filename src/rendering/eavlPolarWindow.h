// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_POLAR_WINDOW_H
#define EAVL_POLAR_WINDOW_H

#include "eavlWindow.h"
#include <eavlTextAnnotation.h>
#include <eavlColorBarAnnotation.h>
#include <eavl2DAxisAnnotation.h>
#include <eavl2DRadialAxisAnnotation.h>
#include <eavl2DFrameAnnotation.h>
#include "eavlScene.h"

class eavlPolarWindow : public eavlWindow
{
  protected:
    eavlColorBarAnnotation *colorbar;
    eavl2DRadialAxisAnnotation *aaxis;
    eavl2DAxisAnnotation *raxis;
    eavl2DFrameAnnotation *frame;
  public:
    eavlPolarWindow(eavlColor bg, eavlRenderSurface *surf, eavlScene *s = NULL)
        : eavlWindow(bg,surf,s)
    {
        /*
        view.vl = -.7;
        view.vr = +.7;
        view.vb = -.7;
        view.vt = +.7;
        */

        colorbar = new eavlColorBarAnnotation(this);
        aaxis = new eavl2DRadialAxisAnnotation(this);
        raxis = new eavl2DAxisAnnotation(this);
        frame = new eavl2DFrameAnnotation(this);
    }
    ~eavlPolarWindow()
    {
        delete colorbar;
        delete aaxis;
        delete raxis;
        delete frame;
    }
    virtual void Render()
    {
        glDisable(GL_DEPTH_TEST);

        // render the plots
        scene->Render(this);

        // render the annotations
        glDisable(GL_DEPTH_TEST);

        double vl, vr, vt, vb;
        view.GetRealViewport(vl,vr,vb,vt);
        frame->SetExtents(vl,vr, vb,vt);
        frame->SetColor(eavlColor(.4,.4,.4));
        //frame->Render(view);

        double radius = view.maxextents[0];

        raxis->SetColor(eavlColor::white);
        raxis->SetWorldSpace(true);
        raxis->SetScreenPosition(0,0, radius,0);
        raxis->SetRangeForAutoTicks(0, radius);
        raxis->SetMajorTickSize(0, radius*.05, 1.0);
        raxis->SetMinorTickSize(0, radius*.02, 1.0);
        raxis->SetLabelAlignment(eavlTextAnnotation::HCenter,
                                 eavlTextAnnotation::Top);
        raxis->Render(view);

        aaxis->SetColor(eavlColor::white);
        aaxis->SetScreenPosition(0,0, radius);
        aaxis->SetRangeForAutoTicks(view.minextents[1], view.maxextents[1]);
        aaxis->SetMajorTickSize(.05, 0.5);
        aaxis->SetMinorTickSize(.02, 0.5);
        aaxis->Render(view);

        if (scene->plots.size() > 0)
        {
            double vmin = scene->plots[0]->GetMinDataExtent();
            double vmax = scene->plots[0]->GetMaxDataExtent();
            colorbar->SetAxisColor(eavlColor::white);
            colorbar->SetRange(vmin, vmax, 5);
            colorbar->SetColorTable(scene->plots[0]->GetColorTableName());
            colorbar->Render(view);
        }

        glFinish();
    }
};

#endif
