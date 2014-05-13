// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_WINDOW_H
#define EAVL_2D_WINDOW_H

#include "eavlWindow.h"
#include <eavlTextAnnotation.h>
#include <eavlColorBarAnnotation.h>
#include <eavl2DAxisAnnotation.h>
#include <eavl2DFrameAnnotation.h>
#include "eavlScene.h"

class eavl2DWindow : public eavlWindow
{
  protected:
    eavlColorBarAnnotation *colorbar;
    eavl2DAxisAnnotation *haxis, *vaxis;
    eavl2DFrameAnnotation *frame;
  public:
    eavl2DWindow(eavlColor bg, eavlRenderSurface *surf, eavlScene *s = NULL)
        : eavlWindow(bg,surf,s)
    {
        view.vl = -.7;
        view.vr = +.7;
        view.vb = -.7;
        view.vt = +.7;

        colorbar = new eavlColorBarAnnotation(this);
        haxis = new eavl2DAxisAnnotation(this);
        vaxis = new eavl2DAxisAnnotation(this);
        frame = new eavl2DFrameAnnotation(this);
    }
    ~eavl2DWindow()
    {
        delete colorbar;
        delete haxis;
        delete vaxis;
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
        frame->SetColor(eavlColor(.7,.7,.7));
        frame->Render(view);

        haxis->SetMoreOrLessTickAdjustment(fabs(view.viewportaspect) < .59 ? -1 : 0);
        vaxis->SetMoreOrLessTickAdjustment(fabs(view.viewportaspect) > 1.7 ? -1 : 0);

        haxis->SetColor(eavlColor::white);
        haxis->SetScreenPosition(vl,vb, vr,vb);
        haxis->SetRangeForAutoTicks(view.view2d.l, view.view2d.r);
        haxis->SetMajorTickSize(0, .05, 1.0);
        haxis->SetMinorTickSize(0, .02, 1.0);
        haxis->SetLabelAlignment(eavlTextAnnotation::HCenter,
                                 eavlTextAnnotation::Top);
        haxis->Render(view);

        vaxis->SetColor(eavlColor::white);
        vaxis->SetScreenPosition(vl,vb, vl,vt);
        vaxis->SetRangeForAutoTicks(view.view2d.b, view.view2d.t);
        vaxis->SetMajorTickSize(.05 / view.windowaspect, 0, 1.0);
        vaxis->SetMinorTickSize(.02 / view.windowaspect, 0, 1.0);
        vaxis->SetLabelAlignment(eavlTextAnnotation::Right,
                                 eavlTextAnnotation::VCenter);
        vaxis->Render(view);

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
