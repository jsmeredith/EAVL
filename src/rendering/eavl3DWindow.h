// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_3D_WINDOW_H
#define EAVL_3D_WINDOW_H

#include "eavlWindow.h"

#include "eavlTextAnnotation.h"
#include "eavlColorBarAnnotation.h"
#include "eavlBoundingBoxAnnotation.h"
#include "eavl3DAxisAnnotation.h"
#include "eavlScene.h"

class eavl3DWindow : public eavlWindow
{
  protected:
    eavlColorBarAnnotation *colorbar;
    eavlBoundingBoxAnnotation *bbox;
    eavl3DAxisAnnotation *xaxis, *yaxis, *zaxis;
  public:
    eavl3DWindow(eavlColor bg, eavlRenderSurface *surf, eavlScene *s = NULL)
        : eavlWindow(bg,surf,s)
    {
        colorbar = new eavlColorBarAnnotation(this);
        bbox = new eavlBoundingBoxAnnotation(this);
        xaxis = new eavl3DAxisAnnotation(this);
        yaxis = new eavl3DAxisAnnotation(this);
        zaxis = new eavl3DAxisAnnotation(this);
    }

    virtual void Render()
    {
        // render the plots
        scene->Render(this);

        // render the annotations
        bbox->SetColor(eavlColor(.3,.3,.3));
        bbox->SetExtents(view.minextents[0],
                         view.maxextents[0],
                         view.minextents[1],
                         view.maxextents[1],
                         view.minextents[2],
                         view.maxextents[2]);
        bbox->Render(view);

        double ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                               (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                               (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

        glDepthRange(-.0001,.9999);

        eavlVector3 viewdir = view.view3d.at - view.view3d.from;
        bool xtest = (viewdir * eavlVector3(1,0,0)) >= 0;
        bool ytest = (viewdir * eavlVector3(0,1,0)) >= 0;
        bool ztest = (viewdir * eavlVector3(0,0,1)) >= 0;

        bool outsideedges = true; // if false, do closesttriad
        if (outsideedges)
        {
            xtest = !xtest;
            ytest = !ytest;
        }

        xaxis->SetAxis(0);
        xaxis->SetColor(eavlColor::white);
        xaxis->SetTickInvert(xtest,ytest,ztest);
        xaxis->SetWorldPosition(view.minextents[0],
                                ytest ? view.minextents[1] : view.maxextents[1],
                                ztest ? view.minextents[2] : view.maxextents[2],
                                view.maxextents[0],
                                ytest ? view.minextents[1] : view.maxextents[1],
                                ztest ? view.minextents[2] : view.maxextents[2]);
        xaxis->SetRange(view.minextents[0], view.maxextents[0]);
        xaxis->SetMajorTickSize(ds_size / 40., 0);
        xaxis->SetMinorTickSize(ds_size / 80., 0);
        xaxis->SetLabelFontScale(ds_size / 30.);
        xaxis->Render(view);

        yaxis->SetAxis(1);
        yaxis->SetColor(eavlColor::white);
        yaxis->SetTickInvert(xtest,ytest,ztest);
        yaxis->SetWorldPosition(xtest ? view.minextents[0] : view.maxextents[0],
                                view.minextents[1],
                                ztest ? view.minextents[2] : view.maxextents[2],
                                xtest ? view.minextents[0] : view.maxextents[0],
                                view.maxextents[1],
                                ztest ? view.minextents[2] : view.maxextents[2]);
        yaxis->SetRange(view.minextents[1], view.maxextents[1]);
        yaxis->SetMajorTickSize(ds_size / 40., 0);
        yaxis->SetMinorTickSize(ds_size / 80., 0);
        yaxis->SetLabelFontScale(ds_size / 30.);
        yaxis->Render(view);

        if (outsideedges)
        {
            //xtest = !xtest;
            ytest = !ytest;
        }
        zaxis->SetAxis(2);
        zaxis->SetColor(eavlColor::white);
        zaxis->SetTickInvert(xtest,ytest,ztest);
        zaxis->SetWorldPosition(xtest ? view.minextents[0] : view.maxextents[0],
                                ytest ? view.minextents[1] : view.maxextents[1],
                                view.minextents[2],
                                xtest ? view.minextents[0] : view.maxextents[0],
                                ytest ? view.minextents[1] : view.maxextents[1],
                                view.maxextents[2]);
        zaxis->SetRange(view.minextents[2], view.maxextents[2]);
        zaxis->SetMajorTickSize(ds_size / 40., 0);
        zaxis->SetMinorTickSize(ds_size / 80., 0);
        zaxis->SetLabelFontScale(ds_size / 30.);
        zaxis->Render(view);

        glDepthRange(0,1);

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

    /*
    void ResetViewForCurrentExtents()
    {
    }
    */
};

#endif
