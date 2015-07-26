// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    eavl3DWindow(eavlColor bg, eavlRenderSurface *surf,
                 eavlScene *s, eavlSceneRenderer *r,
                 eavlWorldAnnotator *w)
        : eavlWindow(bg,surf,s,r,w)
    {
        //view.vt = +.8; // save room for legend (and prove 3D viewports work)

        colorbar = new eavlColorBarAnnotation(this);
        bbox = new eavlBoundingBoxAnnotation(this);
        xaxis = new eavl3DAxisAnnotation(this);
        yaxis = new eavl3DAxisAnnotation(this);
        zaxis = new eavl3DAxisAnnotation(this);
    }
    virtual ~eavl3DWindow()
    {
        delete colorbar;
        delete bbox;
        delete xaxis;
        delete yaxis;
        delete zaxis;
    }
    virtual void RenderScene()
    {
        scene->Render(this);
    }
    virtual void RenderAnnotations()
    {
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

        // note that this should be based on the apparent size if we ever
        // add 3D visual nonuniform scaling, not the size in world space.
        double xrel = fabs(view.maxextents[0]-view.minextents[0]) / ds_size;
        double yrel = fabs(view.maxextents[1]-view.minextents[1]) / ds_size;
        double zrel = fabs(view.maxextents[2]-view.minextents[2]) / ds_size;

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
        xaxis->SetMoreOrLessTickAdjustment(xrel < .3 ? -1 : 0);
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
        yaxis->SetMoreOrLessTickAdjustment(yrel < .3 ? -1 : 0);
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
        zaxis->SetMoreOrLessTickAdjustment(zrel < .3 ? -1 : 0);
        zaxis->Render(view);

        if (scene->plots.size() > 0)
        {
            double vmin = scene->plots[0]->GetMinDataExtent();
            double vmax = scene->plots[0]->GetMaxDataExtent();
            bool logscaling = scene->plots[0]->GetLogarithmicColorScaling();
            colorbar->SetAxisColor(eavlColor::white);
            colorbar->SetRange(vmin, vmax, 5, logscaling);
            colorbar->SetColorTable(scene->plots[0]->GetColorTable());
            colorbar->Render(view);
        }
    }

    /*
    void ResetViewForCurrentExtents()
    {
    }
    */
};
#ifdef HAVE_MPI
/*

	Parallel volume rendering is problamtic since
	all the frames are blended together. Someone is 
	logically in front of the other and the 
	boundign boxes are rendered on top of the 
	all other images. Additionally, only one MPI
	process should render the color bar.

*/
class eavl3DWindowParallelVolume : public eavlWindow
{
  protected:
    eavlColorBarAnnotation *colorbar;
    eavlBoundingBoxAnnotation *bbox;
    eavl3DAxisAnnotation *xaxis, *yaxis, *zaxis;
    bool drawAnnotations;
  public:
    eavl3DWindowParallelVolume(eavlColor bg, eavlRenderSurface *surf,
                 	   		   eavlScene *s, eavlSceneRenderer *r,
                	   		   eavlWorldAnnotator *w)
        : eavlWindow(bg,surf,s,r,w)
    {
        //view.vt = +.8; // save room for legend (and prove 3D viewports work)

        colorbar = new eavlColorBarAnnotation(this);
        bbox = new eavlBoundingBoxAnnotation(this);
        xaxis = new eavl3DAxisAnnotation(this);
        yaxis = new eavl3DAxisAnnotation(this);
        zaxis = new eavl3DAxisAnnotation(this);
        drawAnnotations = false;
    }
    virtual ~eavl3DWindowParallelVolume()
    {
        delete colorbar;
        delete bbox;
        delete xaxis;
        delete yaxis;
        delete zaxis;
    }
    virtual void RenderScene()
    {
        scene->Render(this);
    }
    
    void SetDrawAnnotations(bool on)
    {
    	drawAnnotations = on;
    }
    
    virtual void RenderAnnotations()
    {
		if(!drawAnnotations) return;
        double ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                               (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                               (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

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

        // note that this should be based on the apparent size if we ever
        // add 3D visual nonuniform scaling, not the size in world space.
        double xrel = fabs(view.maxextents[0]-view.minextents[0]) / ds_size;
        double yrel = fabs(view.maxextents[1]-view.minextents[1]) / ds_size;
        double zrel = fabs(view.maxextents[2]-view.minextents[2]) / ds_size;

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
        xaxis->SetMoreOrLessTickAdjustment(xrel < .3 ? -1 : 0);
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
        yaxis->SetMoreOrLessTickAdjustment(yrel < .3 ? -1 : 0);
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
        zaxis->SetMoreOrLessTickAdjustment(zrel < .3 ? -1 : 0);
        zaxis->Render(view);

        if (scene->plots.size() > 0)
        {
            double vmin = scene->plots[0]->GetMinDataExtent();
            double vmax = scene->plots[0]->GetMaxDataExtent();
            bool logscaling = scene->plots[0]->GetLogarithmicColorScaling();
            colorbar->SetAxisColor(eavlColor::white);
            colorbar->SetRange(vmin, vmax, 5, logscaling);
            colorbar->SetColorTable(scene->plots[0]->GetColorTable());
            colorbar->Render(view);
        }
    }

    /*
    void ResetViewForCurrentExtents()
    {
    }
    */
};
#endif

#endif
