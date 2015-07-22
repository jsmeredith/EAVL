// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_H
#define EAVL_SCENE_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlPlot.h"
#include "eavlColorTable.h"

class eavlWindow;

// ****************************************************************************
// Class:  eavlScene
//
// Purpose:
///   Encapsulate a type of output scene (e.g. 3D), i.e. a set of plots.
//
// Programmer:  Jeremy Meredith
// Creation:    December 27, 2012
//
// Modifications:
//   Jeremy Meredith, Mon Mar  4 15:44:23 EST 2013
//   Big refactoring; more consistent internal code with less
//   duplication and cleaner external API.
//
// ****************************************************************************
class eavlScene
{
  public:
    std::vector<eavlPlot*> plots;

  public:
    eavlScene() { }
    ~eavlScene()
    {
        for (unsigned int i=0; i<plots.size(); i++)
        {
            delete plots[i];
        }
        plots.clear();
    }
    virtual void ResetView(eavlWindow *win) = 0;
    virtual void Render(eavlWindow *win) = 0;

  protected:
    void SetViewExtentsFromPlots(eavlView &view)
    {
        view.minextents[0] = view.minextents[1] = view.minextents[2] = FLT_MAX;
        view.maxextents[0] = view.maxextents[1] = view.maxextents[2] = -FLT_MAX;

        for (unsigned int i=0; i<plots.size(); i++)
        {
            eavlPlot *p = plots[i];
            if (!p)
                continue;

            for (int d=0; d<3; d++)
            {
                double vmin = p->GetMinCoordExtentOrig(d);
                if (vmin < view.minextents[d])
                    view.minextents[d] = vmin;
                double vmax = p->GetMaxCoordExtentOrig(d);
                if (vmax > view.maxextents[d])
                    view.maxextents[d] = vmax;
            }
        }

        // untouched dims force to zero
        if (view.minextents[0] > view.maxextents[0])
            view.minextents[0] = view.maxextents[0] = 0;
        if (view.minextents[1] > view.maxextents[1])
            view.minextents[1] = view.maxextents[1] = 0;
        if (view.minextents[2] > view.maxextents[2])
            view.minextents[2] = view.maxextents[2] = 0;

        //cerr << "extents: "
        //     << view.minextents[0]<<":"<<view.maxextents[0]<<"  "
        //     << view.minextents[1]<<":"<<view.maxextents[1]<<"  "
        //     << view.minextents[2]<<":"<<view.maxextents[2]<<"\n";
        view.size = sqrt(pow(view.maxextents[0]-view.minextents[0],2.) +
                         pow(view.maxextents[1]-view.minextents[1],2.) +
                         pow(view.maxextents[2]-view.minextents[2],2.));
                     
    }
};

#include "eavlWindow.h"


// ****************************************************************************
// Class:  eavl3DScene
//
// Purpose:
///   A 3D output scene.
//
// Programmer:  Jeremy Meredith
// Creation:    December 27, 2012
//
// Modifications:
// ****************************************************************************
class eavl3DScene : public eavlScene
{
  public:
    eavl3DScene() : eavlScene()
    {
    }
    virtual ~eavl3DScene()
    {
    }

  protected:
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        SetViewExtentsFromPlots(view);

        float ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                              (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                              (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

        eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2,
                                       (view.maxextents[1]+view.minextents[1]) / 2,
                                       (view.maxextents[2]+view.minextents[2]) / 2);

        view.viewtype = eavlView::EAVL_VIEW_3D;
        view.view3d.perspective = true;
        view.view3d.xpan = 0;
        view.view3d.ypan = 0;
        view.view3d.zoom = 1.0;
        view.view3d.at   = center;
#ifdef LEFTHANDED
        view.view3d.from = view.view3d.at + eavlVector3(0,0, -ds_size*2);
#else
        view.view3d.from = view.view3d.at + eavlVector3(0,0, ds_size*2);
#endif
        view.view3d.up   = eavlVector3(0,1,0);
        view.view3d.fov  = 0.5; // perspective only
        view.view3d.size = ds_size; //orthographic only;
        
        view.view3d.nearplane = ds_size/16.;
        view.view3d.farplane = ds_size*4;
		view.SetupMatrices();
		//squeezing the near and far planes for volume renderer 
		eavlPoint3 mins(view.minextents[0],view.minextents[1],view.minextents[2]);
		eavlPoint3 maxs(view.maxextents[0],view.maxextents[1],view.maxextents[2]);
		mins = view.V * mins;
		maxs = view.V * maxs;
		float far = std::max(-mins.z, -maxs.z);
		float near = std::min(-mins.z, -maxs.z);
		view.view3d.nearplane = near * 0.9f; 
        view.view3d.farplane =  far * 1.1f; 
        view.SetupMatrices();   
		
    }
    virtual void Render(eavlWindow *win)
    {
        eavlView &view = win->view;
        if (plots.size() == 0)
            return;

        win->SetupForWorldSpace();

        eavlSceneRenderer *sr = win->GetSceneRenderer();
        sr->SetView(view);

        // render the plots
        bool needs_update = false;
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            if (plots[i] && sr->NeedsGeometryForPlot(plots[i]->GetID()))
                needs_update = true;
        }

        if (needs_update)
        {
            sr->StartScene();
            for (unsigned int i=0;  i<plots.size(); i++)
            {
                eavlPlot *p = plots[i];
                if (!p)
                    continue;
                sr->SendingGeometryForPlot(p->GetID());
                p->Generate(sr);
            }
            sr->EndScene();
        }

        sr->Render();
    }
};



// ****************************************************************************
// Class:  eavl2DScene
//
// Purpose:
///   A 2D output scene.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavl2DScene : public eavlScene
{
  public:
    eavl2DScene() : eavlScene()
    {
    }
    virtual ~eavl2DScene()
    {
    }

  protected:
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        SetViewExtentsFromPlots(view);

        view.viewtype = eavlView::EAVL_VIEW_2D;
        view.view2d.l = view.minextents[0];
        view.view2d.r = view.maxextents[0];
        view.view2d.b = view.minextents[1];
        view.view2d.t = view.maxextents[1];
        if (view.view2d.b == view.view2d.t)
        {
            view.view2d.b -= .5;
            view.view2d.t += .5;
        }
    }
    virtual void Render(eavlWindow *win)
    {
        eavlView &view = win->view;
        if (plots.size() == 0)
            return;

        win->SetupForWorldSpace();

        eavlSceneRenderer *sr = win->GetSceneRenderer();
        sr->SetView(view);

        ///\todo: the tail of the 1D/2D/3D Render() methods are currently
        /// identical.  Can we merge them?  (If the renderers had
        /// access to the window, or texture cache if it gets moved
        /// out of the window, then we just move the texture mgt into
        /// eavlPlot base, and that makes this code a one-line loop.

        // render the plots
        bool needs_update = false;
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            if (plots[i] && sr->NeedsGeometryForPlot(plots[i]->GetID()))
                needs_update = true;
        }

        if (needs_update)
        {
            sr->StartScene();
            for (unsigned int i=0;  i<plots.size(); i++)
            {
                eavlPlot *p = plots[i];
                if (!p)
                    continue;
                sr->SendingGeometryForPlot(p->GetID());
                p->Generate(sr);
            }
            sr->EndScene();
        }

        sr->Render();
    }
};


class eavlPolarScene : public eavlScene
{
  public:
    eavlPolarScene() : eavlScene()
    {
    }
    virtual ~eavlPolarScene()
    {
    }

  protected:
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        SetViewExtentsFromPlots(view);

        view.viewtype = eavlView::EAVL_VIEW_2D;
        // the *1.3 is because we're currently drawing the axis in world
        // space, so we need to leave room inside the viewport for
        // the labels around the outside of the outer axis
        view.view2d.l = -view.maxextents[0] * 1.3;
        view.view2d.r = +view.maxextents[0] * 1.3;
        view.view2d.b = -view.maxextents[0] * 1.3;
        view.view2d.t = +view.maxextents[0] * 1.3;
        if (view.view2d.b == view.view2d.t)
        {
            view.view2d.b -= .5;
            view.view2d.t += .5;
        }
    }
    virtual void Render(eavlWindow *win)
    {
        eavlView &view = win->view;
        if (plots.size() == 0)
            return;

        win->SetupForWorldSpace();

        eavlSceneRenderer *sr = win->GetSceneRenderer();
        sr->SetView(view);

        ///\todo: the tail of the 1D/2D/3D Render() methods are currently
        /// identical.  Can we merge them?  (If the renderers had
        /// access to the window, or texture cache if it gets moved
        /// out of the window, then we just move the texture mgt into
        /// eavlPlot base, and that makes this code a one-line loop.

        // render the plots
        bool needs_update = false;
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            if (plots[i] && sr->NeedsGeometryForPlot(plots[i]->GetID()))
                needs_update = true;
        }

        if (needs_update)
        {
            sr->StartScene();
            for (unsigned int i=0;  i<plots.size(); i++)
            {
                eavlPlot *p = plots[i];
                if (!p)
                    continue;
                sr->SendingGeometryForPlot(p->GetID());
                p->Generate(sr);
            }
            sr->EndScene();
        }

        sr->Render();
    }
};


// ****************************************************************************
// Class:  eavl1DScene
//
// Purpose:
///   A 1D output scene.
//
// Programmer:  Jeremy Meredith
// Creation:    January 16, 2013
//
// Modifications:
// ****************************************************************************
class eavl1DScene : public eavlScene
{
  public:
    eavl1DScene() : eavlScene()
    {
    }
    virtual ~eavl1DScene()
    {
    }

  protected:
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        SetViewExtentsFromPlots(view);

        view.viewtype = eavlView::EAVL_VIEW_2D;
        view.view2d.l = view.minextents[0];
        view.view2d.r = view.maxextents[0];

        double vmin = 0;
        double vmax = 1;
        if (plots.size() > 0 && plots[0])
        {
            vmin = plots[0]->GetMinDataExtent();
            vmax = plots[0]->GetMaxDataExtent();
        }
        for (unsigned int i=1; i<plots.size(); ++i)
        {
            if (vmin > plots[i]->GetMinDataExtent())
                vmin = plots[i]->GetMinDataExtent();
            if (vmax < plots[i]->GetMaxDataExtent())
                vmax = plots[i]->GetMaxDataExtent();
        }

        if (view.view2d.logy)
        {
            if (vmin <= 0 || vmax <= 0)
            {
                view.view2d.b = 0;
                view.view2d.t = 1;
            }
            else
            {
                view.view2d.b = log10(vmin);
                view.view2d.t = log10(vmax);
                if (view.view2d.b == view.view2d.t)
                {
                    view.view2d.b /= 10.;
                    view.view2d.t *= 10.;
                }
            }
        }
        else
        {
            view.view2d.b = vmin;
            view.view2d.t = vmax;
            if (view.view2d.b == view.view2d.t)
            {
                view.view2d.b -= .5;
                view.view2d.t += .5;
            }
        }

        // we always want to start with a curve being full-frame
        view.view2d.xscale = (float(view.w) / float(view.h)) *
                             (view.view2d.t-view.view2d.b) /
                             (view.view2d.r-view.view2d.l);

    }
    virtual void Render(eavlWindow *win)
    {
        eavlView &view = win->view;
        if (plots.size() == 0)
            return;

        win->SetupForWorldSpace();

        eavlSceneRenderer *sr = win->GetSceneRenderer();
        sr->SetView(view);

        // render the plots
        bool needs_update = false;
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            if (plots[i] && sr->NeedsGeometryForPlot(plots[i]->GetID()))
                needs_update = true;
        }

        if (needs_update)
        {
            sr->StartScene();
            for (unsigned int i=0;  i<plots.size(); i++)
            {
                eavlPlot *p = plots[i];
                if (!p)
                    continue;

                ///\todo: ugly hack to make the curve renderer do log scaling
                eavl1DPlot *p1d = dynamic_cast<eavl1DPlot*>(p);
                if (p1d)
                {
                    p1d->SetLogarithmic(view.view2d.logy);
                }

                sr->SendingGeometryForPlot(p->GetID());
                p->Generate(sr);
            }
            sr->EndScene();
        }

        sr->Render();
    }
};

#ifdef HAVE_MPI
#include <mpi.h>
class eavl2DParallelScene : public eavl2DScene
{
  protected:
    MPI_Comm comm;
  public:
    eavl2DParallelScene(const MPI_Comm &c) :
        eavl2DScene(), comm(c)
    {
    }
    virtual ~eavl2DParallelScene()
    {
    }
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        eavl2DScene::ResetView(win);

        double tmp;
        MPI_Allreduce(&view.minextents[0], &tmp, 1, MPI_DOUBLE, MPI_MIN, comm);
        view.minextents[0] = tmp;
        MPI_Allreduce(&view.minextents[1], &tmp, 1, MPI_DOUBLE, MPI_MIN, comm);
        view.minextents[1] = tmp;
        MPI_Allreduce(&view.minextents[2], &tmp, 1, MPI_DOUBLE, MPI_MIN, comm);
        view.minextents[2] = tmp;

        MPI_Allreduce(&view.maxextents[0], &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
        view.maxextents[0] = tmp;
        MPI_Allreduce(&view.maxextents[1], &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
        view.maxextents[1] = tmp;
        MPI_Allreduce(&view.maxextents[2], &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
        view.maxextents[2] = tmp;

        float ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                              (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                              (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

        view.viewtype = eavlView::EAVL_VIEW_2D;
        view.view2d.l = view.minextents[0];
        view.view2d.r = view.maxextents[0];
        view.view2d.b = view.minextents[1];
        view.view2d.t = view.maxextents[1];
        if (view.view2d.b == view.view2d.t)
        {
            view.view2d.b -= .5;
            view.view2d.t += .5;
        }
    }
};


class eavl3DParallelScene : public eavl3DScene
{
  protected:
    MPI_Comm comm;
  public:
    eavl3DParallelScene(const MPI_Comm &c) :
        eavl3DScene(), comm(c)
    {
    }
    virtual ~eavl3DParallelScene()
    {
    }
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        eavl3DScene::ResetView(win);

        double tmp;
        MPI_Allreduce(&view.minextents[0], &tmp, 1, MPI_DOUBLE, MPI_MIN, comm);
        view.minextents[0] = tmp;
        MPI_Allreduce(&view.minextents[1], &tmp, 1, MPI_DOUBLE, MPI_MIN, comm);
        view.minextents[1] = tmp;
        MPI_Allreduce(&view.minextents[2], &tmp, 1, MPI_DOUBLE, MPI_MIN, comm);
        view.minextents[2] = tmp;

        MPI_Allreduce(&view.maxextents[0], &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
        view.maxextents[0] = tmp;
        MPI_Allreduce(&view.maxextents[1], &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
        view.maxextents[1] = tmp;
        MPI_Allreduce(&view.maxextents[2], &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
        view.maxextents[2] = tmp;

        float ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                              (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                              (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

        eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2,
                                       (view.maxextents[1]+view.minextents[1]) / 2,
                                       (view.maxextents[2]+view.minextents[2]) / 2);

        view.view3d.at   = center;
        view.view3d.from = view.view3d.at + eavlVector3(0,0, -ds_size*2);
        view.view3d.up   = eavlVector3(0,1,0);
        view.view3d.fov  = 0.5;
        view.view3d.nearplane = ds_size/16.;
        view.view3d.farplane = ds_size*4;
        view.SetupMatrices();
		//squeezing the near and far planes for volume renderer 
		eavlPoint3 mins(view.minextents[0],view.minextents[1],view.minextents[2]);
		eavlPoint3 maxs(view.maxextents[0],view.maxextents[1],view.maxextents[2]);
		mins = view.V * mins;
		maxs = view.V * maxs;
		float far = std::max(-mins.z, -maxs.z);
		float near = std::min(-mins.z, -maxs.z);
		view.view3d.nearplane = near * 0.9f; 
        view.view3d.farplane =  far * 1.1f; 
        view.SetupMatrices();   
   		 
    }
};




#endif

#endif
