// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_H
#define EAVL_SCENE_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"
#include "eavlTexture.h"
#include "eavlWindow.h"

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
    std::vector<eavlRenderer*> plots;

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
            eavlRenderer *p = plots[i];
            if (!p)
                continue;

            for (int d=0; d<3; d++)
            {
                double vmin = p->GetMinCoordExtent(d);
                if (vmin < view.minextents[d])
                    view.minextents[d] = vmin;
                double vmax = p->GetMaxCoordExtent(d);
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
    }
};

// ****************************************************************************
// Class:  eavl3DGLScene
//
// Purpose:
///   A 3D output scene with OpenGL/MesaGL rendering.
//
// Programmer:  Jeremy Meredith
// Creation:    December 27, 2012
//
// Modifications:
// ****************************************************************************
class eavl3DGLScene : public eavlScene
{
  public:
    eavl3DGLScene() : eavlScene()
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
        view.view3d.from = view.view3d.at + eavlVector3(0,0, -ds_size*2);
        view.view3d.up   = eavlVector3(0,1,0);
        view.view3d.fov  = 0.5;
        view.view3d.nearplane = ds_size/16.;
        view.view3d.farplane = ds_size*4;

    }
    virtual void Render(eavlWindow *win)
    {
        eavlView &view = win->view;
        if (plots.size() == 0)
            return;

        view.SetupForWorldSpace();

        // We need to set lighting without the view matrix (for an eye
        // light) so load the identity matrix into modelview temporarily....
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        // lighting
        bool lighting = true;
        if (lighting)
        {
            bool twoSidedLighting = true;
            glShadeModel(GL_SMOOTH);
            glEnable(GL_LIGHTING);
            glEnable(GL_COLOR_MATERIAL);
            glEnable(GL_LIGHT0);
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, twoSidedLighting?1:0);
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, eavlColor::grey20.c);
            glLightfv(GL_LIGHT0, GL_AMBIENT, eavlColor::black.c);
            glLightfv(GL_LIGHT0, GL_DIFFUSE, eavlColor::grey50.c);
            float lightdir[4] = {0, 0, 1, 0};
            glLightfv(GL_LIGHT0, GL_POSITION, lightdir);
            glLightfv(GL_LIGHT0, GL_SPECULAR, eavlColor::white.c);
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, eavlColor::grey40.c);
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 8.0f);
        }

        // Okay, safe to set the view matrix back now that lighting's done.
        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());

        glEnable(GL_DEPTH_TEST);

        // render the plots
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            eavlRenderer *r = plots[i];
            if (!r)
                continue;

            eavlTexture *tex = NULL;
            if (r->GetColorTableName() != "")
            {
                tex = win->GetTexture(r->GetColorTableName());
                if (!tex)
                {
                    if (!tex)
                        tex = new eavlTexture;
                    tex->CreateFromColorTable(eavlColorTable(r->GetColorTableName()));
                    win->SetTexture(r->GetColorTableName(), tex);
                }
            }

            if (tex)
                tex->Enable();

            r->Render();

            if (tex)
                tex->Disable();
        }
    }
};



// ****************************************************************************
// Class:  eavl2DGLScene
//
// Purpose:
///   A 2D output scene with OpenGL/MesaGL rendering.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavl2DGLScene : public eavlScene
{
  public:
    eavl2DGLScene() : eavlScene()
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

        view.SetupForWorldSpace();

        ///\todo: the tail of the 1D/2D/3D Render() methods are currently
        /// identical.  Can we merge them?  (If the renderers had
        /// access to the window, or texture cache if it gets moved
        /// out of the window, then we just move the texture mgt into
        /// eavlRenderer base, and that makes this code a one-line loop.

        // render the plots
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            eavlRenderer *r = plots[i];
            if (!r)
                continue;

            eavlTexture *tex = NULL;
            if (r->GetColorTableName() != "")
            {
                tex = win->GetTexture(r->GetColorTableName());
                if (!tex)
                {
                    if (!tex)
                        tex = new eavlTexture;
                    tex->CreateFromColorTable(eavlColorTable(r->GetColorTableName()));
                    win->SetTexture(r->GetColorTableName(), tex);
                }
            }

            if (tex)
                tex->Enable();

            r->Render();

            if (tex)
                tex->Disable();
        }
    }
};


class eavlPolarGLScene : public eavlScene
{
  public:
    eavlPolarGLScene() : eavlScene()
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

        view.SetupForWorldSpace();

        ///\todo: the tail of the 1D/2D/3D Render() methods are currently
        /// identical.  Can we merge them?  (If the renderers had
        /// access to the window, or texture cache if it gets moved
        /// out of the window, then we just move the texture mgt into
        /// eavlRenderer base, and that makes this code a one-line loop.

        // render the plots
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            eavlRenderer *r = plots[i];
            if (!r)
                continue;

            eavlTexture *tex = NULL;
            if (r->GetColorTableName() != "")
            {
                tex = win->GetTexture(r->GetColorTableName());
                if (!tex)
                {
                    if (!tex)
                        tex = new eavlTexture;
                    tex->CreateFromColorTable(eavlColorTable(r->GetColorTableName()));
                    win->SetTexture(r->GetColorTableName(), tex);
                }
            }

            if (tex)
                tex->Enable();

            r->Render();

            if (tex)
                tex->Disable();
        }
    }
};


// ****************************************************************************
// Class:  eavl1DGLScene
//
// Purpose:
///   A 1D output scene with OpenGL/MesaGL rendering.
//
// Programmer:  Jeremy Meredith
// Creation:    January 16, 2013
//
// Modifications:
// ****************************************************************************
class eavl1DGLScene : public eavlScene
{
  public:
    eavl1DGLScene() : eavlScene()
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

        view.SetupForWorldSpace();

        // render the plots
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            eavlRenderer *r = plots[i];
            if (!r)
                continue;

            ///\todo: ugly hack to make the curve renderer do log scaling
            eavlCurveRenderer* cr = dynamic_cast<eavlCurveRenderer*>(r);
            if (cr)
            {
                cr->SetLogarithmic(view.view2d.logy);
            }

            eavlTexture *tex = NULL;
            if (r->GetColorTableName() != "")
            {
                tex = win->GetTexture(r->GetColorTableName());
                if (!tex)
                {
                    if (!tex)
                        tex = new eavlTexture;
                    tex->CreateFromColorTable(eavlColorTable(r->GetColorTableName()));
                    win->SetTexture(r->GetColorTableName(), tex);
                }
            }

            if (tex)
                tex->Enable();

            r->Render();

            if (tex)
                tex->Disable();
        }
    }
};

#ifdef HAVE_MPI

class eavl2DParallelGLScene : public eavl2DGLScene
{
  protected:
    MPI_Comm comm;
  public:
    eavl2DParallelGLScene(const MPI_Comm &c) :
        eavl2DGLScene(), comm(c)
    {
    }
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        eavl2DGLScene::ResetView(win);

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


class eavl3DParallelGLScene : public eavl3DGLScene
{
  protected:
    MPI_Comm comm;
  public:
    eavl3DParallelGLScene(const MPI_Comm &c) :
        eavl3DGLScene(), comm(c)
    {
    }
    virtual void ResetView(eavlWindow *win)
    {
        eavlView &view = win->view;
        eavl3DGLScene::ResetView(win);

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
    }
};




#endif

#endif
