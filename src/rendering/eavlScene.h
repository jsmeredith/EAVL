// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_H
#define EAVL_SCENE_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"
#include "eavlPlot.h"
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
// ****************************************************************************
class eavlScene
{
  public:
    std::vector<eavlPlot> plots;
    eavlView &view;
    eavlWindow *win;

  public:
    eavlScene(eavlWindow *w, eavlView &v) : win(w), view(v) { }
    virtual void ResetView() = 0;
    virtual void Render(eavlView &view) = 0;

  protected:
    void SetViewExtentsFromPlots()
    {
        view.minextents[0] = view.minextents[1] = view.minextents[2] = FLT_MAX;
        view.maxextents[0] = view.maxextents[1] = view.maxextents[2] = -FLT_MAX;

        for (unsigned int i=0; i<plots.size(); i++)
        {
            eavlPlot &p = plots[i];
            if (!p.data)
                continue;

            int npts = p.data->GetNumPoints();
            int dim = p.data->GetCoordinateSystem(0)->GetDimension();

            //CHIMERA HACK
            if (dim > 3)
                dim = 3;
    
            for (int d=0; d<dim; d++)
            {
                for (int i=0; i<npts; i++)
                {
                    double v = p.data->GetPoint(i,d);
                    //cerr << "findspatialextents: d="<<d<<" i="<<i<<"  v="<<v<<endl;
                    if (v < view.minextents[d])
                        view.minextents[d] = v;
                    if (v > view.maxextents[d])
                        view.maxextents[d] = v;
                }
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
    eavl3DGLScene(eavlWindow *w, eavlView &v) : eavlScene(w, v)
    {
        colortexId = 0;
    }

    ///\todo: hack: this shouldn't be public, but I'm not sure it's even
    /// the right spot for it, so I'm working around it at the moment....
  protected:

    ///\todo: big hack for saved_colortable
    string saved_colortable;
    int colortexId;

    virtual void ResetView()
    {
        SetViewExtentsFromPlots();

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
    virtual void Render(eavlView &view)
    {
        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
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
            eavlPlot &p = plots[i];
            if (!p.data)
                continue;

            eavlTexture *tex = win->GetTexture(p.colortable);
            if (!tex)
            {
                if (!tex)
                    tex = new eavlTexture;
                tex->CreateFromColorTable(eavlColorTable(p.colortable));
                win->SetTexture(p.colortable, tex);
            }

            if (p.pcRenderer)
                tex->Enable();

            try
            {
                if (p.cellset_index < 0)
                {
                    if (p.pcRenderer)          p.pcRenderer->RenderPoints();
                    else if (p.meshRenderer)   p.meshRenderer->RenderPoints();
                }
                else
                {
                    eavlCellSet *cs = p.data->GetCellSet(p.cellset_index);
                    if (cs->GetDimensionality() == 1)
                    {
                        if (p.pcRenderer)          p.pcRenderer->RenderCells1D(cs);
                        else if (p.meshRenderer)   p.meshRenderer->RenderCells1D(cs);
                    }
                    else if (cs->GetDimensionality() == 2)
                    {
                        eavlField *normals = NULL;
                        // look for face-centered surface normals first
                        for (int i=0; i<p.data->GetNumFields(); i++)
                        {
                            if (p.data->GetField(i)->GetArray()->GetName() == "surface_normals" &&
                                p.data->GetField(i)->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                                p.data->GetField(i)->GetAssocCellSet() == p.cellset_index)
                            {
                                normals = p.data->GetField(i);
                            }
                        }
                        // override with node-centered ones if we have them
                        for (int i=0; i<p.data->GetNumFields(); i++)
                        {
                            if (p.data->GetField(i)->GetArray()->GetName() == "nodecentered_surface_normals" &&
                                p.data->GetField(i)->GetAssociation() == eavlField::ASSOC_POINTS)
                            {
                                normals = p.data->GetField(i);
                            }
                        }

                        if (p.pcRenderer)          p.pcRenderer->RenderCells2D(cs, normals);
                        else if (p.meshRenderer)   p.meshRenderer->RenderCells2D(cs, normals);
                    }
                }
            }
            catch (const eavlException &e)
            {
                // The user can specify one cell for geometry and
                // a different one for coloring; this currently results
                // in an error; we'll just ignore it.
                cerr << e.GetErrorText() << endl;
                cerr << "-\n";
            }

            if (p.pcRenderer)
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
    eavl2DGLScene(eavlWindow *w, eavlView &v) : eavlScene(w, v)
    {
        colortexId = 0;
    }

  protected:
    int width, height;

    ///\todo: big hack for saved_colortable
    string saved_colortable;
    int colortexId;
    virtual void ResetView()
    {
        SetViewExtentsFromPlots();

        eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2,
                                       (view.maxextents[1]+view.minextents[1]) / 2,
                                       (view.maxextents[2]+view.minextents[2]) / 2);

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
    virtual void Render(eavlView &view)
    {
        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
            return;

        view.SetupForWorldSpace();

        // render the plots
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            eavlPlot &p = plots[i];
            if (!p.data)
                continue;

            eavlTexture *tex = win->GetTexture(p.colortable);
            if (!tex)
            {
                if (!tex)
                    tex = new eavlTexture;
                tex->CreateFromColorTable(eavlColorTable(p.colortable));
                win->SetTexture(p.colortable, tex);
            }

            if (p.pcRenderer)
                tex->Enable();

            try
            {
                if (p.cellset_index < 0)
                {
                    if (p.pcRenderer)          p.pcRenderer->RenderPoints();
                    else if (p.meshRenderer)   p.meshRenderer->RenderPoints();
                }
                else
                {
                    eavlCellSet *cs = p.data->GetCellSet(p.cellset_index);
                    if (cs->GetDimensionality() == 1)
                    {
                        if (p.pcRenderer)          p.pcRenderer->RenderCells1D(cs);
                        else if (p.meshRenderer)   p.meshRenderer->RenderCells1D(cs);
                    }
                    else if (cs->GetDimensionality() == 2)
                    {
                        eavlField *normals = NULL;
                        // look for face-centered surface normals first
                        for (int i=0; i<p.data->GetNumFields(); i++)
                        {
                            if (p.data->GetField(i)->GetArray()->GetName() == "surface_normals" &&
                                p.data->GetField(i)->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                                p.data->GetField(i)->GetAssocCellSet() == p.cellset_index)
                            {
                                normals = p.data->GetField(i);
                            }
                        }
                        // override with node-centered ones if we have them
                        for (int i=0; i<p.data->GetNumFields(); i++)
                        {
                            if (p.data->GetField(i)->GetArray()->GetName() == "nodecentered_surface_normals" &&
                                p.data->GetField(i)->GetAssociation() == eavlField::ASSOC_POINTS)
                            {
                                normals = p.data->GetField(i);
                            }
                        }

                        if (p.pcRenderer)          p.pcRenderer->RenderCells2D(cs, normals);
                        else if (p.meshRenderer)   p.meshRenderer->RenderCells2D(cs, normals);
                    }
                }
            }
            catch (const eavlException &e)
            {
                // The user can specify one cell for geometry and
                // a different one for coloring; this currently results
                // in an error; we'll just ignore it.
                cerr << e.GetErrorText() << endl;
                cerr << "-\n";
            }

            if (p.pcRenderer)
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
    eavl1DGLScene(eavlWindow *w, eavlView &v) : eavlScene(w, v)
    {
        colortexId = 0;
    }

  protected:
    int width, height;

    ///\todo: big hack for saved_colortable
    string saved_colortable;
    int colortexId;
    virtual void ResetView()
    {
        SetViewExtentsFromPlots();

        eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2,
                                       (view.maxextents[1]+view.minextents[1]) / 2,
                                       (view.maxextents[2]+view.minextents[2]) / 2);

        view.viewtype = eavlView::EAVL_VIEW_2D;
        view.view2d.l = view.minextents[0];
        view.view2d.r = view.maxextents[0];
        // It's 1D; we'll use the field limits, but in case we don't
        // have a field, just set it to something reasonable.
        view.view2d.b = 0;
        view.view2d.t = 1;

        if (plots[0].curveRenderer)
        {
            double vmin, vmax;
            ((eavlCurveRenderer*)(plots[0].curveRenderer))->GetLimits(vmin, vmax);
            view.view2d.b = vmin;
            view.view2d.t = vmax;
        }
        else if (plots[0].barRenderer)
        {
            double vmin, vmax;
            ((eavlBarRenderer*)(plots[0].barRenderer))->GetLimits(vmin, vmax);
            view.view2d.b = vmin;
            view.view2d.t = vmax;
        }
        if (view.view2d.b == view.view2d.t)
        {
            view.view2d.b -= .5;
            view.view2d.t += .5;
        }

        // we always want to start with a curve being full-frame
        view.view2d.xscale = (float(view.w) / float(view.h)) *
                             (view.view2d.t-view.view2d.b) /
                             (view.view2d.r-view.view2d.l);

    }
    virtual void Render(eavlView &view)
    {
        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
            return;

        view.SetupForWorldSpace();

        // render the plots
        for (unsigned int i=0;  i<plots.size(); i++)
        {
            eavlPlot &p = plots[i];
            if (!p.data)
                continue;

            try
            {
                if (p.cellset_index < 0)
                {
                    if (p.curveRenderer) p.curveRenderer->RenderPoints();
                    else if (p.barRenderer) p.barRenderer->RenderPoints();
                }
                else
                {
                    eavlCellSet *cs = p.data->GetCellSet(p.cellset_index);
                    if (cs->GetDimensionality() == 1)
                    {
                        if (p.curveRenderer) p.curveRenderer->RenderCells1D(cs);
                        else if (p.barRenderer) p.barRenderer->RenderCells1D(cs);
                    }
                }
            }
            catch (const eavlException &e)
            {
                // The user can specify one cell for geometry and
                // a different one for coloring; this currently results
                // in an error; we'll just ignore it.
                cerr << e.GetErrorText() << endl;
                cerr << "-\n";
            }
        }
    }
};

#ifdef HAVE_MPI
#include <boost/mpi.hpp>

class eavl3DParallelGLScene : public eavl3DGLScene
{
  protected:
    boost::mpi::communicator &comm;
  public:
    eavl3DParallelGLScene(boost::mpi::communicator &c,
                          eavlWindow *w, eavlView &v) :
        eavl3DGLScene(w,v), comm(c)
    {
    }
    virtual void ResetView()
    {
        eavl3DGLScene::ResetView();

        boost::mpi::all_reduce(comm, view.minextents[0], view.minextents[0], boost::mpi::minimum<float>());
        boost::mpi::all_reduce(comm, view.minextents[1], view.minextents[1], boost::mpi::minimum<float>());
        boost::mpi::all_reduce(comm, view.minextents[2], view.minextents[2], boost::mpi::minimum<float>());

        boost::mpi::all_reduce(comm, view.maxextents[0], view.maxextents[0], boost::mpi::maximum<float>());
        boost::mpi::all_reduce(comm, view.maxextents[1], view.maxextents[1], boost::mpi::maximum<float>());
        boost::mpi::all_reduce(comm, view.maxextents[2], view.maxextents[2], boost::mpi::maximum<float>());

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
