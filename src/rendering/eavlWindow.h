// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"
#include "eavlPlot.h"

class eavlTexture;

// ****************************************************************************
// Class:  eavlWindow
//
// Purpose:
///   Encapsulate a type of output window (e.g. 3D), plots and annotations.
//
// Programmer:  Jeremy Meredith
// Creation:    December 27, 2012
//
// Modifications:
// ****************************************************************************
class eavlWindow
{
  public:
    std::vector<eavlPlot> plots;
    eavlView &view;
    std::map<std::string,eavlTexture*> textures;

  public:
    eavlWindow(eavlView &v) : view(v) { }
    virtual void ResetView() = 0;
    virtual void Initialize() = 0;
    virtual void Resize(int w, int h) = 0;
    virtual void Paint() = 0;
    eavlTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlTexture *tex)
    {
        textures[s] = tex;
    }
};

// ****************************************************************************
// Class:  eavl3DGLWindow
//
// Purpose:
///   A 3D output window with OpenGL/MesaGL rendering.
//
// Programmer:  Jeremy Meredith
// Creation:    December 27, 2012
//
// Modifications:
// ****************************************************************************
class eavl3DGLWindow : public eavlWindow
{
  public:
    eavl3DGLWindow(eavlView &v) : eavlWindow(v)
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
    
            if (dim < 2 || dim > 3)
                THROW(eavlException,"only supports 2 or 3 dimensions for now");
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

        float ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                              (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                              (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

        eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2,
                                       (view.maxextents[1]+view.minextents[1]) / 2,
                                       (view.maxextents[2]+view.minextents[2]) / 2);

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
    virtual void Initialize()
    {
    }
    virtual void Resize(int w, int h)
    {
        ///\todo: I think we need to delete this line
        glViewport(0, 0, w, h);
    }
    virtual void Paint()
    {
        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
            return;

        // matrices
        view.SetMatricesForViewport();
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

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

        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());

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
        }
    }
};



// ****************************************************************************
// Class:  eavl2DGLWindow
//
// Purpose:
///   A 2D output window with OpenGL/MesaGL rendering.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavl2DGLWindow : public eavlWindow
{
  public:
    eavl2DGLWindow(eavlView &v) : eavlWindow(v)
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
    
            if (dim < 2 || dim > 3)
                THROW(eavlException,"only supports 2 or 3 dimensions for now");
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

        eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2,
                                       (view.maxextents[1]+view.minextents[1]) / 2,
                                       (view.maxextents[2]+view.minextents[2]) / 2);

        view.view2d.l = view.minextents[0];
        view.view2d.r = view.maxextents[0];
        view.view2d.b = view.minextents[1];
        view.view2d.t = view.maxextents[1];
    }
    virtual void Initialize()
    {
    }
    virtual void Resize(int w, int h)
    {
    }
    virtual void Paint()
    {
        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
            return;

        // matrices
        float vl, vr, vt, vb;
        view.GetReal2DViewport(vl,vr,vb,vt);
        glViewport(float(view.w)*(1.+vl)/2.,
                   float(view.h)*(1.+vb)/2.,
                   float(view.w)*(vr-vl)/2.,
                   float(view.h)*(vt-vb)/2.);

        view.SetMatricesForViewport();

        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());

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
        }

        view.SetMatricesForScreen();
        glViewport(0,0,view.w,view.h);

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(-1,1, -1,1, -1,1);
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        //glLineWidth(2);
        glColor3f(.4,.4,.4);
        glBegin(GL_LINES);
        glVertex2d(vl, vt);
        glVertex2d(vl, vb);

        glVertex2d(vr, vt);
        glVertex2d(vr, vb);

        glVertex2d(vl, vt);
        glVertex2d(vr, vt);

        glVertex2d(vl, vb);
        glVertex2d(vr, vb);
        glEnd();
    }
};

#ifdef HAVE_MPI
#include <boost/mpi.hpp>

class eavl3DParallelGLWindow : public eavl3DGLWindow
{
  protected:
    boost::mpi::communicator &comm;
  public:
    eavl3DParallelGLWindow(boost::mpi::communicator &c) : comm(c)
    {
    }
    virtual void ResetView()
    {
        eavl3DGLWindow::ResetView();

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
