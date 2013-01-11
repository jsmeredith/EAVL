// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlCamera.h"
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
    eavlCamera camera;
    std::map<std::string,eavlTexture*> textures;

  public:
    eavlWindow() { }
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
    eavl3DGLWindow() : eavlWindow()
    {
        colortexId = 0;
    }

    ///\todo: hack: this shouldn't be public, but I'm not sure it's even
    /// the right spot for it, so I'm working around it at the moment....
    float      dmin[3], dmax[3];
  protected:

    ///\todo: big hack for saved_colortable
    string saved_colortable;
    int colortexId;

    virtual void ResetView()
    {
        dmin[0] = dmin[1] = dmin[2] = FLT_MAX;
        dmax[0] = dmax[1] = dmax[2] = -FLT_MAX;

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
                    if (v < dmin[d])
                        dmin[d] = v;
                    if (v > dmax[d])
                        dmax[d] = v;
                }
            }
        }

        // untouched dims force to zero
        if (dmin[0] > dmax[0])
            dmin[0] = dmax[0] = 0;
        if (dmin[1] > dmax[1])
            dmin[1] = dmax[1] = 0;
        if (dmin[2] > dmax[2])
            dmin[2] = dmax[2] = 0;

        //cerr << "extents: "
        //     << dmin[0]<<":"<<dmax[0]<<"  "
        //     << dmin[1]<<":"<<dmax[1]<<"  "
        //     << dmin[2]<<":"<<dmax[2]<<"\n";

        float ds_size = sqrt( (dmax[0]-dmin[0])*(dmax[0]-dmin[0]) +
                              (dmax[1]-dmin[1])*(dmax[1]-dmin[1]) +
                              (dmax[2]-dmin[2])*(dmax[2]-dmin[2]) );

        eavlPoint3 center = eavlPoint3((dmax[0]+dmin[0]) / 2,
                                       (dmax[1]+dmin[1]) / 2,
                                       (dmax[2]+dmin[2]) / 2);

        camera.at   = center;
        camera.from = camera.at + eavlVector3(0,0, -ds_size*2);
        camera.up   = eavlVector3(0,1,0);
        camera.fov  = 0.5;
        camera.nearplane = ds_size/16.;
        camera.farplane = ds_size*4;

    }
    virtual void Initialize()
    {
    }
    virtual void Resize(int w, int h)
    {
        glViewport(0, 0, w, h);
        camera.aspect = float(w)/float(h);
    }
    virtual void Paint()
    {
        glClearColor(0.0, 0.2, 0.3, 1.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
            return;

        // matrices
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        camera.UpdateProjectionMatrix();
        glMultMatrixf(camera.P.GetOpenGLMatrix4x4());

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

        camera.UpdateViewMatrix();
        glMultMatrixf(camera.V.GetOpenGLMatrix4x4());

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
    eavl2DGLWindow() : eavlWindow()
    {
        colortexId = 0;
    }

  protected:
    float      dmin[3], dmax[3];

    ///\todo: big hack for saved_colortable
    string saved_colortable;
    int colortexId;

    virtual void ResetView()
    {
        dmin[0] = dmin[1] = dmin[2] = FLT_MAX;
        dmax[0] = dmax[1] = dmax[2] = -FLT_MAX;

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
                    if (v < dmin[d])
                        dmin[d] = v;
                    if (v > dmax[d])
                        dmax[d] = v;
                }
            }
        }

        // untouched dims force to zero
        if (dmin[0] > dmax[0])
            dmin[0] = dmax[0] = 0;
        if (dmin[1] > dmax[1])
            dmin[1] = dmax[1] = 0;
        if (dmin[2] > dmax[2])
            dmin[2] = dmax[2] = 0;

        //cerr << "extents: "
        //     << dmin[0]<<":"<<dmax[0]<<"  "
        //     << dmin[1]<<":"<<dmax[1]<<"  "
        //     << dmin[2]<<":"<<dmax[2]<<"\n";

        eavlPoint3 center = eavlPoint3((dmax[0]+dmin[0]) / 2,
                                       (dmax[1]+dmin[1]) / 2,
                                       (dmax[2]+dmin[2]) / 2);

        camera.twod = true;
        camera.l = dmin[0];
        camera.r = dmax[0];
        camera.t = dmin[1];
        camera.b = dmax[1];
    }
    virtual void Initialize()
    {
    }
    virtual void Resize(int w, int h)
    {
        //glViewport(0, 0, w, h);
        glViewport(w*.1, h*.1, w*.8, h*.4);
        camera.aspect = float(w)/float(h);
    }
    virtual void Paint()
    {
        glClearColor(0.0, 0.2, 0.3, 1.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        if (plots.size() == 0)
            return;

        int plotcount = 0;
        for (unsigned int i=0; i<plots.size(); i++)
            plotcount += (plots[i].data) ? 1 : 0;
        if (plotcount == 0)
            return;

        // matrices
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        camera.UpdateProjectionMatrix();
        glMultMatrixf(camera.P.GetOpenGLMatrix4x4());

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

        camera.UpdateViewMatrix();
        glMultMatrixf(camera.V.GetOpenGLMatrix4x4());

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

        // bounding box
        glDisable(GL_LIGHTING);
        glLineWidth(1);
        glColor3f(.6,.6,.6);
        glBegin(GL_LINES);
        glVertex3d(dmin[0],dmin[1],dmin[2]); glVertex3d(dmin[0],dmin[1],dmax[2]);
        glVertex3d(dmin[0],dmax[1],dmin[2]); glVertex3d(dmin[0],dmax[1],dmax[2]);
        glVertex3d(dmax[0],dmin[1],dmin[2]); glVertex3d(dmax[0],dmin[1],dmax[2]);
        glVertex3d(dmax[0],dmax[1],dmin[2]); glVertex3d(dmax[0],dmax[1],dmax[2]);

        glVertex3d(dmin[0],dmin[1],dmin[2]); glVertex3d(dmin[0],dmax[1],dmin[2]);
        glVertex3d(dmin[0],dmin[1],dmax[2]); glVertex3d(dmin[0],dmax[1],dmax[2]);
        glVertex3d(dmax[0],dmin[1],dmin[2]); glVertex3d(dmax[0],dmax[1],dmin[2]);
        glVertex3d(dmax[0],dmin[1],dmax[2]); glVertex3d(dmax[0],dmax[1],dmax[2]);

        glVertex3d(dmin[0],dmin[1],dmin[2]); glVertex3d(dmax[0],dmin[1],dmin[2]);
        glVertex3d(dmin[0],dmin[1],dmax[2]); glVertex3d(dmax[0],dmin[1],dmax[2]);
        glVertex3d(dmin[0],dmax[1],dmin[2]); glVertex3d(dmax[0],dmax[1],dmin[2]);
        glVertex3d(dmin[0],dmax[1],dmax[2]); glVertex3d(dmax[0],dmax[1],dmax[2]);
        glEnd();

        //delete[] pts;
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

        boost::mpi::all_reduce(comm, dmin[0], dmin[0], boost::mpi::minimum<float>());
        boost::mpi::all_reduce(comm, dmin[1], dmin[1], boost::mpi::minimum<float>());
        boost::mpi::all_reduce(comm, dmin[2], dmin[2], boost::mpi::minimum<float>());

        boost::mpi::all_reduce(comm, dmax[0], dmax[0], boost::mpi::maximum<float>());
        boost::mpi::all_reduce(comm, dmax[1], dmax[1], boost::mpi::maximum<float>());
        boost::mpi::all_reduce(comm, dmax[2], dmax[2], boost::mpi::maximum<float>());

        float ds_size = sqrt( (dmax[0]-dmin[0])*(dmax[0]-dmin[0]) +
                              (dmax[1]-dmin[1])*(dmax[1]-dmin[1]) +
                              (dmax[2]-dmin[2])*(dmax[2]-dmin[2]) );

        eavlPoint3 center = eavlPoint3((dmax[0]+dmin[0]) / 2,
                                       (dmax[1]+dmin[1]) / 2,
                                       (dmax[2]+dmin[2]) / 2);

        camera.at   = center;
        camera.from = camera.at + eavlVector3(0,0, -ds_size*2);
        camera.up   = eavlVector3(0,1,0);
        camera.fov  = 0.5;
        camera.nearplane = ds_size/16.;
        camera.farplane = ds_size*4;
    }
};




#endif

#endif
