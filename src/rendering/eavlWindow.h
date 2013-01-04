// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlCamera.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"

struct eavlPlot
{
    eavlDataSet  *data;
    string        colortable;
    int           cellset_index;
    int           variable_fieldindex;
    //int           variable_cellindex;
    eavlRenderer *pcRenderer;
    eavlRenderer *meshRenderer;

    eavlPlot()
        : data(NULL),
          colortable("default"),
          cellset_index(-1),
          variable_fieldindex(-1),
          pcRenderer(NULL),
          meshRenderer(NULL)
    {
    }
};



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

  public:
    eavlWindow() { }
    virtual void ResetView() = 0;
    virtual void Initialize() = 0;
    virtual void Resize(int w, int h) = 0;
    virtual void Paint() = 0;
};

// ****************************************************************************
// Class:  eavlWindow
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

  protected:
    float      dmin[3], dmax[3];
    float      ds_size;

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

        ds_size = sqrt( (dmax[0]-dmin[0])*(dmax[0]-dmin[0]) +
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

        // set up matrices
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(-1,1, -1,1, -1,1);

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        // create a color table
        ///\todo: we're only showing first-plot color table
        eavlColorTable ct(plots[0].colortable);
        if (plots[0].colortable != saved_colortable)
        {
            if (colortexId == 0)
            {
                glGenTextures(1, (GLuint*)&colortexId);
            }

            glBindTexture(GL_TEXTURE_1D, colortexId);
            // note: 2048 was NOT supported on Jeremy's Intel IGP laptop
            //       but 1024 IS.  Real NVIDIA cards can go up to 8192.
            const int n = 1024;
            float colors[n*3];
            for (int i=0; i<n; i++)
            {
                eavlColor c = ct.Map(float(i)/float(n-1));
                colors[3*i+0] = c.c[0];
                colors[3*i+1] = c.c[1];
                colors[3*i+2] = c.c[2];
            }
            glTexImage1D(GL_TEXTURE_1D, 0,
                         GL_RGB,
                         n,
                         0,
                         GL_RGB,
                         GL_FLOAT,
                         colors);

            saved_colortable = plots[0].colortable;
        }

        // draw the color table across a big 3d rectangle
        //glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glEnable(GL_TEXTURE_1D);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        //glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_R, GL_CLAMP); // R is the 3rd coord (not alphabetical)

        if (ct.smooth)
        {
            glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
        else
        {
            glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }
        glBindTexture(GL_TEXTURE_1D, colortexId); 
        glColor3fv(eavlColor::white.c);
        glBegin(GL_QUADS);
        glTexCoord1f(0);
        glVertex3f(-.9, .87 ,.99);
        glVertex3f(-.9, .95 ,.99);
        glTexCoord1f(1);
        glVertex3f(+.9, .95 ,.99);
        glVertex3f(+.9, .87 ,.99);
        glEnd();

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
        glDisable(GL_TEXTURE_1D);
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

        ds_size = sqrt( (dmax[0]-dmin[0])*(dmax[0]-dmin[0]) +
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
