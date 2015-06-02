// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_H
#define EAVL_SCENE_RENDERER_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlView.h"
#include "eavlRenderSurface.h"

static inline float MapValueToNorm(double value,
                                   double vmin,
                                   double vmax,
                                   bool logarithmic)
{
    double norm = 0.5;
    if (logarithmic)
    {
        if (value <= 0)
        {
            norm = 0;
        }
        else if (vmin != vmax)
        {
            if (vmin <= 0)
                vmin = 1.e-100;
            if (vmax <= 0)
                vmax = 1.e-100;

            norm = (log(value) - log(vmin)) / (log(vmax) - log(vmin));
        }

        if (norm < 0)
            norm = 0;
        if (norm > 1)
            norm = 1;
    }
    else
    {
        if (vmin != vmax)
        {
            norm = (value - vmin) / (vmax - vmin);
        }
    }
    return norm;
}


struct ColorByOptions
{
    bool singleColor; ///< if true, use one flat color, else field+colortable
    eavlColor color;  ///< color to use when singleColor==true
    eavlField *field; ///< field to color by when singleColor==false
    bool logscale;    ///< true if using logarithmic scaling for colortable
    double vmin, vmax; ///< field min and max
    eavlColorTable ct; ///< colortable to color by when singleColor==false
};

// ****************************************************************************
// Class:  eavlSceneRenderer
//
// Purpose:
///   Base class for renderers.
//
// Programmer:  Jeremy Meredith
// Creation:    July 15, 2014
//
// Modifications:
//   Jeremy Meredith, Mon Mar  4 15:44:23 EST 2013
//   Big refactoring; more consistent internal code with less
//   duplication and cleaner external API.
//
// ****************************************************************************
class eavlSceneRenderer
{
  protected:
    int ncolors;
    float colors[3*1024];
    set<int> plotcontents;

    eavlView view;

    float Ka;
    float Kd;
    float Ks;
    float Lx, Ly, Lz;
    bool  eyeLight;
    eavlRenderSurface *surface;
  public:
    eavlSceneRenderer()
    {
        Ka = 0.2;
        Kd = 0.8;
        Ks = 0.2;

        Lx = 0.2;
        Ly = 0.2;
        Lz = 1.0;

        eyeLight = true;

        ncolors = 1;
        colors[0] = colors[1] = colors[2] = 0.5;
    }
    virtual ~eavlSceneRenderer()
    {
    }

    void SetRenderSurface(eavlRenderSurface *surf)
    {
        surface = surf;
    }
    void SetAmbientCoefficient(float a)
    {
        Ka = a;
    }
    void SetDiffuseCoefficient(float d)
    {
        Kd = d;
    }
    void SetSpecularCoefficient(float s)
    {
        Ks = s;
    }
    void SetLightDirection(float x, float y, float z)
    {
        Lx = x;
        Ly = y;
        Lz = z;
    }
    void SetEyeLight(bool eye)
    {
        eyeLight = eye;
    }
    void SetView(eavlView v)
    {
        view = v;
    }


    virtual bool NeedsGeometryForPlot(int plotid)
    {
        bool containsplot = plotcontents.count(plotid) > 0;
        //cerr << "NeedsGeometryForPlot("<<plotid<<"): "<<(containsplot?"no":"yes")<<endl;
        return !containsplot;
    }
    virtual void SendingGeometryForPlot(int plotid)
    {
        plotcontents.insert(plotid);
    }

    virtual void Render() = 0;
    virtual bool ShouldRenderAgain() { return false; }

    virtual void StartScene()
    {
        //cerr << "StartScene\n";
        plotcontents.clear();
    }
    virtual void EndScene()
    {
        //cerr << "EndScene\n";
    }

    virtual void StartTriangles() { }
    virtual void EndTriangles() { }

    virtual void StartTetrahedra() { }
    virtual void EndTetrahedra() { }

    virtual void StartPoints() { }
    virtual void EndPoints() { }

    virtual void StartLines() { }
    virtual void EndLines() { }

    //
    // per-plot properties (in essence, at least) follow:
    //

    virtual void SetActiveColor(eavlColor c)
    {
        ncolors = 1;
        colors[0] = c.c[0];
        colors[1] = c.c[1];
        colors[2] = c.c[2];
    }
    virtual void SetActiveColorTable(eavlColorTable colortable)
    {
        ncolors = 1024;
        colortable.Sample(ncolors, colors);
    }
    //virtual void SetActiveMaterial() { } // diffuse, specular, ambient
    //virtual void SetActiveLighting() { } // etc.

    virtual unsigned char *GetRGBAPixels()
    {
        return NULL;
    }

    virtual float *GetDepthPixels()
    {
        return NULL;
    }

    // ----------------------------------------
    // Vertex Normal
    //----------------------------------------

    // vertex scalar
    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2) = 0;

    // face scalar
    virtual void AddTriangleVnCs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s)
    {
        AddTriangleVnVs(x0,y0,z0,
                        x1,y1,z1,
                        x2,y2,z2,
                        u0,v0,w0,
                        u1,v1,w1,
                        u2,v2,w2,
                        s, s, s);
    }
    // no scalar
    virtual void AddTriangleVn(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double u0, double v0, double w0,
                               double u1, double v1, double w1,
                               double u2, double v2, double w2)
    {
        AddTriangleVnVs(x0,y0,z0,
                        x1,y1,z1,
                        x2,y2,z2,
                        u0,v0,w0,
                        u1,v1,w1,
                        u2,v2,w2,
                        0, 0, 0);
    }

    // ----------------------------------------
    // Face Normal
    //----------------------------------------

    // vertex scalar
    virtual void AddTriangleCnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u,  double v,  double w,
                                 double s0, double s1, double s2)
    {
        AddTriangleVnVs(x0,y0,z0,
                        x1,y1,z1,
                        x2,y2,z2,
                        u ,v ,w ,
                        u ,v ,w ,
                        u ,v ,w ,
                        s0,s1,s2);
    }
    // face scalar
    virtual void AddTriangleCnCs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u,  double v,  double w,
                                 double s)
    {
        AddTriangleVnVs(x0,y0,z0,
                        x1,y1,z1,
                        x2,y2,z2,
                        u ,v ,w ,
                        u ,v ,w ,
                        u ,v ,w ,
                        s, s, s);
    }
    // no scalar
    virtual void AddTriangleCn(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double u,  double v,  double w)
    {
        AddTriangleVnVs(x0,y0,z0,
                        x1,y1,z1,
                        x2,y2,z2,
                        u ,v ,w ,
                        u ,v ,w ,
                        u ,v ,w ,
                        0, 0, 0);
    }

    // ----------------------------------------
    // No Normal
    // ----------------------------------------

    // vertex scalar
    virtual void AddTriangleVs(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double s0, double s1, double s2)
    {
        eavlVector3 e0(x1-x0, y1-y0, z1-z0);
        eavlVector3 e1(x2-x1, y2-y1, z2-z1);
        eavlVector3 n((e0 % e1).normalized());
        double u = n[0];
        double v = n[1];
        double w = n[2];
        AddTriangleCnVs(x0,y0,z0,
                        x1,y1,z1,
                        x2,y2,z2,
                        u ,v ,w ,
                        s0,s1,s2);
    }
    // face scalar
    virtual void AddTriangleCs(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double s)
    {
        AddTriangleVs(x0,y0,z0,
                      x1,y1,z1,
                      x2,y2,z2,
                      s, s, s);
    }
    // no scalar
    virtual void AddTriangle(double x0, double y0, double z0,
                             double x1, double y1, double z1,
                             double x2, double y2, double z2)
    {
        double s = 0;
        AddTriangleVs(x0,y0,z0,
                      x1,y1,z1,
                      x2,y2,z2,
                      s, s, s);
    }

    // ----------------------------------------
    // Point
    // ----------------------------------------
    virtual void AddPoint(double x, double y, double z, double r)
    {
        AddPointVs(x,y,z,r,0);
    }
    virtual void AddPointVs(double x, double y, double z, double r, double s)
        = 0;

    // ----------------------------------------
    // Line
    // ----------------------------------------
    virtual void AddLine(double x0, double y0, double z0,
                         double x1, double y1, double z1)
    {
        AddLineVs(x0,y0,z0, x1,y1,z1, 0,0);
    }
    virtual void AddLineCs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s)
    {
        AddLineVs(x0,y0,z0, x1,y1,z1, s,s);
    }
    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
        = 0;

    // ----------------------------------------
    // Tetrahedron
    // ----------------------------------------
    virtual void AddTetrahedron(double x0, double y0, double z0,
                                double x1, double y1, double z1,
                                double x2, double y2, double z2,
                                double x3, double y3, double z3)
    {
        AddTetrahedronVs(x0,y0,z0,
                         x1,y1,z1,
                         x2,y2,z2,
                         x3,y3,z3,
                         0,0,0,0);
    }
    virtual void AddTetrahedronCs(double x0, double y0, double z0,
                                  double x1, double y1, double z1,
                                  double x2, double y2, double z2,
                                  double x3, double y3, double z3,
                                  double s)
    {
        AddTetrahedronVs(x0,y0,z0,
                         x1,y1,z1,
                         x2,y2,z2,
                         x3,y3,z3,
                         s,s,s,s);
    }
    virtual void AddTetrahedronVs(double x0, double y0, double z0,
                                  double x1, double y1, double z1,
                                  double x2, double y2, double z2,
                                  double x3, double y3, double z3,
                                  double s0, double s1, double s2, double s3)
    {
        ///\todo: is having this be implemented (no-op) in base class
        /// the right thing, since most renderers don't do volumes?
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------

    virtual void RenderPoints(int npts, double *pts,
                              ColorByOptions opts)
    {
        eavlField *f = opts.field;
        bool NoColors = (opts.field == NULL);
        bool PointColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);

        if (opts.singleColor)
            SetActiveColor(opts.color);
        else
            SetActiveColorTable(opts.ct);

        StartPoints();

        double radius = view.size / 300.;
        for (int j=0; j<npts; j++)
        {
            double x0 = pts[j*3+0];
            double y0 = pts[j*3+1];
            double z0 = pts[j*3+2];

            if (PointColors)
            {
                double v = MapValueToNorm(f->GetArray()->GetComponentAsDouble(j,0),
                                          opts.vmin, opts.vmax, opts.logscale);
                AddPointVs(x0,y0,z0, radius, v);
            }
            else
            {
                AddPoint(x0,y0,z0, radius);
            }
        }

        EndPoints();
    }
    virtual void RenderCells0D(eavlCellSet *cs,
                               int , double *pts,
                               ColorByOptions opts)
    {
        eavlField *f = opts.field;
        bool NoColors = (opts.field == NULL);
        bool PointColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                opts.field->GetAssocCellSet() == cs->GetName());

        if (opts.singleColor)
            SetActiveColor(opts.color);
        else
            SetActiveColorTable(opts.ct);

        StartPoints();

        double radius = view.size / 300.;
        int ncells = cs->GetNumCells();
        for (int c=0; c<ncells; c++)
        {
            eavlCell cell = cs->GetCellNodes(c);
            if (cell.type != EAVL_POINT)
                continue;

            int p = cell.indices[0];

            // get vertex coordinates
            double x0 = pts[p*3+0];
            double y0 = pts[p*3+1];
            double z0 = pts[p*3+2];

            // get scalars (if applicable)
            if (CellColors)
            {
                double s = MapValueToNorm(f->GetArray()->
                                          GetComponentAsDouble(c,0),
                                          opts.vmin, opts.vmax, opts.logscale);
                AddPointVs(x0,y0,z0, radius, s);
            }
            else if (PointColors)
            {
                double s = MapValueToNorm(f->GetArray()->
                                          GetComponentAsDouble(p,0),
                                          opts.vmin, opts.vmax, opts.logscale);
                AddPointVs(x0,y0,z0, radius, s);
            }
            else
            {
                AddPoint(x0,y0,z0, radius);
            }
        }


        EndPoints();
    }
    virtual void RenderCells1D(eavlCellSet *cs,
                               int , double *pts,
                               ColorByOptions opts)
    {
        eavlField *f = opts.field;
        bool NoColors = (opts.field == NULL);
        bool PointColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                opts.field->GetAssocCellSet() == cs->GetName());

        if (opts.singleColor)
            SetActiveColor(opts.color);
        else
            SetActiveColorTable(opts.ct);

        StartLines();

        int ncells = cs->GetNumCells();
        for (int j=0; j<ncells; j++)
        {
            eavlCell cell = cs->GetCellNodes(j);
            if (cell.type != EAVL_BEAM)
                continue;

            int i0 = cell.indices[0];
            int i1 = cell.indices[1];

            // get vertex coordinates
            double x0 = pts[i0*3+0];
            double y0 = pts[i0*3+1];
            double z0 = pts[i0*3+2];

            double x1 = pts[i1*3+0];
            double y1 = pts[i1*3+1];
            double z1 = pts[i1*3+2];


            // get scalars (if applicable)
            double s=0, s0=0, s1=0;
            if (CellColors)
            {
                s = MapValueToNorm(f->GetArray()->
                                   GetComponentAsDouble(j,0),
                                   opts.vmin, opts.vmax, opts.logscale);
            }
            else if (PointColors)
            {
                s0 = MapValueToNorm(f->GetArray()->
                                    GetComponentAsDouble(i0,0),
                                    opts.vmin, opts.vmax, opts.logscale);
                s1 = MapValueToNorm(f->GetArray()->
                                    GetComponentAsDouble(i1,0),
                                    opts.vmin, opts.vmax, opts.logscale);
            }

            if (NoColors)
            {
                AddLine(x0,y0,z0, x1,y1,z1);
            }
            else if (CellColors)
            {
                AddLineCs(x0,y0,z0, x1,y1,z1, s);
            }
            else if (PointColors)
            {
                AddLineVs(x0,y0,z0, x1,y1,z1, s0,s1);
            }
        }


        EndLines();

    }
    virtual void RenderCells2D(eavlCellSet *cs,
                               int , double *pts,
                               ColorByOptions opts,
                               bool wireframe,
                               eavlField *normals)
    {
        eavlField *f = opts.field;
        bool NoColors = (opts.field == NULL);
        bool PointColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                opts.field->GetAssocCellSet() == cs->GetName());
        bool NoNormals = (normals == NULL);
        bool PointNormals = (normals &&
                normals->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellNormals = (normals &&
                normals->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                normals->GetAssocCellSet() == cs->GetName());

        if (opts.singleColor)
            SetActiveColor(opts.color);
        else
            SetActiveColorTable(opts.ct);

        StartTriangles();

        int ncells = cs->GetNumCells();

        // triangles, quads, pixels, and polygons; all 2D shapes
        for (int j=0; j<ncells; j++)
        {
            eavlCell cell = cs->GetCellNodes(j);
            if (cell.type != EAVL_TRI &&
                cell.type != EAVL_QUAD &&
                cell.type != EAVL_PIXEL &&
                cell.type != EAVL_POLYGON)
            {
                continue;
            }

            // tesselate polygons with more than 3 points
            for (int pass = 3; pass <= cell.numIndices; ++pass)
            {
                int i0 = cell.indices[0];
                int i1 = cell.indices[pass-2];
                int i2 = cell.indices[pass-1];
                // pixel is a special case
                if (pass == 4 && cell.type == EAVL_PIXEL)
                {
                    i0 = cell.indices[1];
                    i1 = cell.indices[3];
                    i2 = cell.indices[2];
                }

                // get vertex coordinates
                double x0 = pts[i0*3+0];
                double y0 = pts[i0*3+1];
                double z0 = pts[i0*3+2];

                double x1 = pts[i1*3+0];
                double y1 = pts[i1*3+1];
                double z1 = pts[i1*3+2];

                double x2 = pts[i2*3+0];
                double y2 = pts[i2*3+1];
                double z2 = pts[i2*3+2];

                // get scalars (if applicable)
                double s=0, s0=0, s1=0, s2=0;
                if (CellColors)
                {
                    s = MapValueToNorm(f->GetArray()->
                                       GetComponentAsDouble(j,0),
                                       opts.vmin, opts.vmax, opts.logscale);
                }
                else if (PointColors)
                {
                    s0 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i0,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    s1 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i1,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    s2 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i2,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                }

                // get normals (if applicable)
                double u=0,v=0,w=0, u0=0,v0=0,w0=0, u1=0,v1=0,w1=0, u2=0,v2=0,w2=0;
                if (CellNormals)
                {
                    u = normals->GetArray()->GetComponentAsDouble(j,0);
                    v = normals->GetArray()->GetComponentAsDouble(j,1);
                    w = normals->GetArray()->GetComponentAsDouble(j,2);
                }
                else if (PointNormals)
                {
                    u0 = normals->GetArray()->GetComponentAsDouble(i0,0);
                    v0 = normals->GetArray()->GetComponentAsDouble(i0,1);
                    w0 = normals->GetArray()->GetComponentAsDouble(i0,2);

                    u1 = normals->GetArray()->GetComponentAsDouble(i1,0);
                    v1 = normals->GetArray()->GetComponentAsDouble(i1,1);
                    w1 = normals->GetArray()->GetComponentAsDouble(i1,2);

                    u2 = normals->GetArray()->GetComponentAsDouble(i2,0);
                    v2 = normals->GetArray()->GetComponentAsDouble(i2,1);
                    w2 = normals->GetArray()->GetComponentAsDouble(i2,2);
                }


                // send the triangle down
                if (NoNormals)
                {
                    if (NoColors)
                    {
                        AddTriangle(x0,y0,z0, x1,y1,z1, x2,y2,z2);
                    }
                    else if (CellColors)
                    {
                        AddTriangleCs(x0,y0,z0, x1,y1,z1, x2,y2,z2, s);
                    }
                    else if (PointColors)
                    {
                        AddTriangleVs(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                      s0,s1,s2);
                    }
                }
                else if (CellNormals)
                {
                    if (NoColors)
                    {
                        AddTriangleCn(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                      u,v,w);
                    }
                    else if (CellColors)
                    {
                        AddTriangleCnCs(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                        u,v,w, s);
                    }
                    else if (PointColors)
                    {
                        AddTriangleCnVs(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                        u,v,w, s0,s1,s2);
                    }
                }
                else if (PointNormals)
                {
                    if (NoColors)
                    {
                        AddTriangleVn(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                      u0,v0,w0, u1,v1,w1, u2,v2,w2);
                    }
                    else if (CellColors)
                    {
                        AddTriangleVnCs(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                        u0,v0,w0, u1,v1,w1, u2,v2,w2,
                                        s);
                    }
                    else if (PointColors)
                    {
                        AddTriangleVnVs(x0,y0,z0, x1,y1,z1, x2,y2,z2,
                                        u0,v0,w0, u1,v1,w1, u2,v2,w2,
                                        s0,s1,s2);
                    }
                }
            }
        }

        EndTriangles();
    }
    virtual void RenderCells3D(eavlCellSet *cs,
                               int , double *pts,
                               ColorByOptions opts)
    {
        eavlField *f = opts.field;
        bool NoColors = (opts.field == NULL);
        bool PointColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellColors = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                opts.field->GetAssocCellSet() == cs->GetName());

        if (opts.singleColor)
            SetActiveColor(opts.color);
        else
            SetActiveColorTable(opts.ct);

        StartTetrahedra();

        int ncells = cs->GetNumCells();

        // tetrahedralize all 3d shapes
        int tet[] = {0,1,2,3};

        int pyr[] = {0,1,2,4,
                     0,2,3,4};

        int wdg[] = {0,2,1,4,
                     0,3,2,4,
                     3,5,2,4};

        int hex[] = {0,1,2,5,
                     0,2,3,7,
                     0,7,4,5,
                     2,6,7,5,
                     0,5,2,7};
        
        int vox[] = {0,1,3,5,
                     0,3,2,6,
                     0,6,4,5,
                     3,7,6,5,
                     0,5,3,6};
        
        for (int j=0; j<ncells; j++)
        {
            eavlCell cell = cs->GetCellNodes(j);

            int *shapes = NULL;
            int nshapes = 0;
            switch (cell.type)
            {
              case EAVL_TET:
                shapes = tet;
                nshapes = 1;
                break;
              case EAVL_PYRAMID:
                shapes = pyr;
                nshapes = 2;
                break;
              case EAVL_WEDGE:
                shapes = wdg;
                nshapes = 3;
                break;
              case EAVL_HEX:
                shapes = hex;
                nshapes = 5;
                break;
              case EAVL_VOXEL:
                shapes = vox;
                nshapes = 5;
                break;
              default:
                shapes = NULL;
                nshapes = 0;
                continue;
            }

            for (int s=0; s<nshapes; ++s)
            {
                int i0 = cell.indices[shapes[4*s+0]];
                int i1 = cell.indices[shapes[4*s+1]];
                int i2 = cell.indices[shapes[4*s+2]];
                int i3 = cell.indices[shapes[4*s+3]];

                double x0 = pts[i0*3+0];
                double y0 = pts[i0*3+1];
                double z0 = pts[i0*3+2];

                double x1 = pts[i1*3+0];
                double y1 = pts[i1*3+1];
                double z1 = pts[i1*3+2];

                double x2 = pts[i2*3+0];
                double y2 = pts[i2*3+1];
                double z2 = pts[i2*3+2];

                double x3 = pts[i3*3+0];
                double y3 = pts[i3*3+1];
                double z3 = pts[i3*3+2];

                // get scalars (if applicable)
                double ss, s0, s1, s2, s3;
                if (CellColors)
                {
                    ss = MapValueToNorm(f->GetArray()->
                                       GetComponentAsDouble(j,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    AddTetrahedronCs(x0,y0,z0,
                                     x1,y1,z1,
                                     x2,y2,z2,
                                     x3,y3,z3,
                                     ss);
                }
                else if (PointColors)
                {
                    s0 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i0,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    s1 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i1,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    s2 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i2,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    s3 = MapValueToNorm(f->GetArray()->
                                        GetComponentAsDouble(i3,0),
                                        opts.vmin, opts.vmax, opts.logscale);
                    AddTetrahedronVs(x0,y0,z0,
                                     x1,y1,z1,
                                     x2,y2,z2,
                                     x3,y3,z3,
                                     s0,s1,s2,s3);
                }
                else
                {
                    AddTetrahedron(x0,y0,z0,
                                   x1,y1,z1,
                                   x2,y2,z2,
                                   x3,y3,z3);
                }
            }

        }

        EndTetrahedra();
    }
};


#endif
