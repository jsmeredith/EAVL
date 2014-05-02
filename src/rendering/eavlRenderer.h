// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RENDERER_H
#define EAVL_RENDERER_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"

static inline eavlColor MapValueToColor(double value,
                                 double vmin, double vmax,
                                 eavlColorTable &ct)
{
    double norm = 0.5;
    if (vmin != vmax)
        norm = (value - vmin) / (vmax - vmin);
    eavlColor c = ct.Map(norm);
    return c;
}

static inline float MapValueToNorm(double value,
                            double vmin, double vmax)
{
    double norm = 0.5;
    if (vmin != vmax)
        norm = (value - vmin) / (vmax - vmin);
    return norm;
}


// ----------------------------------------------------------------------------
template <bool PointColors>
void eavlRenderPoints(int npts, double *pts,
                      eavlField *f, double vmin, double vmax,
                      eavlColorTable *)
{
    glDisable(GL_LIGHTING);
    if (PointColors)
    {
        glColor3fv(eavlColor::white.c);
        glEnable(GL_TEXTURE_1D);
    }
    else
    {
        glDisable(GL_TEXTURE_1D);
    }
        
    glBegin(GL_POINTS);
    for (int j=0; j<npts; j++)
    {
        if (PointColors)
        {
            double value = f->GetArray()->GetComponentAsDouble(j,0);
            glTexCoord1f(MapValueToNorm(value, vmin, vmax));
        }
        glVertex3dv(&(pts[j*3]));
    }
    glEnd();

    glDisable(GL_TEXTURE_1D);
}


// ----------------------------------------------------------------------------
template <bool PointColors, bool CellColors>
void eavlRenderCells1D(eavlCellSet *cs,
                  int , double *pts,
                  eavlField *f, double vmin, double vmax,
                  eavlColorTable *)
{
    glDisable(GL_LIGHTING);
    if (PointColors || CellColors)
    {
        glColor3fv(eavlColor::white.c);
        glEnable(GL_TEXTURE_1D);
    }
    else
    {
        glDisable(GL_TEXTURE_1D);
    }

    int ncells = cs->GetNumCells();
    glBegin(GL_LINES);
    for (int j=0; j<ncells; j++)
    {
        eavlCell cell = cs->GetCellNodes(j);
        if (cell.type != EAVL_BEAM)
            continue;

        int i0 = cell.indices[0];
        int i1 = cell.indices[1];

        if (PointColors)
        {
            double v0 = f->GetArray()->GetComponentAsDouble(i0,0);
            double v1 = f->GetArray()->GetComponentAsDouble(i1,0);
            glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
            glVertex3dv(&(pts[i0*3]));
            glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
            glVertex3dv(&(pts[i1*3]));
        }
        else
        {
            if (CellColors)
            {
                double value = f->GetArray()->GetComponentAsDouble(j,0);
                glTexCoord1f(MapValueToNorm(value, vmin, vmax));
            }
            glVertex3dv(&(pts[i0*3]));
            glVertex3dv(&(pts[i1*3]));
        }
    }
    glEnd();

    glDisable(GL_TEXTURE_1D);
}


// ----------------------------------------------------------------------------
template <bool PointColors, bool CellColors, bool PointNormals, bool CellNormals>
void eavlRenderCells2D(eavlCellSet *cs,
                       int , double *pts,
                       eavlField *f, double vmin, double vmax,
                       eavlColorTable *,
                       eavlField *normals)
{
    if (PointColors || CellColors)
    {
        glColor3fv(eavlColor::white.c);
        glEnable(GL_TEXTURE_1D);
    }
    else
    {
        glDisable(GL_TEXTURE_1D);
    }
    if (PointNormals || CellNormals)
    {
        glEnable(GL_LIGHTING);
    }
    else
    {
        glDisable(GL_LIGHTING);
    }


    int ncells = cs->GetNumCells();
    // triangles
    glBegin(GL_TRIANGLES);
    for (int j=0; j<ncells; j++)
    {
        eavlCell cell = cs->GetCellNodes(j);
        if (cell.numIndices == 3)
        {
            int i0 = cell.indices[0];
            int i1 = cell.indices[1];
            int i2 = cell.indices[2];

            if (CellNormals)
            {
                double normal[3] = {
                    normals->GetArray()->GetComponentAsDouble(j,0),
                    normals->GetArray()->GetComponentAsDouble(j,1),
                    normals->GetArray()->GetComponentAsDouble(j,2)
                };
                glNormal3dv(normal);
            }
            if (PointColors)
            {
                double v0 = f->GetArray()->GetComponentAsDouble(i0,0);
                double v1 = f->GetArray()->GetComponentAsDouble(i1,0);
                double v2 = f->GetArray()->GetComponentAsDouble(i2,0);

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glTexCoord1f(MapValueToNorm(v2,vmin,vmax));
                glVertex3dv(&(pts[i2*3]));
            }
            else // no point colors
            {
                if (CellColors)
                {
                    double value = f->GetArray()->GetComponentAsDouble(j,0);
                    glTexCoord1f(MapValueToNorm(value, vmin, vmax));
                }

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glVertex3dv(&(pts[i2*3]));
            }
        }
    }
    glEnd();

    // quads
    glBegin(GL_QUADS);
    for (int j=0; j<ncells; j++)
    {
        eavlCell cell = cs->GetCellNodes(j);
        if (cell.numIndices == 4)
        {
            int i0 = cell.indices[0];
            int i1 = cell.indices[1];
            int i2 = cell.indices[2];
            int i3 = cell.indices[3];
            if (cell.type == EAVL_PIXEL)
            {
                int t = i2;
                i2 = i3;
                i3 = t;
            }

            if (CellNormals)
            {
                double normal[3] = {
                    normals->GetArray()->GetComponentAsDouble(j,0),
                    normals->GetArray()->GetComponentAsDouble(j,1),
                    normals->GetArray()->GetComponentAsDouble(j,2)
                };
                glNormal3dv(normal);
            }
            if (PointColors)
            {
                double v0 = f->GetArray()->GetComponentAsDouble(i0,0);
                double v1 = f->GetArray()->GetComponentAsDouble(i1,0);
                double v2 = f->GetArray()->GetComponentAsDouble(i2,0);
                double v3 = f->GetArray()->GetComponentAsDouble(i3,0);

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glTexCoord1f(MapValueToNorm(v2,vmin,vmax));
                glVertex3dv(&(pts[i2*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i3,0),
                               normals->GetArray()->GetComponentAsDouble(i3,1),
                               normals->GetArray()->GetComponentAsDouble(i3,2));
                glTexCoord1f(MapValueToNorm(v3,vmin,vmax));
                glVertex3dv(&(pts[i3*3]));
            }
            else // no point colors
            {
                if (CellColors)
                {
                    double value = f->GetArray()->GetComponentAsDouble(j,0);
                    glTexCoord1f(MapValueToNorm(value, vmin, vmax));
                }
                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glVertex3dv(&(pts[i2*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i3,0),
                               normals->GetArray()->GetComponentAsDouble(i3,1),
                               normals->GetArray()->GetComponentAsDouble(i3,2));
                glVertex3dv(&(pts[i3*3]));
            }
        }
    }
    glEnd();

    glDisable(GL_TEXTURE_1D);
}

// ----------------------------------------------------------------------------
template <bool PointColors, bool CellColors, bool PointNormals, bool CellNormals>
void eavlRenderCellsWireframe2D(eavlCellSet *cs,
                                int , double *pts,
                                eavlField *f, double vmin, double vmax,
                                eavlColorTable *,
                                eavlField *normals)
{
    if (PointColors || CellColors)
    {
        glColor3fv(eavlColor::white.c);
        glEnable(GL_TEXTURE_1D);
    }
    else
    {
        glDisable(GL_TEXTURE_1D);
    }
    if (PointNormals || CellNormals)
    {
        glEnable(GL_LIGHTING);
    }
    else
    {
        glDisable(GL_LIGHTING);
    }


    int ncells = cs->GetNumCells();
    // triangles
    glBegin(GL_LINES);
    for (int j=0; j<ncells; j++)
    {
        eavlCell cell = cs->GetCellNodes(j);
        if (cell.numIndices == 3)
        {
            int i0 = cell.indices[0];
            int i1 = cell.indices[1];
            int i2 = cell.indices[2];

            if (CellNormals)
            {
                double normal[3] = {
                    normals->GetArray()->GetComponentAsDouble(j,0),
                    normals->GetArray()->GetComponentAsDouble(j,1),
                    normals->GetArray()->GetComponentAsDouble(j,2)
                };
                glNormal3dv(normal);
            }
            if (PointColors)
            {
                double v0 = f->GetArray()->GetComponentAsDouble(i0,0);
                double v1 = f->GetArray()->GetComponentAsDouble(i1,0);
                double v2 = f->GetArray()->GetComponentAsDouble(i2,0);

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
                glVertex3dv(&(pts[i1*3]));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glTexCoord1f(MapValueToNorm(v2,vmin,vmax));
                glVertex3dv(&(pts[i2*3]));
                glVertex3dv(&(pts[i2*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));

            }
            else // no point colors
            {
                if (CellColors)
                {
                    double value = f->GetArray()->GetComponentAsDouble(j,0);
                    glTexCoord1f(MapValueToNorm(value, vmin, vmax));
                }

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glVertex3dv(&(pts[i1*3]));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glVertex3dv(&(pts[i2*3]));
                glVertex3dv(&(pts[i2*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glVertex3dv(&(pts[i0*3]));
            }
        }
    }
    glEnd();

    // quads
    glBegin(GL_LINES);
    for (int j=0; j<ncells; j++)
    {
        eavlCell cell = cs->GetCellNodes(j);
        if (cell.numIndices == 4)
        {
            int i0 = cell.indices[0];
            int i1 = cell.indices[1];
            int i2 = cell.indices[2];
            int i3 = cell.indices[3];
            if (cell.type == EAVL_PIXEL)
            {
                int t = i2;
                i2 = i3;
                i3 = t;
            }

            if (CellNormals)
            {
                double normal[3] = {
                    normals->GetArray()->GetComponentAsDouble(j,0),
                    normals->GetArray()->GetComponentAsDouble(j,1),
                    normals->GetArray()->GetComponentAsDouble(j,2)
                };
                glNormal3dv(normal);
            }
            if (PointColors)
            {
                double v0 = f->GetArray()->GetComponentAsDouble(i0,0);
                double v1 = f->GetArray()->GetComponentAsDouble(i1,0);
                double v2 = f->GetArray()->GetComponentAsDouble(i2,0);
                double v3 = f->GetArray()->GetComponentAsDouble(i3,0);

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
                glVertex3dv(&(pts[i1*3]));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glTexCoord1f(MapValueToNorm(v2,vmin,vmax));
                glVertex3dv(&(pts[i2*3]));
                glVertex3dv(&(pts[i2*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i3,0),
                               normals->GetArray()->GetComponentAsDouble(i3,1),
                               normals->GetArray()->GetComponentAsDouble(i3,2));
                glTexCoord1f(MapValueToNorm(v3,vmin,vmax));
                glVertex3dv(&(pts[i3*3]));
                glVertex3dv(&(pts[i3*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));

            }
            else // no point colors
            {
                if (CellColors)
                {
                    double value = f->GetArray()->GetComponentAsDouble(j,0);
                    glTexCoord1f(MapValueToNorm(value, vmin, vmax));
                }
                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glVertex3dv(&(pts[i0*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i1,0),
                               normals->GetArray()->GetComponentAsDouble(i1,1),
                               normals->GetArray()->GetComponentAsDouble(i1,2));
                glVertex3dv(&(pts[i1*3]));
                glVertex3dv(&(pts[i1*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i2,0),
                               normals->GetArray()->GetComponentAsDouble(i2,1),
                               normals->GetArray()->GetComponentAsDouble(i2,2));
                glVertex3dv(&(pts[i2*3]));
                glVertex3dv(&(pts[i2*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i3,0),
                               normals->GetArray()->GetComponentAsDouble(i3,1),
                               normals->GetArray()->GetComponentAsDouble(i3,2));
                glVertex3dv(&(pts[i3*3]));
                glVertex3dv(&(pts[i3*3]));

                if (PointNormals)
                    glNormal3d(normals->GetArray()->GetComponentAsDouble(i0,0),
                               normals->GetArray()->GetComponentAsDouble(i0,1),
                               normals->GetArray()->GetComponentAsDouble(i0,2));
                glVertex3dv(&(pts[i0*3]));
            }
        }
    }
    glEnd();

    glDisable(GL_TEXTURE_1D);
}


// ****************************************************************************
// Class:  eavlRenderer
//
// Purpose:
///   Base class for renderers.
//
// Programmer:  Jeremy Meredith
// Creation:    July 18, 2012
//
// Modifications:
//   Jeremy Meredith, Mon Mar  4 15:44:23 EST 2013
//   Big refactoring; more consistent internal code with less
//   duplication and cleaner external API.
//
// ****************************************************************************
class eavlRenderer
{
  protected:
    eavlDataSet *dataset;
    int          npts;
    double      *pts;
    eavlCellSet *cellset;
    eavlField   *field;
    bool         field_nodal;
    eavlField   *normals;
    string       name;

    double min_coord_extents[3];
    double max_coord_extents[3];

    double min_data_extents;
    double max_data_extents;
  public:
    eavlRenderer(eavlDataSet *ds,
                 void (*xform)(double c0, double c1, double c2,
                                 double &x, double &y, double &z),
                 const string &csname = "",
                 const string &fieldname = "")
        : dataset(ds), cellset(NULL), field(NULL), normals(NULL)
    {
        if (fieldname != "")
            name = fieldname;
        else if (csname != "")
            name = csname;
        else
            name = "points";

        //
        // extract the points and find coordinate extents
        //

        min_coord_extents[0] = min_coord_extents[1] = min_coord_extents[2] = +DBL_MAX;
        max_coord_extents[0] = max_coord_extents[1] = max_coord_extents[2] = -DBL_MAX;

        npts = dataset->GetNumPoints();
        int dim = dataset->GetCoordinateSystem(0)->GetDimension();
    
        //CHIMERA HACK
        if (dim > 3)
            dim = 3;

        pts = new double[npts*3];
        for (int i=0; i<npts; i++)
        {
            pts[3*i+0] = 0;
            pts[3*i+1] = 0;
            pts[3*i+2] = 0;
            for (int d=0; d<dim; d++)
            {
                double v = dataset->GetPoint(i,d);
                pts[3*i+d] = v;
                if (v < min_coord_extents[d])
                    min_coord_extents[d] = v;
                if (v > max_coord_extents[d])
                    max_coord_extents[d] = v;
            }
        }

        // untouched dims force to zero
        if (min_coord_extents[0] > max_coord_extents[0])
            min_coord_extents[0] = max_coord_extents[0] = 0;
        if (min_coord_extents[1] > max_coord_extents[1])
            min_coord_extents[1] = max_coord_extents[1] = 0;
        if (min_coord_extents[2] > max_coord_extents[2])
            min_coord_extents[2] = max_coord_extents[2] = 0;

        //
        // if they gave us a transform, get pts[] into cartesian space
        //
        if (xform)
        {
            for (int i=0; i<npts; i++)
            {
                double x,y,z;
                xform(pts[3*i+0], pts[3*i+1], pts[3*i+2], x, y, z);
                pts[3*i+0] = x;
                pts[3*i+1] = y;
                pts[3*i+2] = z;
            }
        }

        //
        // if they gave us a cell set, grab a pointer to it
        //
        if (csname != "")
        {
            cellset = ds->GetCellSet(csname);

            for (int i=0; i<dataset->GetNumFields(); i++)
            {
                eavlField *f = ds->GetField(i);
                if (f->GetArray()->GetName() == "surface_normals" &&
                    f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                    dataset->GetCellSet(f->GetAssocCellSet()) == cellset)
                {
                    normals = f;
                }
            }
            // override with node-centered ones if we have them
            for (int i=0; i<dataset->GetNumFields(); i++)
            {
                eavlField *f = ds->GetField(i);
                if (f->GetArray()->GetName() == "nodecentered_surface_normals" &&
                    f->GetAssociation() == eavlField::ASSOC_POINTS)
                {
                    normals = dataset->GetField(i);
                }
            }
        }

        //
        // if they gave us a field, find its data extents
        //
        min_data_extents = 0;
        max_data_extents = 0;
        if (fieldname != "")
        {
            min_data_extents = +DBL_MAX;
            max_data_extents = -DBL_MAX;
            field_nodal = false;
            for (int i=0; i<ds->GetNumFields(); ++i)
            {
                eavlField *f = ds->GetField(i);
                if (f->GetArray()->GetName() == fieldname)
                {
                    if (f->GetAssociation() == eavlField::ASSOC_CELL_SET)
                    {
                        if (cellset &&
                            cellset != ds->GetCellSet(f->GetAssocCellSet()))
                        {
                            // the caller has specified a cell set, but not this one
                            continue;
                        }
                    }

                    field_nodal = (f->GetAssociation() == eavlField::ASSOC_POINTS);
                
                    // get its limits
                    int nvals = f->GetArray()->GetNumberOfTuples();
                    //int ncomp = f->GetArray()->GetNumberOfComponents();
                    for (int j=0; j<nvals; j++)
                    {
                        // just do min/max based on first component for now
                        double value = f->GetArray()->GetComponentAsDouble(j,0);
                        if (value < min_data_extents)
                            min_data_extents = value;
                        if (value > max_data_extents)
                            max_data_extents = value;
                    }

                    // Do we break here?  In the old code, we would
                    // plot all cell sets for a field.  Now we pick one.
                    // So we can probably safely break now.
                    field = f;
                    break;
                }
            }
        }
    }
    ~eavlRenderer()
    {
        delete[] pts;
    }
    string GetName() { return name; }
    eavlDataSet *GetDataSet() { return dataset; }
    double GetMinCoordExtent(int axis) { return min_coord_extents[axis]; }
    double GetMaxCoordExtent(int axis) { return max_coord_extents[axis]; }
    double GetMinDataExtent() { return min_data_extents; }
    double GetMaxDataExtent() { return max_data_extents; }
    void SetDataExtents(double minval, double maxval)
    {
        min_data_extents = minval;
        max_data_extents = maxval;
    }
    virtual string GetColorTableName() { return ""; }
    void Render()
    {
        try
        {
            if (!cellset)
            {
                RenderPoints();
            }
            else
            {
                if (cellset->GetDimensionality() == 1)
                {
                    RenderCells1D();
                }
                else if (cellset->GetDimensionality() == 2)
                {
                    RenderCells2D();
                }
                else if (cellset->GetDimensionality() == 3)
                {
                    // do anything with 3D?
                    // not right now, now.
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
    virtual void RenderPoints() { }
    virtual void RenderCells0D() { }
    virtual void RenderCells1D() { };
    virtual void RenderCells2D() { };
    virtual void RenderCells3D() { };
};

// ****************************************************************************
// Class:  eavlPseudocolorRenderer
//
// Purpose:
///   Render a cell set (or the points) using a field and color table, and
///   potentially with surface normals for lighting.
//
// Programmer:  Jeremy Meredith
// Creation:    July 18, 2012
//
// Modifications:
//   Jeremy Meredith, Mon Aug 20 17:02:05 EDT 2012
//   Allow fields to have the same name but associate with multiple cell sets.
//
// ****************************************************************************
class eavlPseudocolorRenderer : public eavlRenderer
{
  protected:
    string         colortablename;
    eavlColorTable colortable;
    bool           wireframe;
  public:
    eavlPseudocolorRenderer(eavlDataSet *ds,
                            void (*xform)(double c0, double c1, double c2,
                                            double &x, double &y, double &z),
                            const std::string &ctname,
                            bool wire,
                            const string &csname,
                            const string &fieldname)
        : eavlRenderer(ds, xform, csname, fieldname),
          colortablename(ctname),
          colortable(ctname),
          wireframe(wire)
    {
    }
    virtual string GetColorTableName()
    {
        return colortablename;
    }
    virtual void RenderPoints()
    {
        if (!field_nodal)
            THROW(eavlException, "Can't render points for cell-centered field.");

        glPointSize(2);

        eavlRenderPoints<true>(npts, pts, field,
                               min_data_extents, max_data_extents,
                               &colortable);
    }
    virtual void RenderCells1D()
    {
        if (!field)
            return;

        glLineWidth(2);

        if (field_nodal)
        {
            eavlRenderCells1D<true, false>(cellset, npts, pts,
                                           field,
                                           min_data_extents, max_data_extents,
                                           &colortable);
            return;
        }
        else if (dataset->GetCellSet(field->GetAssocCellSet()) == cellset)
        {
            eavlRenderCells1D<false, true>(cellset, npts, pts,
                                           field,
                                           min_data_extents, max_data_extents,
                                           &colortable);
            return;
        }

        THROW(eavlException,"Error finding field to render given cell set.");
    }
    virtual void RenderCells2D()
    {
        if (!field)
            return;

        if (field_nodal)
        {
            if (wireframe)
            {
                if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                    eavlRenderCellsWireframe2D<true, false, true, false>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else if (normals)
                    eavlRenderCellsWireframe2D<true, false, false, true>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else
                    eavlRenderCellsWireframe2D<true, false, false, false>(cellset,
                                                                 npts, pts,
                                                                 field,
                                                                 min_data_extents,
                                                                 max_data_extents,
                                                                 &colortable,
                                                                 NULL);
            }
            else
            {
                if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                    eavlRenderCells2D<true, false, true, false>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else if (normals)
                    eavlRenderCells2D<true, false, false, true>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else
                    eavlRenderCells2D<true, false, false, false>(cellset,
                                                                 npts, pts,
                                                                 field,
                                                                 min_data_extents,
                                                                 max_data_extents,
                                                                 &colortable,
                                                                 NULL);
            }
            return;
        }
        else if (dataset->GetCellSet(field->GetAssocCellSet()) == cellset)
        {
            if (wireframe)
            {
                if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                    eavlRenderCellsWireframe2D<false, true, true, false>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else if (normals)
                    eavlRenderCellsWireframe2D<false, true, false, true>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else
                    eavlRenderCellsWireframe2D<false, true, false, false>(cellset,
                                                                 npts, pts,
                                                                 field,
                                                                 min_data_extents,
                                                                 max_data_extents,
                                                                 &colortable,
                                                                 NULL);
            }
            else
            {
                if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                    eavlRenderCells2D<false, true, true, false>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else if (normals)
                    eavlRenderCells2D<false, true, false, true>(cellset,
                                                                npts, pts,
                                                                field,
                                                                min_data_extents,
                                                                max_data_extents,
                                                                &colortable, normals);
                else
                    eavlRenderCells2D<false, true, false, false>(cellset,
                                                                 npts, pts,
                                                                 field,
                                                                 min_data_extents,
                                                                 max_data_extents,
                                                                 &colortable,
                                                                 NULL);
            }
            return;
        }

        THROW(eavlException,"Error finding field to render given cell set.");
    }
};

// ****************************************************************************
// Class:  eavlSingleColorRenderer
//
// Purpose:
///   Render a cell set (or the points) using a single color and potentially
///   lighting with surface normals.
//
// Programmer:  Jeremy Meredith
// Creation:    July 18, 2012
//
// Modifications:
// ****************************************************************************
class eavlSingleColorRenderer : public eavlRenderer
{
  protected:
    eavlColor color;
    bool wireframe;
  public:
    eavlSingleColorRenderer(eavlDataSet *ds,
                            void (*xform)(double c0, double c1, double c2,
                                            double &x, double &y, double &z),
                            eavlColor c,
                            bool wire,
                            const string &csname)
        : eavlRenderer(ds, xform, csname), color(c), wireframe(wire)
    {
    }
    virtual void RenderPoints()
    {
        glDisable(GL_LIGHTING);
        glPointSize(2);
        glColor3fv(color.c);
        eavlRenderPoints<false>(npts, pts, NULL,0,0,NULL);
    }
    virtual void RenderCells1D()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2);
        glColor3fv(color.c);
        eavlRenderCells1D<false, false>(cellset, npts, pts, NULL,0,0,NULL);
    }
    virtual void RenderCells2D()
    {
        glDisable(GL_LIGHTING);
        glColor3fv(color.c);
        if (wireframe)
        {
            if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                eavlRenderCellsWireframe2D<false, false, true, false>(cellset, npts, pts, NULL,0,0,NULL, normals);
            else if (normals)
                eavlRenderCellsWireframe2D<false, false, false, true>(cellset, npts, pts, NULL,0,0,NULL, normals);
            else
                eavlRenderCellsWireframe2D<false, false, false, false>(cellset, npts, pts, NULL,0,0,NULL, NULL);
        }
        else
        {
            if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                eavlRenderCells2D<false, false, true, false>(cellset, npts, pts, NULL,0,0,NULL, normals);
            else if (normals)
                eavlRenderCells2D<false, false, false, true>(cellset, npts, pts, NULL,0,0,NULL, normals);
            else
                eavlRenderCells2D<false, false, false, false>(cellset, npts, pts, NULL,0,0,NULL, NULL);
        }
    }
};

// ****************************************************************************
// Class:  eavlCurveRenderer
//
// Purpose:
///   Render a 1D field as a curve.
//
// Programmer:  Jeremy Meredith
// Creation:    January 16, 2013
//
// Modifications:
//
// ****************************************************************************
class eavlCurveRenderer : public eavlRenderer
{
  protected:
    eavlColor color;
    bool      logarithmic;
  public:
    eavlCurveRenderer(eavlDataSet *ds,
                      void (*xform)(double c0, double c1, double c2,
                                      double &x, double &y, double &z),
                      eavlColor c,
                      const string &csname,
                      const string &fieldname)
        : eavlRenderer(ds, xform, csname, fieldname), color(c), logarithmic(false)
    {
    }
    void SetLogarithmic(bool l)
    {
        logarithmic = l;
    }
    eavlColor GetColor() { return color; }
    virtual void RenderPoints()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2);

        glColor3fv(color.c);
        glBegin(GL_LINES);
        for (int j=0; j<npts; j++)
        {
            double value = field->GetArray()->GetComponentAsDouble(j,0);
            if (logarithmic)
                value = log10(value);
            glVertex2d(pts[j*3+0], value);
            if (j>0 && j<npts-1)
                glVertex2d(pts[j*3+0], value);
        }
        glEnd();
    }
    virtual void RenderCells1D()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2);

        glColor3fv(color.c);

        glBegin(GL_LINES);
        if (field_nodal)
        {
            ///\todo: should probably render the cells still, not
            /// simply assume the cell set is point indices 1..n
            for (int j=0; j<npts; j++)
            {
                double value = field->GetArray()->GetComponentAsDouble(j,0);
                if (logarithmic)
                    value = log10(value);

                glVertex2d(pts[j*3+0], value);
                if (j>0 && j<npts-1)
                    glVertex2d(pts[j*3+0], value);
            }
        }
        else
        {
            int ncells = cellset->GetNumCells();
            for (int j=0; j<ncells; j++)
            {
                eavlCell cell = cellset->GetCellNodes(j);
                if (cell.type != EAVL_BEAM)
                    continue;

                double value = field->GetArray()->GetComponentAsDouble(j,0);
                if (logarithmic)
                    value = log10(value);

                int i0 = cell.indices[0];
                int i1 = cell.indices[1];

                if (j>0)
                    glVertex2d(pts[i0*3 + 0], value);
                glVertex2d(pts[i0*3 + 0], value);
                glVertex2d(pts[i1*3 + 0], value);
                if (j<npts-1)
                    glVertex2d(pts[i1*3 + 0], value);

            }
        }
        glEnd();
    }
};

// ****************************************************************************
// Class:  eavlBarRenderer
//
// Purpose:
///   Render a 1D field as a series of vertical bars;
//
// Programmer:  Jeremy Meredith
// Creation:    January 16, 2013
//
// Modifications:
//
// ****************************************************************************
class eavlBarRenderer : public eavlRenderer
{
  protected:
    std::vector<int> fieldindices;
    eavlColor color;
    float gap;
  public:
    eavlBarRenderer(eavlDataSet *ds,
                    void (*xform)(double c0, double c1, double c2,
                                    double &x, double &y, double &z),
                    eavlColor c,
                    float interbargap,
                    const string &csname,
                    const string &fieldname)
        : eavlRenderer(ds, xform, csname, fieldname), color(c), gap(interbargap)
    {
        ///\todo: a bit of a hack: force min data value to 0
        /// since a histpgram starts at zero
        min_data_extents = 0;
    }
    virtual void RenderCells1D()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2);

        glColor3fv(color.c);

        glBegin(GL_QUADS);
        if (field_nodal)
        {
            for (int j=0; j<npts; j++)
            {
                //double x = pts[j*3+0];
                //double value = field->GetArray()->GetComponentAsDouble(j,0);
                ///\todo: how to handle this?
            }
        }
        else
        {
            int ncells = cellset->GetNumCells();
            for (int j=0; j<ncells; j++)
            {
                eavlCell cell = cellset->GetCellNodes(j);
                if (cell.type != EAVL_BEAM)
                    continue;

                double value = field->GetArray()->GetComponentAsDouble(j,0);

                int i0 = cell.indices[0];
                int i1 = cell.indices[1];

                double x0 = pts[i0*3 + 0];
                double x1 = pts[i1*3 + 0];

                double w = x1-x0;
                double g = w * gap / 2.;
                glVertex2d(x0 + g, 0);
                glVertex2d(x1 - g, 0);
                glVertex2d(x1 - g, value);
                glVertex2d(x0 + g, value);
            }
        }
        glEnd();
    }
};

#endif
