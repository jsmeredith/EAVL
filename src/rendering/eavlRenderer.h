// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RENDERER_H
#define EAVL_RENDERER_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"

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
}


// ----------------------------------------------------------------------------
template <bool PointColors, bool CellColors, bool CellNormals>
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
    if (CellNormals)
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
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));
                glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
                glVertex3dv(&(pts[i1*3]));
                glTexCoord1f(MapValueToNorm(v2,vmin,vmax));
                glVertex3dv(&(pts[i2*3]));
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
                glTexCoord1f(MapValueToNorm(v0,vmin,vmax));
                glVertex3dv(&(pts[i0*3]));
                glTexCoord1f(MapValueToNorm(v1,vmin,vmax));
                glVertex3dv(&(pts[i1*3]));
                glTexCoord1f(MapValueToNorm(v2,vmin,vmax));
                glVertex3dv(&(pts[i2*3]));
                glTexCoord1f(MapValueToNorm(v3,vmin,vmax));
                glVertex3dv(&(pts[i3*3]));
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
                glVertex3dv(&(pts[i2*3]));
                glVertex3dv(&(pts[i3*3]));
            }
        }
    }
    glEnd();
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
// ****************************************************************************
class eavlRenderer
{
  protected:
    eavlDataSet *dataset;
    eavlCellSet *cellset;
    int          npts;
    double      *pts;
  public:
    eavlRenderer(eavlDataSet *ds)
        : dataset(ds)
    {
        // extract the points
        npts = dataset->npoints;
        int dim = dataset->coordinateSystems[0]->GetDimension();
    
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
                pts[3*i+d] = dataset->GetPoint(i,d);
            }
        }        
    }
    ~eavlRenderer()
    {
        delete[] pts;
    }
    virtual void RenderPoints() { }
    virtual void RenderCells(eavlCellSet *) { }
    virtual void RenderCells0D(eavlCellSet *) { }
    virtual void RenderCells1D(eavlCellSet *) { };
    virtual void RenderCells2D(eavlCellSet *, eavlField *) { };
    virtual void RenderCells3D(eavlCellSet *) { };
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
// ****************************************************************************
class eavlPseudocolorRenderer : public eavlRenderer
{
  protected:
    eavlColorTable colortable;
    int fieldindex;
    double vmin, vmax;
  public:
    eavlPseudocolorRenderer(eavlDataSet *ds,
                            const std::string &ctname,
                            const std::string &fieldname)
        : eavlRenderer(ds), colortable(ctname)
    {
        fieldindex = -1;
        vmin = vmax = 0;
        for (size_t i=0; i<ds->fields.size(); ++i)
        {
            eavlField *f = ds->fields[i];
            if (f->GetArray()->GetName() == fieldname)
            {
                fieldindex = i;

                // get its limits
                vmin = FLT_MAX;
                vmax = -FLT_MAX;

                int nvals = f->GetArray()->GetNumberOfTuples();
                for (int j=0; j<nvals; j++)
                {
                    // just do min/max based on first component for now
                    double value = f->GetArray()->GetComponentAsDouble(j,0);
                    if (value < vmin)
                        vmin = value;
                    if (value > vmax)
                        vmax = value;
                }
                break;
            }
        }
    }
    virtual void RenderPoints()
    {
        if (fieldindex < 0)
            return;

        glEnable(GL_DEPTH_TEST);
        glPointSize(2);

        eavlRenderPoints<true>(npts, pts,
                        dataset->fields[fieldindex], vmin,vmax, &colortable);
    }
    virtual void RenderCells1D(eavlCellSet *cs)
    {
        if (fieldindex < 0)
            return;

        eavlField *f = dataset->fields[fieldindex];
        bool nodal = (f->GetAssociation() == eavlField::ASSOC_POINTS);
        if (!nodal)
        {
            if (f->GetAssociation() != eavlField::ASSOC_CELL_SET)
                THROW(eavlException, "Only supports cell and node fields.");
            if (dataset->cellsets[f->GetAssocCellSet()] != cs)
            {
                THROW(eavlException,"Mismatch between cell set for "
                      "field to color by and cell set for geometry.");
            }
        }

        glEnable(GL_DEPTH_TEST);
        glLineWidth(2);
        if (nodal)
            eavlRenderCells1D<true, false>(cs, npts, pts,
                                           f, vmin, vmax, &colortable);
        else
            eavlRenderCells1D<false, true>(cs, npts, pts,
                                           f, vmin, vmax, &colortable);
    }
    virtual void RenderCells2D(eavlCellSet *cs, eavlField *normals=NULL)
    {
        if (fieldindex < 0)
            return;

        eavlField *f = dataset->fields[fieldindex];
        bool nodal = (f->GetAssociation() == eavlField::ASSOC_POINTS);
        if (!nodal)
        {
            if (f->GetAssociation() != eavlField::ASSOC_CELL_SET)
                THROW(eavlException, "Only supports cell and node fields.");
            if (dataset->cellsets[f->GetAssocCellSet()] != cs)
            {
                THROW(eavlException,"Mismatch between cell set for "
                      "field to color by and cell set for geometry.");
            }
        }

        glEnable(GL_DEPTH_TEST);
        glLineWidth(2);
        if (nodal)
        {
            if (normals)
                eavlRenderCells2D<true, false, true>(cs, npts, pts,
                                           f, vmin, vmax, &colortable, normals);
            else
                eavlRenderCells2D<true, false, false>(cs, npts, pts,
                                           f, vmin, vmax, &colortable, NULL);
        }
        else
        {
            if (normals)
                eavlRenderCells2D<false, true, true>(cs, npts, pts,
                                               f, vmin, vmax, &colortable, normals);
            else
                eavlRenderCells2D<false, true, false>(cs, npts, pts,
                                               f, vmin, vmax, &colortable, NULL);
        }
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
  public:
    eavlSingleColorRenderer(eavlDataSet *ds,
                            eavlColor c)
        : eavlRenderer(ds), color(c)
    {
    }
    virtual void RenderPoints()
    {
        glDisable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);
        glPointSize(2);

        glColor3fv(color.c);
        eavlRenderPoints<false>(npts, pts, NULL,0,0,NULL);
    }
    virtual void RenderCells1D(eavlCellSet *cs)
    {
        glDisable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);
        glLineWidth(2);
        eavlRenderCells1D<false, false>(cs, npts, pts, NULL,0,0,NULL);
    }
    virtual void RenderCells2D(eavlCellSet *cs, eavlField *normals=NULL)
    {
        glDisable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);
        if (normals)
            eavlRenderCells2D<false, false, true>(cs, npts, pts, NULL,0,0,NULL, normals);
        else
            eavlRenderCells2D<false, false, false>(cs, npts, pts, NULL,0,0,NULL, NULL);
    }
};

#endif
