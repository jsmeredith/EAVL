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
//   Jeremy Meredith, Mon Aug 20 17:02:05 EDT 2012
//   Allow fields to have the same name but associate with multiple cell sets.
//
// ****************************************************************************
class eavlPseudocolorRenderer : public eavlRenderer
{
  protected:
    eavlColorTable colortable;
    std::vector<int> fieldindices;
    double vmin, vmax;
    bool nodal;
  public:
    eavlPseudocolorRenderer(eavlDataSet *ds,
                            const std::string &ctname,
                            const std::string &fieldname)
        : eavlRenderer(ds), colortable(ctname)
    {
        vmin = FLT_MAX;
        vmax = -FLT_MAX;
        nodal = false;
        for (int i=0; i<ds->GetNumFields(); ++i)
        {
            eavlField *f = ds->GetField(i);
            if (f->GetArray()->GetName() == fieldname)
            {
                nodal |= (f->GetAssociation() == eavlField::ASSOC_POINTS);
                if (nodal && fieldindices.size() > 0)
                    THROW(eavlException, "Can only have one nodal field with a given name.");
                fieldindices.push_back(i);
                
                // get its limits
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
                // don't break; we probably want to get the
                // extents of all fields with this same name
            }
        }
    }
    virtual void RenderPoints()
    {
        if (fieldindices.size() == 0)
            return;

        if (!nodal)
            THROW(eavlException, "Can't render points for cell-centered field.");

        glEnable(GL_DEPTH_TEST);
        glPointSize(2);

        eavlRenderPoints<true>(npts, pts,
                        dataset->GetField(fieldindices[0]), vmin,vmax, &colortable);
    }
    virtual void RenderCells1D(eavlCellSet *cs)
    {
        if (fieldindices.size() == 0)
            return;

        glEnable(GL_DEPTH_TEST);
        glLineWidth(2);

        for (unsigned int i=0; i<fieldindices.size(); ++i)
        {
            eavlField *f = dataset->GetField(fieldindices[i]);
            if (nodal)
            {
                eavlRenderCells1D<true, false>(cs, npts, pts,
                                               f, vmin, vmax, &colortable);
                return;
            }
            else if (dataset->GetCellSet(f->GetAssocCellSet()) == cs)
            {
                eavlRenderCells1D<false, true>(cs, npts, pts,
                                               f, vmin, vmax, &colortable);
                return;
            }
        }

        THROW(eavlException,"Error finding field to render given cell set.");
    }
    virtual void RenderCells2D(eavlCellSet *cs, eavlField *normals=NULL)
    {
        if (fieldindices.size() == 0)
            return;


        glEnable(GL_DEPTH_TEST);
        for (unsigned int i=0; i<fieldindices.size(); ++i)
        {
            eavlField *f = dataset->GetField(fieldindices[i]);
            if (nodal)
            {
                if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                    eavlRenderCells2D<true, false, true, false>(cs, npts, pts,
                                          f, vmin, vmax, &colortable, normals);
                else if (normals)
                    eavlRenderCells2D<true, false, false, true>(cs, npts, pts,
                                          f, vmin, vmax, &colortable, normals);
                else
                    eavlRenderCells2D<true, false, false, false>(cs, npts, pts,
                                          f, vmin, vmax, &colortable, NULL);
                return;
            }
            else if (dataset->GetCellSet(f->GetAssocCellSet()) == cs)
            {
                if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                    eavlRenderCells2D<false, true, true, false>(cs, npts, pts,
                                         f, vmin, vmax, &colortable, normals);
                else if (normals)
                    eavlRenderCells2D<false, true, false, true>(cs, npts, pts,
                                         f, vmin, vmax, &colortable, normals);
                else
                    eavlRenderCells2D<false, true, false, false>(cs, npts, pts,
                                         f, vmin, vmax, &colortable, NULL);
                return;
            }
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
        if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
            eavlRenderCells2D<false, false, true, false>(cs, npts, pts, NULL,0,0,NULL, normals);
        else if (normals)
            eavlRenderCells2D<false, false, false, true>(cs, npts, pts, NULL,0,0,NULL, normals);
        else
            eavlRenderCells2D<false, false, false, false>(cs, npts, pts, NULL,0,0,NULL, NULL);
    }
};

#endif
