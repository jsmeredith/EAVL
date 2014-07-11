// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_GL_H
#define EAVL_SCENE_RENDERER_GL_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"

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
// Class:  eavlSceneRendererGL
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
class eavlSceneRendererGL : public eavlSceneRenderer
{
  public:
    virtual void RenderPoints(int npts, double *pts,
                              eavlField *field,
                              double vmin, double vmax,
                              eavlColorTable *ct)
    {
        ///\todo: HACK! PASS THESE IN!
        bool singlecolor = (field==NULL);
        eavlColor color(.5,.5,.5);
        bool field_nodal = (field && field->GetAssociation() == eavlField::ASSOC_POINTS);

        glPointSize(2);
        if (singlecolor)
        {
            glDisable(GL_LIGHTING);
            glColor3fv(color.c);

            eavlRenderPoints<false>(npts, pts, NULL,0,0,NULL);
        }
        else
        {
            if (!field_nodal)
                THROW(eavlException,
                      "Can't render points for cell-centered field.");

            eavlRenderPoints<true>(npts, pts, field,
                                   vmin, vmax,
                                   ct);
        }

    }
    virtual void RenderCells1D(eavlCellSet *cellset,
                               int npts, double *pts,
                               eavlField *field,
                               double vmin, double vmax,
                               eavlColorTable *ct)
    {
        ///\todo: HACK! PASS THESE IN!
        bool singlecolor = (field==NULL);
        eavlColor color(.5,.5,.5);
        bool field_nodal = (field && field->GetAssociation() == eavlField::ASSOC_POINTS);

        glLineWidth(2);
        if (singlecolor)
        {
            glDisable(GL_LIGHTING);
            glColor3fv(color.c);

            eavlRenderCells1D<false, false>(cellset, npts, pts, NULL,0,0,NULL);
        }
        else
        {
            if (!field)
                return;

            if (field_nodal)
            {
                eavlRenderCells1D<true, false>(cellset, npts, pts,
                                               field,
                                               vmin, vmax,
                                               ct);
                return;
            }
            else // is there any reason to suspect the field is not associated with this cell set? we should have double checked already...

            {
                eavlRenderCells1D<false, true>(cellset, npts, pts,
                                               field,
                                               vmin, vmax,
                                               ct);
                return;
            }

            THROW(eavlException,"Error finding field to render given cell set.");
        }
    }
    virtual void RenderCells2D(eavlCellSet *cellset,
                               int npts, double *pts,
                               eavlField *field,
                               double vmin, double vmax,
                               eavlColorTable *ct,
                               eavlField *normals)
    {
        ///\todo: HACK! PASS THESE IN!
        bool singlecolor = (field==NULL);
        eavlColor color(.5,.5,.5);
        bool field_nodal = (field && field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool wireframe = false;

        if (singlecolor)
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
        else
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
                                                                             vmin,
                                                                             vmax,
                                                                             ct, normals);
                    else if (normals)
                        eavlRenderCellsWireframe2D<true, false, false, true>(cellset,
                                                                             npts, pts,
                                                                             field,
                                                                             vmin,
                                                                             vmax,
                                                                             ct, normals);
                    else
                        eavlRenderCellsWireframe2D<true, false, false, false>(cellset,
                                                                              npts, pts,
                                                                              field,
                                                                              vmin,
                                                                              vmax,
                                                                              ct,
                                                                              NULL);
                }
                else
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCells2D<true, false, true, false>(cellset,
                                                                    npts, pts,
                                                                    field,
                                                                    vmin,
                                                                    vmax,
                                                                    ct, normals);
                    else if (normals)
                        eavlRenderCells2D<true, false, false, true>(cellset,
                                                                    npts, pts,
                                                                    field,
                                                                    vmin,
                                                                    vmax,
                                                                    ct, normals);
                    else
                        eavlRenderCells2D<true, false, false, false>(cellset,
                                                                     npts, pts,
                                                                     field,
                                                                     vmin,
                                                                     vmax,
                                                                     ct,
                                                                     NULL);
                }
                return;
            }
            else // is there any reason to suspect the field is not associated with this cell set? we should have double checked already...
            {
                if (wireframe)
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCellsWireframe2D<false, true, true, false>(cellset,
                                                                             npts, pts,
                                                                             field,
                                                                             vmin,
                                                                             vmax,
                                                                             ct, normals);
                    else if (normals)
                        eavlRenderCellsWireframe2D<false, true, false, true>(cellset,
                                                                             npts, pts,
                                                                             field,
                                                                             vmin,
                                                                             vmax,
                                                                             ct, normals);
                    else
                        eavlRenderCellsWireframe2D<false, true, false, false>(cellset,
                                                                              npts, pts,
                                                                              field,
                                                                              vmin,
                                                                              vmax,
                                                                              ct,
                                                                              NULL);
                }
                else
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCells2D<false, true, true, false>(cellset,
                                                                    npts, pts,
                                                                    field,
                                                                    vmin,
                                                                    vmax,
                                                                    ct, normals);
                    else if (normals)
                        eavlRenderCells2D<false, true, false, true>(cellset,
                                                                    npts, pts,
                                                                    field,
                                                                    vmin,
                                                                    vmax,
                                                                    ct, normals);
                    else
                        eavlRenderCells2D<false, true, false, false>(cellset,
                                                                     npts, pts,
                                                                     field,
                                                                     vmin,
                                                                     vmax,
                                                                     ct,
                                                                     NULL);
                }
                return;
            }

            THROW(eavlException,"Error finding field to render given cell set.");
        }
    }
};


#endif
