// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_GL_H
#define EAVL_SCENE_RENDERER_GL_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"

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
                              ColorByOptions opts)
    {
        bool field_nodal = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);

        glPointSize(2);
        if (opts.singleColor)
        {
            glDisable(GL_LIGHTING);
            glColor3fv(opts.color.c);

            eavlRenderPoints<false>(npts, pts, NULL,0,0,NULL);
        }
        else
        {
            if (!field_nodal)
                THROW(eavlException,
                      "Can't render points for cell-centered field.");

            eavlRenderPoints<true>(npts, pts, opts.field,
                                   opts.vmin, opts.vmax,
                                   opts.ct);
        }

    }
    virtual void RenderCells1D(eavlCellSet *cellset,
                               int npts, double *pts,
                               ColorByOptions opts)
    {
        bool field_nodal = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);

        glLineWidth(2);
        if (opts.singleColor)
        {
            glDisable(GL_LIGHTING);
            glColor3fv(opts.color.c);

            eavlRenderCells1D<false, false>(cellset, npts, pts, NULL,0,0,NULL);
        }
        else
        {
            if (!opts.field)
                return;

            if (field_nodal)
            {
                eavlRenderCells1D<true, false>(cellset, npts, pts,
                                               opts.field,
                                               opts.vmin, opts.vmax,
                                               opts.ct);
                return;
            }
            else if (opts.field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                     opts.field->GetAssocCellSet() == cellset->GetName())
            {
                eavlRenderCells1D<false, true>(cellset, npts, pts,
                                               opts.field,
                                               opts.vmin, opts.vmax,
                                               opts.ct);
                return;
            }

            THROW(eavlException,"Error finding field to render given cell set.");
        }
    }
    virtual void RenderCells2D_old(eavlCellSet *cellset,
                               int npts, double *pts,
                               ColorByOptions opts,
                               bool wireframe,
                               eavlField *normals)
    {
        bool field_nodal = (opts.field &&
                opts.field->GetAssociation() == eavlField::ASSOC_POINTS);

        if (opts.singleColor)
        {
            glDisable(GL_LIGHTING);
            glColor3fv(opts.color.c);

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
            if (!opts.field)
                return;

            if (field_nodal)
            {
                if (wireframe)
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCellsWireframe2D<true, false, true, false>(cellset,
                                                                             npts, pts,
                                                                             opts.field,
                                                                             opts.vmin,
                                                                             opts.vmax,
                                                                             opts.ct, normals);
                    else if (normals)
                        eavlRenderCellsWireframe2D<true, false, false, true>(cellset,
                                                                             npts, pts,
                                                                             opts.field,
                                                                             opts.vmin,
                                                                             opts.vmax,
                                                                             opts.ct, normals);
                    else
                        eavlRenderCellsWireframe2D<true, false, false, false>(cellset,
                                                                              npts, pts,
                                                                              opts.field,
                                                                              opts.vmin,
                                                                              opts.vmax,
                                                                              opts.ct,
                                                                              NULL);
                }
                else
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCells2D<true, false, true, false>(cellset,
                                                                    npts, pts,
                                                                    opts.field,
                                                                    opts.vmin,
                                                                    opts.vmax,
                                                                    opts.ct, normals);
                    else if (normals)
                        eavlRenderCells2D<true, false, false, true>(cellset,
                                                                    npts, pts,
                                                                    opts.field,
                                                                    opts.vmin,
                                                                    opts.vmax,
                                                                    opts.ct, normals);
                    else
                        eavlRenderCells2D<true, false, false, false>(cellset,
                                                                     npts, pts,
                                                                     opts.field,
                                                                     opts.vmin,
                                                                     opts.vmax,
                                                                     opts.ct,
                                                                     NULL);
                }
                return;
            }
            else if (opts.field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                     opts.field->GetAssocCellSet() == cellset->GetName())
            {
                if (wireframe)
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCellsWireframe2D<false, true, true, false>(cellset,
                                                                             npts, pts,
                                                                             opts.field,
                                                                             opts.vmin,
                                                                             opts.vmax,
                                                                             opts.ct, normals);
                    else if (normals)
                        eavlRenderCellsWireframe2D<false, true, false, true>(cellset,
                                                                             npts, pts,
                                                                             opts.field,
                                                                             opts.vmin,
                                                                             opts.vmax,
                                                                             opts.ct, normals);
                    else
                        eavlRenderCellsWireframe2D<false, true, false, false>(cellset,
                                                                              npts, pts,
                                                                              opts.field,
                                                                              opts.vmin,
                                                                              opts.vmax,
                                                                              opts.ct,
                                                                              NULL);
                }
                else
                {
                    if (normals && normals->GetAssociation()==eavlField::ASSOC_POINTS)
                        eavlRenderCells2D<false, true, true, false>(cellset,
                                                                    npts, pts,
                                                                    opts.field,
                                                                    opts.vmin,
                                                                    opts.vmax,
                                                                    opts.ct, normals);
                    else if (normals)
                        eavlRenderCells2D<false, true, false, true>(cellset,
                                                                    npts, pts,
                                                                    opts.field,
                                                                    opts.vmin,
                                                                    opts.vmax,
                                                                    opts.ct, normals);
                    else
                        eavlRenderCells2D<false, true, false, false>(cellset,
                                                                     npts, pts,
                                                                     opts.field,
                                                                     opts.vmin,
                                                                     opts.vmax,
                                                                     opts.ct,
                                                                     NULL);
                }
                return;
            }

            THROW(eavlException,"Error finding field to render given cell set.");
        }
    }

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
        glNormal3d(u0,v0,w0);
        glTexCoord1f(s0);
        glVertex3d(x0,y0,z0);

        glNormal3d(u1,v1,w1);
        glTexCoord1f(s1);
        glVertex3d(x1,y1,z1);

        glNormal3d(u2,v2,w2);
        glTexCoord1f(s2);
        glVertex3d(x2,y2,z2);
    }
    virtual void StartTriangles()
    {
        glEnable(GL_TEXTURE_1D);
        glBegin(GL_TRIANGLES);
    }

    virtual void EndTriangles()
    {
        glEnd();
        glDisable(GL_TEXTURE_1D);
    }

};


#endif
