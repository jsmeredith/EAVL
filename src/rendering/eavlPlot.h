// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_PLOT_H
#define EAVL_PLOT_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"

class eavlPlot
{
  protected:
    int          id;
    eavlDataSet *dataset;
    int          npts;
    double      *origpts;
    double      *finalpts;
    string       cellsetname;
    eavlCellSet *cellset;
    eavlField   *normals;

    double min_coord_extents_orig[3];
    double max_coord_extents_orig[3];

    double min_coord_extents_final[3];
    double max_coord_extents_final[3];

    eavlField   *field;
    bool         field_nodal;
    double min_data_extents;
    double max_data_extents;
    string name;

  protected:
    ///\todo:
    /// some of this is common, some is dimensionality-specific
    eavlColorTable colortable;
    bool           logcolorscaling;
    bool           wireframe;
    eavlColor      color;

  public:
    int GetID() const { return id; }
    eavlPlot(eavlDataSet *ds,
             const string &csname = "")
        : dataset(ds), cellsetname(csname), cellset(NULL), normals(NULL)
    {
        static int next_id = 1;
        id = next_id;
        next_id++;

        /// initializer for other stuff
        field = NULL;
        wireframe = false;
        logcolorscaling = false;
        color = eavlColor(.5,.5,.5);
        min_data_extents = max_data_extents = 0;
        if (csname != "")
            name = csname;
        else
            name = "points";

        //
        // extract the points and find coordinate extents
        //

        min_coord_extents_orig[0] = min_coord_extents_orig[1] = min_coord_extents_orig[2] = +DBL_MAX;
        max_coord_extents_orig[0] = max_coord_extents_orig[1] = max_coord_extents_orig[2] = -DBL_MAX;

        npts = dataset->GetNumPoints();
        int dim = dataset->GetCoordinateSystem(0)->GetDimension();
    
        //CHIMERA HACK
        if (dim > 3)
            dim = 3;

        origpts = new double[npts*3];
        for (int i=0; i<npts; i++)
        {
            origpts[3*i+0] = 0;
            origpts[3*i+1] = 0;
            origpts[3*i+2] = 0;
            for (int d=0; d<dim; d++)
            {
                double v = dataset->GetPoint(i,d);
                origpts[3*i+d] = v;
                if (v < min_coord_extents_orig[d])
                    min_coord_extents_orig[d] = v;
                if (v > max_coord_extents_orig[d])
                    max_coord_extents_orig[d] = v;
            }
        }

        // untouched dims force to zero
        for (int dim=0; dim<3; ++dim)
        {
            if (min_coord_extents_orig[dim] > max_coord_extents_orig[dim])
                min_coord_extents_orig[dim] = max_coord_extents_orig[dim] = 0;
        }

        // initialize final points and extents to original
        finalpts = origpts;
        for (int dim=0; dim<3; ++dim)
            min_coord_extents_final[dim] = min_coord_extents_orig[dim];
        for (int dim=0; dim<3; ++dim)
            max_coord_extents_final[dim] = max_coord_extents_orig[dim];

        //
        // if they gave us a cell set, grab a pointer to it
        //
        if (csname != "")
        {
            cellset = dataset->GetCellSet(csname);

            for (int i=0; i<dataset->GetNumFields(); i++)
            {
                eavlField *f = dataset->GetField(i);
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
                eavlField *f = dataset->GetField(i);
                if (f->GetArray()->GetName() == "nodecentered_surface_normals" &&
                    f->GetAssociation() == eavlField::ASSOC_POINTS)
                {
                    normals = dataset->GetField(i);
                }
            }
        }
    }

    void SetLogarithmicColorScaling(bool ls)
    {
        logcolorscaling = ls;
    }
    bool GetLogarithmicColorScaling() const
    {
        return logcolorscaling;
    }

    void SetTransformFunction(void (*xform)(double c0, double c1, double c2,
                                            double &x, double &y, double &z))
    {
        if (finalpts == origpts)
            finalpts = new double[npts*3];
        min_coord_extents_final[0] = min_coord_extents_final[1] = min_coord_extents_final[2] = +DBL_MAX;
        max_coord_extents_final[0] = max_coord_extents_final[1] = max_coord_extents_final[2] = -DBL_MAX;
        for (int i=0; i<npts; i++)
        {
            double x,y,z;
            xform(origpts[3*i+0], origpts[3*i+1], origpts[3*i+2], x, y, z);
            finalpts[3*i+0] = x;
            finalpts[3*i+1] = y;
            finalpts[3*i+2] = z;
            for (int dim=0; dim<3; ++dim)
            {
                double v = finalpts[3*i+dim];
                if (v < min_coord_extents_final[dim])
                    min_coord_extents_final[dim] = v;
                if (v > max_coord_extents_final[dim])
                    max_coord_extents_final[dim] = v;
            }
        }
    }

    void SetField(string fieldname)
    {
        field = NULL;
        if (fieldname != "")
            name = fieldname;
        else if (cellsetname != "")
            name = cellsetname;
        else
            name = "points";

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
            for (int i=0; i<dataset->GetNumFields(); ++i)
            {
                eavlField *f = dataset->GetField(i);
                if (f->GetArray()->GetName() == fieldname)
                {
                    if (f->GetAssociation() == eavlField::ASSOC_CELL_SET)
                    {
                        if (cellset &&
                            cellset != dataset->GetCellSet(f->GetAssocCellSet()))
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

    virtual ~eavlPlot()
    {
        if (finalpts != origpts)
            delete[] finalpts;
        delete[] origpts;
        finalpts = NULL;
        origpts = NULL;
    }
    string GetName() { return name; }
    double GetMinDataExtent() { return min_data_extents; }
    double GetMaxDataExtent() { return max_data_extents; }
    void SetDataExtents(double minval, double maxval)
    {
        min_data_extents = minval;
        max_data_extents = maxval;
    }
    void SetColorTableByName(string colortablename, bool reverse = false)
    {
        colortable = eavlColorTable(colortablename);
        if (reverse)
            colortable.Reverse();
    }
    void SetColorTable(eavlColorTable ctable)
    {
        colortable = eavlColorTable(ctable);
    }
    eavlColorTable GetColorTable()
    {
        return colortable;
    }
    eavlColor GetColor()
    {
        return color;
    }
    void SetWireframe(bool wf)
    {
        wireframe = wf;
    }
    void SetSingleColor(eavlColor c)
    {
        color = c;
    }

    eavlDataSet *GetDataSet() { return dataset; }
    double GetMinCoordExtentFinal(int axis) { return min_coord_extents_final[axis]; }
    double GetMaxCoordExtentFinal(int axis) { return max_coord_extents_final[axis]; }
    double GetMinCoordExtentOrig(int axis) { return min_coord_extents_orig[axis]; }
    double GetMaxCoordExtentOrig(int axis) { return max_coord_extents_orig[axis]; }

    virtual void Generate(eavlSceneRenderer *r)
    {
        ColorByOptions opts;
        opts.singleColor = (field == NULL);
        opts.color = color;
        opts.field = field;
        opts.vmin = min_data_extents;
        opts.vmax = max_data_extents;
        opts.logscale = logcolorscaling;
        opts.ct = colortable;

        try
        {
            //cerr << "RENDERING\n";
            if (!cellset)
            {
                //cerr << "RENDERING POINTS\n";
                r->RenderPoints(npts, finalpts, opts);
            }
            else
            {
                if (cellset->GetDimensionality() == 0)
                {
                    //cerr << "RENDERING 1D CELLS\n";
                    r->RenderCells0D(cellset, npts, finalpts, opts);
                }
                else if (cellset->GetDimensionality() == 1)
                {
                    //cerr << "RENDERING 1D CELLS\n";
                    r->RenderCells1D(cellset, npts, finalpts, opts);
                }
                else if (cellset->GetDimensionality() == 2)
                {
                    //cerr << "RENDERING 2D CELLS\n";
                    r->RenderCells2D(cellset, npts, finalpts, opts,
                                     wireframe, normals);
                }
                else if (cellset->GetDimensionality() == 3)
                {
                    r->RenderCells3D(cellset, npts, finalpts, opts);
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
};

class eavl1DPlot : public eavlPlot
{
  protected:
    bool barstyle;
    bool logarithmic;
  public:
    eavl1DPlot(eavlDataSet *ds, const string &csname = "")
        : eavlPlot(ds, csname)
    {
        barstyle = false;
        logarithmic = false;
    }
    virtual void SetBarStyle(bool bs)
    {
        barstyle = bs;
    }
    virtual void SetLogarithmic(bool l)
    {
        logarithmic = l;
    }
    virtual void Generate(eavlSceneRenderer *r)
    {
        if (logarithmic && min_data_extents <= 0)
        {
            cerr << "ERROR: logarithmic plot with values <= 0\n";
            //THROW(eavlException, "Log plot with nonpositive values");
        }

        if (barstyle)
        {
            if (cellset)
                GenerateBars(r);
            else
                GenerateBarsForPoints(r);
        }
        else
        {
            if (cellset)
                GenerateCells(r);
            else
                GeneratePoints(r);
        }
    }

    void GeneratePoints(eavlSceneRenderer *r)
    {
        bool PointField = (field &&
            field->GetAssociation() == eavlField::ASSOC_POINTS);

        r->SetActiveColor(color);

        r->StartPoints();

        double radius = 1.0;
        for (int j=0; j<npts; j++)
        {
            double x = finalpts[j*3+0];

            if (PointField)
            {
                double v = field->GetArray()->GetComponentAsDouble(j,0);
                if (logarithmic)
                    v = log10(v);
                r->AddPoint(x,v,0.0, radius);
            }
            else
            {
                r->AddPoint(x,0.0,0.0, radius);
            }
        }

        r->EndPoints();
    }

    void GenerateCells(eavlSceneRenderer *r)
    {
        bool PointField = (field &&
            field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellField = (field &&
            field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            field->GetAssocCellSet() == cellset->GetName());

        r->SetActiveColor(color);

        r->StartLines();

        int ncells = cellset->GetNumCells();
        for (int j=0; j<ncells; j++)
        {
            eavlCell cell = cellset->GetCellNodes(j);
            if (cell.type != EAVL_BEAM)
                continue;

            int i0 = cell.indices[0];
            int i1 = cell.indices[1];

            // get vertex coordinates
            double x0 = finalpts[i0*3+0];
            double x1 = finalpts[i1*3+0];

            if (CellField)
            {
                double v = field->GetArray()->GetComponentAsDouble(j,0);
                if (logarithmic)
                    v = log10(v);
                if (j > 0)
                {
                    double v_last = field->GetArray()->GetComponentAsDouble(j-1,0);
                    if (logarithmic)
                        v_last = log10(v_last);
                    r->AddLine(x0,v_last, 0.0,  x0,v, 0.0);
                }
                r->AddLine(x0,v, 0.0, x1,v, 0.0);
            }
            else if (PointField)
            {
                double v0 = field->GetArray()->GetComponentAsDouble(i0,0);
                double v1 = field->GetArray()->GetComponentAsDouble(i1,0);
                if (logarithmic)
                {
                    v0 = log10(v0);
                    v1 = log10(v1);
                }
                r->AddLine(x0,v0, 0.0, x1,v1, 0.0);
            }
            else
            {
                r->AddLine(x0,0.0,0.0, x1,0.0,0.0);
            }
        }

        r->EndLines();
    }

    void GenerateBarsForPoints(eavlSceneRenderer *r)
    {
        bool PointField = (field &&
            field->GetAssociation() == eavlField::ASSOC_POINTS);

        if (!PointField)
            return;

        double minval = 0;
        if (logarithmic)
            minval = floor(log10(min_data_extents));

        r->SetActiveColor(color);

        r->StartLines();

        for (int j=0; j<npts; j++)
        {
            double x = finalpts[j*3+0];
            double v = field->GetArray()->GetComponentAsDouble(j,0);
            if (logarithmic)
                v = log10(v);
            r->AddLine(x,minval,0, x,v,0);
        }

        r->EndLines();
    }

    void GenerateBars(eavlSceneRenderer *r)
    {
        bool PointField = (field &&
            field->GetAssociation() == eavlField::ASSOC_POINTS);
        bool CellField = (field &&
            field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            field->GetAssocCellSet() == cellset->GetName());

        if (!CellField && !PointField)
            return;

        double minval = 0;
        if (logarithmic)
            minval = floor(log10(min_data_extents));

        r->SetActiveColor(color);

        r->StartTriangles();

        int ncells = cellset->GetNumCells();
        for (int j=0; j<ncells; j++)
        {
            eavlCell cell = cellset->GetCellNodes(j);
            if (cell.type != EAVL_BEAM)
                continue;

            int i0 = cell.indices[0];
            int i1 = cell.indices[1];

            // get vertex coordinates
            double x0 = finalpts[i0*3+0];
            double x1 = finalpts[i1*3+0];

            double gap = (CellField ? 0.1 : 0);

            double w = fabs(x1-x0);
            double g = w * gap / 2.;

            if (CellField)
            {
                double v = field->GetArray()->GetComponentAsDouble(j,0);
                if (logarithmic)
                    v = log10(v);
                r->AddTriangle(x0+g, minval, 0,
                               x1-g, minval, 0,
                               x1-g, v,      0);
                r->AddTriangle(x0+g, minval, 0,
                               x1-g, v,      0,
                               x0+g, v,      0);
            }
            else if (PointField)
            {
                double v0 = field->GetArray()->GetComponentAsDouble(i0,0);
                double v1 = field->GetArray()->GetComponentAsDouble(i1,0);
                if (logarithmic)
                {
                    v0 = log10(v0);
                    v1 = log10(v1);
                }
                r->AddTriangle(x0+g, minval, 0,
                               x1-g, minval, 0,
                               x1-g, v1,     0);
                r->AddTriangle(x0+g, minval, 0,
                               x1-g, v1,     0,
                               x0+g, v0,     0);
            }
        }

        r->EndTriangles();
    }
};


#endif
