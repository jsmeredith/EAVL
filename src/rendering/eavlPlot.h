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
    string         colortablename;
    eavlColorTable *colortable;
    bool           wireframe;
    eavlColor      color;

  public:
    eavlPlot(eavlDataSet *ds,
             const string &csname = "")
        : dataset(ds), cellsetname(csname), cellset(NULL), normals(NULL)
    {
        /// initializer for other stuff
        field = NULL;
        wireframe = false;
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

    void SetTransformField(int dim)
    {
        if (finalpts == origpts)
            finalpts = new double[npts*3];
        min_coord_extents_final[0] = min_coord_extents_final[1] = min_coord_extents_final[2] = +DBL_MAX;
        max_coord_extents_final[0] = max_coord_extents_final[1] = max_coord_extents_final[2] = -DBL_MAX;
        if (!field)
        {
            THROW(eavlException, "Can't call SetTransformField without "
                  "an active field.");
        }
        ///\todo: yeah, we really need this pretty urgently.
        if (!field_nodal)
        {
            THROW(eavlException, "Can't yet call SetTransformField on "
                  "a cell-centered field.");
        }
        for (int i=0; i<npts; i++)
        {
            double x,y,z;
            finalpts[3*i+0] = origpts[3*i+0];
            finalpts[3*i+1] = origpts[3*i+1];
            finalpts[3*i+2] = origpts[3*i+2];

            finalpts[3*i + dim] = field->GetArray()->GetComponentAsDouble(i,0);

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
    void SetColorTableName(string ct)
    {
        colortablename = ct;
        colortable = new eavlColorTable(colortablename);
    }
    string GetColorTableName()
    {
        return colortablename;
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

    virtual void Render(eavlSceneRenderer *r)
    {
        ColorByOptions opts;
        opts.singleColor = (field == NULL);
        opts.color = color;
        opts.field = field;
        opts.vmin = min_data_extents;
        opts.vmax = max_data_extents;
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
                if (cellset->GetDimensionality() == 1)
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
};

#endif
