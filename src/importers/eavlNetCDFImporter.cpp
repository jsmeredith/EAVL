// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlNetCDFImporter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetAllStructured.h"
#include "eavlException.h"

static const bool debugoutput = false;

eavlNetCDFImporter::eavlNetCDFImporter(const string &filename)
{
    file = new NcFile(filename.c_str(), NcFile::ReadOnly);
     
    if (!file->is_valid())
    {
        THROW(eavlException,"Couldn't open file!\n");
    }

    if (debugoutput) cerr << "num_dims="<<file->num_dims()<<endl;
    if (debugoutput) cerr << "num_vars="<<file->num_vars()<<endl;
    if (debugoutput) cerr << "num_atts="<<file->num_atts()<<endl;

    for (int i=0; i<file->num_dims(); i++)
    {
        NcDim *d = file->get_dim(i);
        if (debugoutput) cerr << "  dim["<<i<<"]: name="<<d->name()<<" size="<<d->size()<<endl;
    }

    for (int i=0; i<file->num_atts(); i++)
    {
        NcAtt *a = file->get_att(i);
        if (debugoutput) cerr << "  att["<<i<<"]: name="<<a->name()<<" numvals="<<a->num_vals()<<endl;
    }

    bool found_grid = false;

    for (int i=0; i<file->num_vars(); i++)
    {
        NcVar *v = file->get_var(i);
        if (debugoutput) 
        {
            cerr << "  var["<<i<<"]: name="<<v->name();
            cerr << "  ndims="<<v->num_dims();
            cerr << "  dims = ";
            for (int j=0; j<v->num_dims(); j++)
            {
                cerr << v->get_dim(j)->name();
                if (j<v->num_dims()-1)
                    cerr << "*";
            }
            cerr << endl;
        }

        // Here's the condition for what we're going to use;
        // we only support one mesh for the moment, so we're picking one.
        // Also, the netcdf files we have have the time dim size as "1"
        if (v->num_dims() == 4 && string(v->get_dim(0)->name())=="time")
        {
            if (!found_grid)
            {
                dims.push_back(v->get_dim(1));
                dims.push_back(v->get_dim(2));
                dims.push_back(v->get_dim(3));
                found_grid = true;
                vars.push_back(v);
                if (debugoutput) cerr << "     * using as first real var\n";
            }
            else
            {
                if (string(v->get_dim(1)->name()) == dims[0]->name() &&
                    string(v->get_dim(2)->name()) == dims[1]->name() &&
                    string(v->get_dim(3)->name()) == dims[2]->name())
                {
                    vars.push_back(v);
                    if (debugoutput) cerr << "     * using as another var; matches the first real one's dims\n";
                }
            }
        }

    }
}


eavlNetCDFImporter::~eavlNetCDFImporter()
{
    file->close();
}

eavlDataSet*
eavlNetCDFImporter::GetMesh(const string &mesh, int)
{
    // NOTE: the data ordering isn't what we expected; for the moment
    // we've swapped X, Y, and Z to some degree, but this is a good use
    // to make sure we're doing it "right".

    eavlDataSet *data = new eavlDataSet;

    vector<vector<double> > coords;
    vector<string> coordNames;

    coordNames.push_back(dims[2]->name());
    {
        vector<double> c;
        int nc = dims[2]->size();
        c.resize(nc);
        for (int i = 0; i < nc; i++)
            c[i] = (double)i;
        coords.push_back(c);
    }
    
    coordNames.push_back(dims[1]->name());
    {
        vector<double> c;
        int nc = dims[1]->size();
        c.resize(nc);
        for (int i = 0; i < nc; i++)
            c[i] = (double)i;
        coords.push_back(c);
    }
    coordNames.push_back(dims[0]->name());
    {
        vector<double> c;
        int nc = dims[0]->size();
        c.resize(nc);
        for (int i = 0; i < nc; i++)
            c[i] = (double)i;
        coords.push_back(c);
    }
    
    AddRectilinearMesh(data, coords, coordNames, true, "RectilinearGridCells");

    return data;
}

vector<string>
eavlNetCDFImporter::GetFieldList(const string &mesh)
{
    vector<string> retval;
    for (unsigned int v=0; v<vars.size(); v++)
    {
        NcVar *var = vars[v];
        retval.push_back(var->name());
    }
    return retval;
}

eavlField*
eavlNetCDFImporter::GetField(const string &name, const string &mesh, int)
{
    for (unsigned int v=0; v<vars.size(); v++)
    {
        NcVar *var = vars[v];
        if (name != var->name())
            continue;

        if (debugoutput) cerr << "reading var "<<v+1<<" / "<<vars.size()<<endl;
        eavlFloatArray *arr = new eavlFloatArray(var->name(), 1);
        arr->SetNumberOfTuples(var->num_vals());
        NcValues *vals = var->values();
        int n = var->num_vals();
        for (int i=0; i<n; i++)
        {
            arr->SetComponentFromDouble(i,0, vals->as_double(i));
        }

        eavlField *field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        return field;
    }

    return NULL;
}
