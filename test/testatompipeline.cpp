// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"
#include "eavlLAMMPSDumpImporter.h"
#include "eavlVTKExporter.h"

#include "eavlIsosurfaceFilter.h"
#include "eavlPointDistanceFieldFilter.h"
#include "eavlThresholdMutator.h"
#include "eavlBoxMutator.h"

//
// GLOBAL SETTING
//

// if you're reading from a lammps file, you need to tell
// the helper function which species type you're working with.
// Use zero-oriign.  I.e., it's either 0 (W) or 1 (He).
int matchtype = 1;

void WriteToVTKFile(eavlDataSet *data, const string &filename,
        int cellSetIndex = 0)
{
    ofstream *p = new ofstream(filename.c_str());
    ostream *s = p;
    eavlVTKExporter exporter(data, cellSetIndex);
    exporter.Export(*s);
    p->close();
    delete p;
}

eavlDataSet *ReadMeshFromFile(const string &filename, int meshindex)
{
    //eavlImporter *importer = eavlImporterFactory::GetImporterForFile(filename);
    eavlImporter *importer = new eavlLAMMPSDumpImporter(filename);
    
    if (!importer)
        THROW(eavlException,"Didn't determine proper file reader to use");

    vector<string> allmeshes = importer->GetMeshList();
    if (meshindex >= (int)allmeshes.size())
        return NULL;

    string meshname = allmeshes[meshindex];
    // always read the first domain for now
    eavlDataSet *out = importer->GetMesh(meshname, 0);
    vector<string> allvars = importer->GetFieldList(meshname);
    for (size_t i=0; i<allvars.size(); i++)
        out->AddField(importer->GetField(allvars[i], meshname, 0));

    delete importer;
    return out;
}

vector<double> x, y, z;

void FillXYZArraysFromFile(const std::string &filename)
{
    int meshindex = 0;
    eavlDataSet *data = ReadMeshFromFile(filename, meshindex);

    int npts = data->GetNumPoints();
    eavlField *type = data->GetField("type");

    for (int i=0; i<npts; ++i)
    {
        if (type->GetArray()->GetComponentAsDouble(i, 0) == matchtype)
        {
            x.push_back(data->GetPoint(i, 0));
            y.push_back(data->GetPoint(i, 1));
            z.push_back(data->GetPoint(i, 2));
        }
    }
    delete data;

    cerr << "USING " << x.size() << " ATOMS OUT OF "<<npts<<" IN THE FILE\n";

}

eavlDataSet *CreatePointDataSet(const vector<double> &x,
                                const vector<double> &y,
                                const vector<double> &z)
{
    int n = x.size();
   
    eavlDataSet *data = new eavlDataSet;

    // Create the coordinates structure
    data->SetNumPoints(n);
    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    data->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("xcoord",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("ycoord",0));
    coords->SetAxis(2,new eavlCoordinateAxisField("zcoord",0));

    // Create the coordinates values
    eavlFloatArray *axisValues[3] = {
        new eavlFloatArray("xcoord",1, n),
        new eavlFloatArray("ycoord",1, n),
        new eavlFloatArray("zcoord",1, n)
    };
    for (int i=0; i<n; i++)
    {
        axisValues[0]->SetValue(i, x[i]);
        axisValues[1]->SetValue(i, y[i]);
        axisValues[2]->SetValue(i, z[i]);
    }

    // Add the coordinates values to the mesh as new fields
    for (int d=0; d<3; d++)
        data->AddField(new eavlField(1, axisValues[d], eavlField::ASSOC_POINTS));

    return data;
}

void DoIt(const vector<double> &x,
          const vector<double> &y,
          const vector<double> &z)
{
    //
    // Settings
    //
    bool debug = false;

    // settings for step 2: bounds and number of cells for distance field?
    double xmin =  0, xmax = 499;
    double ymin =  0, ymax = 503;
    double zmin = -4, zmax = 160;
    int nx = 150;
    int ny = 150;
    int nz = 70;

    // settings for step 3: how far is the cutoff distance from each atom?
    double cutoffdist = 2.0;

    // settings for step 4: should we restrict the result
    //                      only to triangles within some spatial range?
    bool apply_box = false;
    double box_xmin =  20, box_xmax = 490;
    double box_ymin =  20, box_ymax = 490;
    double box_zmin =  15, box_zmax = 145;

    //
    // Process
    //

    // step 1: create a simple eavl data set
    cerr << "READING FILE\n";
    eavlDataSet *data = CreatePointDataSet(x,y,z);
    if (debug)
        data->PrintSummary(cout);

    // step 2: create a new distance field
    cerr << "CREATING DISTANCE FILTER\n";
    eavlPointDistanceFieldFilter *df = new eavlPointDistanceFieldFilter();
    df->SetInput(data);
    df->SetRange3D(nx, ny, nz, 
                   xmin, xmax,
                   ymin, ymax, 
                   zmin, zmax);
    df->Execute();

    eavlDataSet *distance = df->GetOutput();
    delete data;
    delete df;

    if (debug)
        distance->PrintSummary(cout);
    //WriteToVTKFile(distance, "distance.vtk");

    // step 3: isosurface the distance field
    cerr << "ISOSURFACING\n";
    eavlIsosurfaceFilter *iso = new eavlIsosurfaceFilter;
    iso->SetInput(distance);
    iso->SetCellSet("cells");
    iso->SetField("dist");
    iso->SetIsoValue(cutoffdist);
    iso->Execute();

    eavlDataSet *triangles = iso->GetOutput();
    delete distance;
    delete iso;
    if (debug)
        triangles->PrintSummary(cout);

    // step 4 (optional):
    if (!apply_box)
    {
        cerr << "WRITING RESULT\n";
        WriteToVTKFile(triangles, "all_triangles.vtk", 0);
    }
    else
    {
        cerr << "APPLYING BOX SELECTION\n";
        eavlBoxMutator *box = new eavlBoxMutator();
        box->SetDataSet(triangles);
        box->SetCellSet(triangles->GetCellSet(0)->GetName());
        box->SetRange3D(box_xmin, box_xmax,
                        box_ymin, box_ymax,
                        box_zmin, box_zmax);
        box->Execute();
        cerr << "WRITING RESULT\n";
        WriteToVTKFile(triangles, "box_triangles.vtk", 1);
        delete box;
    }
    delete triangles;
}
 
int main(int argc, char *argv[])
{
    try
    {   
        if (argc != 2)
            THROW(eavlException,"Incorrect number of arguments");

        // Fill the global x, y, and z arrays with atom positions
        FillXYZArraysFromFile(argv[1]);
        
        DoIt(x, y, z);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile>\n";
        return 1;
    }


    return 0;
}
