// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"
#include "eavlVTKExporter.h"

#include "eavlIsosurfaceFilter.h"
#include "eavlExecutor.h"

#include "eavlVTKDataSet.h"

#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>

eavlDataSet *ReadMeshFromFile(const string &filename, int meshindex)
{
    eavlImporter *importer = eavlImporterFactory::GetImporterForFile(filename);
    
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

    return out;
}

vtkDataSet *ReadVTKMeshFromFile(const string &filename)
{
     vtkDataSetReader *rdr = vtkDataSetReader::New();
     rdr->SetFileName(filename.c_str());
     vtkDataSet *out = rdr->GetOutput();
     rdr->Update();
     out->Register(NULL);
     rdr->Delete();
     return out;
}

void PrintVTKObjectSummary(ostream &out, vtkDataSet *ds)
{
    out << ""<<ds->GetClassName()<<":"<<endl;
    vtkRectilinearGrid *rgrid = dynamic_cast<vtkRectilinearGrid*>(ds);
    vtkStructuredGrid *sgrid = dynamic_cast<vtkStructuredGrid*>(ds);
    vtkFieldData *pointdata = ds->GetPointData();
    vtkFieldData *celldata = ds->GetCellData();
    vtkFieldData *fielddata = ds->GetFieldData();
    int dims[3];
    if (rgrid)
        rgrid->GetDimensions(dims);
    if (sgrid)
        sgrid->GetDimensions(dims);
    if (rgrid || sgrid)
        out << "  dimensions: " << dims[0] << " " << dims[1] << " " << dims[2] << endl;
    out << "  num cells: "<<ds->GetNumberOfCells()<<endl;
    out << "  num points: "<<ds->GetNumberOfPoints()<<endl;
    out << "  field data: " << fielddata->GetNumberOfArrays() << " array(s)" << endl;
    for (int i=0; fielddata && i<fielddata->GetNumberOfArrays(); ++i)
        out << "    field array "<<i<<": "<< fielddata->GetArrayName(i) << endl;
    out << "  point data: " << pointdata->GetNumberOfArrays() << " array(s)" << endl;
    for (int i=0; pointdata && i<pointdata->GetNumberOfArrays(); ++i)
        out << "    point array "<<i<<": "<< pointdata->GetArrayName(i) << endl;
    out << "  cell data: " << celldata->GetNumberOfArrays() << " array(s)" << endl;
    for (int i=0; celldata && i<celldata->GetNumberOfArrays(); ++i)
        out << "    cell array "<<i<<": "<< celldata->GetArrayName(i) << endl;
}

int main(int argc, char *argv[])
{
    try
    {   
        if (argc != 2)
            THROW(eavlException,"Incorrect number of arguments");

        eavlDataSet *eavldata = ReadMeshFromFile(argv[1], 0);
        vtkDataSet  *vtkdata = ReadVTKMeshFromFile(argv[1]);

        cout << "Read " << eavldata->GetCellSet(0)->GetNumCells() << " cells through EAVL reader\n";
        cout << "Read " << vtkdata->GetNumberOfCells() << " cells through VTK reader\n";

        eavlDataSet *eavldata_from_vtk = ConvertVTKToEAVL(vtkdata);
        vtkDataSet *vtkdata_from_eavl = ConvertEAVLToVTK(eavldata);

        cout << "EAVL converted from VTK has " << eavldata_from_vtk->GetCellSet(0)->GetNumCells() << " cells\n";
        cout << "VTK converted from EAVL has " << vtkdata_from_eavl->GetNumberOfCells() << " cells\n";

        cout << ">> eavl data as read from original file <<\n";
        eavldata->PrintSummary(cout);
        cout << endl;
        cout << ">> eavl data as converted from true vtk read of original file <<\n";
        eavldata_from_vtk->PrintSummary(cout);
        cout << endl;

        cout << ">> vtk data as true vtk read of original file <<\n";
        PrintVTKObjectSummary(cout, vtkdata);
        cout << endl;
        cout << ">> vtk data as converted from eavl read of original file <<\n";
        PrintVTKObjectSummary(cout, vtkdata_from_eavl);
        cout << endl;

    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile>\n";
        return 1;
    }


    return 0;
}
