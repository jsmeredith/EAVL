// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
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


INSERTING ERROR HERE TO TEST REGRESSION TEST

eavlDataSet *ReadWholeFile(const string &filename)
{
    eavlImporter *importer = eavlImporterFactory::GetImporterForFile(filename);
    
    if (!importer)
        THROW(eavlException,"Didn't determine proper file reader to use");

    string mesh = importer->GetMeshList()[0];
    eavlDataSet *out = importer->GetMesh(mesh, 0);
    vector<string> allvars = importer->GetFieldList(mesh);
    for (size_t i=0; i<allvars.size(); i++)
        out->AddField(importer->GetField(allvars[i], mesh, 0));

    return out;
}
 
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

int main(int argc, char *argv[])
{
    try
    {   
        eavlInitializeGPU();

        if (argc != 4 && argc != 5)
            THROW(eavlException,"Incorrect number of arguments");

        char *tmp;
        double value = strtod(argv[1], &tmp);
        if (tmp == argv[1])
            THROW(eavlException,"Expected a value for first argument");

        const char *fieldname = argv[2];
    
        // Read the input
        eavlDataSet *data = ReadWholeFile(argv[3]);
        //cout << "\n\n-- summary of data set input --\n";	
        //data->PrintSummary(cout);

        int cellsetindex = -1;
        for (int i=0; i<data->GetNumCellSets(); i++)
        {
            if (data->GetCellSet(i)->GetDimensionality() == 1 ||
                data->GetCellSet(i)->GetDimensionality() == 2 ||
                data->GetCellSet(i)->GetDimensionality() == 3)
            {
                cellsetindex = i;
                cerr << "Found 1D, 2D or 3D topo dim cell set name '"
                     << data->GetCellSet(i)->GetName()
                     << "' index " << cellsetindex << endl;
                break;
            }
        }
        if (cellsetindex < 0)
            THROW(eavlException,"Couldn't find a 1D, 2D or 3D cell set.  Aborting.");

        cerr << "\n\n-- isosurfacing --\n";
        eavlIsosurfaceFilter *iso = new eavlIsosurfaceFilter;
        iso->SetInput(data);
        iso->SetCellSet(data->GetCellSet(cellsetindex)->GetName());
        iso->SetField(fieldname);
        iso->SetIsoValue(value);
        int th = eavlTimer::Start();
        iso->Execute();
        cerr << "TOTAL RUNTIME: "<<eavlTimer::Stop(th,"whole isosurface")<<endl;
        //iso->GetOutput()->Clear();
        //int th2 = eavlTimer::Start();
        //iso->Execute();
        //cerr << "SECOND PASS (AFTER INIT): "<<eavlTimer::Stop(th2,"whole isosurface")<<endl;

        //eavlTimer::Dump(cerr);

        // For debugging we can add some temp arrays to input data set
        //WriteToVTKFile(data, "input_modified.vtk", 0);

        if (argc == 5)
        {
            cerr << "\n\n-- done isosurfacing, writing to file --\n";	
            WriteToVTKFile(iso->GetOutput(), argv[4], 0);
        }
        else
        {
            cerr << "No output filename given; not writing result\n";
        }


        cout << "\n\n-- summary of data set result --\n";	
        iso->GetOutput()->PrintSummary(cout);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <value> <fieldname> <infile.vtk> [<outfile.vtk>]\n";
        return 1;
    }


    return 0;
}
