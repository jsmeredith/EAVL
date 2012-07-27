// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
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


eavlDataSet *ReadWholeFile(const string &filename)
{
    eavlImporter *importer = eavlImporterFactory::GetImporterForFile(filename);
    
    if (!importer)
        THROW(eavlException,"Didn't determine proper file reader to use");

    eavlDataSet *out = importer->GetMesh(0);
    vector<string> allvars = importer->GetFieldList();
    for (int i=0; i<allvars.size(); i++)
        out->fields.push_back(importer->GetField(0, allvars[i]));

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
        for (int i=0; i<data->cellsets.size(); i++)
        {
            if (data->cellsets[i]->GetDimensionality() == 3)
            {
                cellsetindex = i;
                cerr << "Found 3D topo dim cell set name '"
                     << data->cellsets[i]->GetName()
                     << "' index " << cellsetindex << endl;
                break;
            }
        }
        if (cellsetindex < 0)
            THROW(eavlException,"Couldn't find a 3D cell set.  Aborting.");

        if (data->coordinateSystems[0]->GetDimension() != 3)
            THROW(eavlException,"Not 3D coords.  Want 3D coords for now.\n");

        cerr << "\n\n-- isosurfacing --\n";
        eavlIsosurfaceFilter *iso = new eavlIsosurfaceFilter;
        iso->SetInput(data);
        iso->SetCellSet(data->cellsets[cellsetindex]->GetName());
        iso->SetField(fieldname);
        iso->SetIsoValue(value);
        int th = eavlTimer::Start();
        iso->Execute();
        cerr << "TOTAL RUNTIME: "<<eavlTimer::Stop(th,"whole isosurface")<<endl;
        iso->GetOutput()->fields.clear();
        iso->GetOutput()->coordinateSystems.clear();
        iso->GetOutput()->cellsets.clear();
        int th2 = eavlTimer::Start();
        iso->Execute();
        cerr << "SECOND PASS (AFTER INIT): "<<eavlTimer::Stop(th2,"whole isosurface")<<endl;

        eavlTimer::Dump(cerr);

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
