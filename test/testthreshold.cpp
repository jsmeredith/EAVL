// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"
#include "eavlVTKExporter.h"

#include "eavlThresholdMutator.h"
#include "eavlExecutor.h"


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
        eavlExecutor::SetExecutionMode(eavlExecutor::PreferGPU);
        eavlInitializeGPU();

        if (argc != 5 && argc != 6)
            THROW(eavlException,"Incorrect number of arguments");

        // Read the input
        eavlDataSet *data = ReadWholeFile(argv[1]);

        eavlField *f = data->GetField(argv[2]);
        if (f->GetAssociation() != eavlField::ASSOC_CELL_SET)
            THROW(eavlException, "Wanted a cell-centered field.");

        string cellsetname = f->GetAssocCellSet();
        int cellsetindex = data->GetCellSetIndex(cellsetname);

        eavlThresholdMutator *thresh = new eavlThresholdMutator;
        thresh->SetDataSet(data);
        thresh->SetField(argv[2]);
        thresh->SetRange(strtod(argv[3],NULL), strtod(argv[4],NULL));
        thresh->SetCellSet(data->GetCellSet(cellsetindex)->GetName());
        thresh->Execute();

        if (argc == 6)
        {
            cerr << "\n\n-- done with surface normal, writing to file --\n";	
            WriteToVTKFile(data, argv[5], data->GetNumCellSets()-1);
        }
        else
        {
            cerr << "No output filename given; not writing result\n";
        }


        cout << "\n\n-- summary of data set result --\n";	
        data->PrintSummary(cout);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile.vtk> <fieldname> <low> <hi> [<outfile.vtk>]\n";
        return 1;
    }


    return 0;
}
