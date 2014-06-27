// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"
#include "eavlVTKExporter.h"

#include "eavlTransformMutator.h"
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

        if (argc != 2 && argc != 3)
            THROW(eavlException,"Incorrect number of arguments");

        // Read the input
        eavlDataSet *data = ReadWholeFile(argv[1]);

        int cellsetindex = 0;

        eavlMatrix4x4 m1, m2, m3, m;
        m1.CreateRotateX(0.2);
        m2.CreateRotateY(0.4);
        m3.CreateScale(0.8, 1.4, 1.0);
        m = m1 * m2 * m3;

        eavlTransformMutator *xform = new eavlTransformMutator;
        xform->SetDataSet(data);
        xform->SetCoordinateSystemIndex(0);
        xform->SetTransformCoordinates(true);
        xform->SetTransform(m);
        xform->Execute();

        if (argc == 3)
        {
            cerr << "\n\n-- done with transform, writing to file --\n";	
            WriteToVTKFile(data, argv[2], cellsetindex);
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
        cerr << "\nUsage: "<<argv[0]<<" <value> <fieldname> <infile.vtk> [<outfile.vtk>]\n";
        return 1;
    }


    return 0;
}
