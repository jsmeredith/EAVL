#include "eavlCUDA.h"
#include "eavlImporterFactory.h"
#include "eavlPointDistanceFieldFilter.h"
#include "eavlExecutor.h"
#include "eavlVTKExporter.h"
#include "eavlImporterFactory.h"

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
        eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);
        eavlInitializeGPU();

        if (argc != 2 && argc != 3)
            THROW(eavlException,"Incorrect number of arguments");

        // assume first mesh, single (or first) domain
        int meshindex = 0;
        int domainindex = 0;

        // read the data file (argv[1]) 
        eavlImporter *importer = eavlImporterFactory::GetImporterForFile(argv[1]);
        string meshname = importer->GetMeshList()[meshindex];
        eavlDataSet *input = importer->GetMesh(meshname, domainindex);

        eavlPointDistanceFieldFilter *df = new eavlPointDistanceFieldFilter();
        df->SetInput(input);
        //df->SetRange1D(20,
        //               -10,10);
        //df->SetRange2D(20,20,
        //               -10,10,
        //               -10,10);
        df->SetRange3D(40,40,40,
                       20,40,
                       -70,-40,
                       0,25);
        df->Execute();

        eavlDataSet *result = df->GetOutput();

        if (argc == 3)
        {
            cerr << "\n\n-- done with distance field, writing to file --\n";	
            WriteToVTKFile(result, argv[2], 0);
        }
        else
        {
            cerr << "No output filename given; not writing result\n";
        }

        // print the result
        result->PrintSummary(cout);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <file> <field> [numbins]\n";
        return 1;
    }

    return 0;
}
