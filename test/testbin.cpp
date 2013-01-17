#include "eavlCUDA.h"
#include "eavlImporterFactory.h"
#include "eavlScalarBinFilter.h"
#include "eavlExecutor.h"

int main(int argc, char *argv[])
{
    try
    {   
        eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);
        eavlInitializeGPU();

        if (argc != 3 && argc != 4)
            THROW(eavlException,"Incorrect number of arguments");

        int nbins = 0;
        if (argc == 4)
            nbins = atoi(argv[3]);
        if (nbins == 0)
            nbins = 10;

        // assume first mesh, single (or first) domain
        int meshindex = 0;
        int domainindex = 0;

        // read the data file (argv[1]) and the given field (argv[2])
        eavlImporter *importer = eavlImporterFactory::GetImporterForFile(argv[1]);
        string meshname = importer->GetMeshList()[meshindex];
        eavlDataSet *data = importer->GetMesh(meshname, domainindex);
        data->AddField(importer->GetField(argv[2], meshname, domainindex));

        // bin given scalar field (argv[2])
        eavlScalarBinFilter *scalarbin = new eavlScalarBinFilter();
        scalarbin->SetInput(data);
        scalarbin->SetNumBins(nbins);
        scalarbin->SetField(argv[2]);
        scalarbin->Execute();

        eavlDataSet *result = scalarbin->GetOutput();

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
