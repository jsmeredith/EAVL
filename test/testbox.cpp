#include "eavlCUDA.h"
#include "eavlImporterFactory.h"
#include "eavlBoxMutator.h"
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
 

int main(int argc, char *argv[])
{
    try
    {   
        eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);
        eavlInitializeGPU();

        // usage
        // testbox infile min maxx [outfile]
        // testbox infile min maxx miny maxy [outfile]
        // testbox infile min maxx miny maxy minz maxz [outfile]

        int dim = 1;
        double minx=-1, miny=-1, minz=-1;
        double maxx=+1, maxy=+1, maxz=+1;
        string outfile = "";

        if (argc == 4 || argc == 5)
        {
            minx = strtod(argv[2],NULL);
            maxx = strtod(argv[3],NULL);
            if (argc==5)
                outfile = argv[4];
            dim = 1;
        }
        else if (argc == 6 || argc == 7)
        {
            minx = strtod(argv[2],NULL);
            maxx = strtod(argv[3],NULL);
            miny = strtod(argv[4],NULL);
            maxy = strtod(argv[5],NULL);
            if (argc==7)
                outfile = argv[6];
            dim = 2;
        }
        else if (argc == 8 || argc == 9)
        {
            minx = strtod(argv[2],NULL);
            maxx = strtod(argv[3],NULL);
            miny = strtod(argv[4],NULL);
            maxy = strtod(argv[5],NULL);
            minz = strtod(argv[6],NULL);
            maxz = strtod(argv[7],NULL);
            if (argc==9)
                outfile = argv[8];
            dim = 3;
        }
        else
            THROW(eavlException,"Incorrect number of arguments");

        // read the data file (argv[1]) 
        eavlDataSet *dataset = ReadWholeFile(argv[1]);
        cerr << "Finished reading input file.\n";

        eavlBoxMutator *box = new eavlBoxMutator();
        box->SetDataSet(dataset);
        box->SetCellSet(dataset->GetCellSet(0)->GetName()); // assume first cell set
        if (dim == 3)
        {
            box->SetRange3D(minx, maxx,
                           miny, maxy,
                           minz, maxz);
        }
        else if (dim == 2)
        {
            box->SetRange2D(minx, maxx,
                           miny, maxy);
        }
        else if (dim == 1)
        {
            box->SetRange1D(minx, maxx);
        }
        else
        {
            THROW(eavlException,"Unexpected dimension");
        }

        box->Execute();

        if (outfile != "")
        {
            cerr << "\n\n-- done with box, writing to file --\n";	
            WriteToVTKFile(dataset, outfile, dataset->GetNumCellSets()-1);
        }
        else
        {
            cerr << "No output filename given; not writing dataset\n";
        }

        // print the result
        dataset->PrintSummary(cout);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" infile min maxx [outfile]\n";
        cerr << "\nUsage: "<<argv[0]<<" infile min maxx miny maxy [outfile]\n";
        cerr << "\nUsage: "<<argv[0]<<" infile min maxx miny maxy minz maxz [outfile]\n";
        return 1;
    }

    return 0;
}
