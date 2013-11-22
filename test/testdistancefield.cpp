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

        // usage
        // testdistancefield infile nx min maxx [outfile]
        // testdistancefield infile nx ny min maxx miny maxy [outfile]
        // testdistancefield infile nx ny nz min maxx miny maxy minz maxz [outfile]

        int nx=1, ny=1, nz=1;
        double minx=-1, miny=-1, minz=-1;
        double maxx=+1, maxy=+1, maxz=+1;
        string outfile = "";

        if (argc == 5 || argc == 6)
        {
            nx = strtol(argv[2],NULL,10);
            minx = strtod(argv[3],NULL);
            maxx = strtod(argv[4],NULL);
            if (argc==6)
                outfile = argv[5];
        }
        else if (argc == 8 || argc == 9)
        {
            nx = strtol(argv[2],NULL,10);
            ny = strtol(argv[3],NULL,10);
            minx = strtod(argv[4],NULL);
            maxx = strtod(argv[5],NULL);
            miny = strtod(argv[6],NULL);
            maxy = strtod(argv[7],NULL);
            if (argc==9)
                outfile = argv[8];
        }
        else if (argc == 11 || argc == 12)
        {
            nx = strtol(argv[2],NULL,10);
            ny = strtol(argv[3],NULL,10);
            nz = strtol(argv[4],NULL,10);
            minx = strtod(argv[5],NULL);
            maxx = strtod(argv[6],NULL);
            miny = strtod(argv[7],NULL);
            maxy = strtod(argv[8],NULL);
            minz = strtod(argv[9],NULL);
            maxz = strtod(argv[10],NULL);
            if (argc==12)
                outfile = argv[11];
        }
        else
            THROW(eavlException,"Incorrect number of arguments");

        // assume first mesh, single (or first) domain
        int meshindex = 0;
        int domainindex = 0;

        // read the data file (argv[1]) 
        eavlImporter *importer = eavlImporterFactory::GetImporterForFile(argv[1]);
        string meshname = importer->GetMeshList()[meshindex];
        eavlDataSet *input = importer->GetMesh(meshname, domainindex);

        cerr << "Finished reading input file.\n";

        eavlPointDistanceFieldFilter *df = new eavlPointDistanceFieldFilter();
        df->SetInput(input);
        if (nz > 1)
        {
            df->SetRange3D(nx, ny, nz,
                           minx, maxx,
                           miny, maxy,
                           minz, maxz);
        }
        else if (ny > 1)
        {
            df->SetRange2D(nx, ny,
                           minx, maxx,
                           miny, maxy);
        }
        else
        {
            df->SetRange1D(nx,
                           minx, maxx);
        }
        df->Execute();

        eavlDataSet *result = df->GetOutput();

        if (outfile != "")
        {
            cerr << "\n\n-- done with distance field, writing to file --\n";	
            WriteToVTKFile(result, outfile, 0);
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
        cerr << "\nUsage: "<<argv[0]<<" infile nx min maxx [outfile]\n";
        cerr << "\nUsage: "<<argv[0]<<" infile nx ny min maxx miny maxy [outfile]\n";
        cerr << "\nUsage: "<<argv[0]<<" infile nx ny nz min maxx miny maxy minz maxz [outfile]\n";
        return 1;
    }

    return 0;
}
