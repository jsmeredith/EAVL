// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"
#include "eavlVTKExporter.h"

#include "eavlExternalFaceMutator.h"
#include "eavlSurfaceNormalMutator.h"
#include "eavlCellToNodeRecenterMutator.h"
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

        int cellsetindex = -1;
        for (int i=0; i<data->GetNumCellSets(); i++)
        {
            if (data->GetCellSet(i)->GetDimensionality() == 2)
            {
                cellsetindex = i;
                cerr << "Found 2D topo dim cell set name '"
                     << data->GetCellSet(i)->GetName()
                     << "' index " << cellsetindex << endl;
                break;
            }
        }
        if (cellsetindex < 0)
        {
            cerr << "Couldn't find a 2D cell set.  Trying to add external faces to a 3D set.\n";
            int cellsetindex3d = -1;
            for (int i=0; i<data->GetNumCellSets(); i++)
            {
                if (data->GetCellSet(i)->GetDimensionality() == 3)
                {
                    cellsetindex3d = i;
                    cerr << "Found 3D topo dim cell set index "<<cellsetindex3d<<endl;
                    break;
                }
            }
            if (cellsetindex3d < 0)
                THROW(eavlException,"Couldn't find a 3D set to add faces, to either.  Aborting.");

            eavlExternalFaceMutator *extface = new eavlExternalFaceMutator;
            extface->SetDataSet(data);
            extface->SetCellSet(data->GetCellSet(cellsetindex3d)->GetName());
            extface->Execute();
            delete extface;
            cellsetindex = data->GetNumCellSets() - 1;
        }

        if (data->GetCoordinateSystem(0)->GetDimension() != 3)
            THROW(eavlException,"Not 3D coords.  Want 3D coords for now.\n");

        eavlSurfaceNormalMutator *surfnorm = new eavlSurfaceNormalMutator;
        surfnorm->SetDataSet(data);
        surfnorm->SetCellSet(data->GetCellSet(cellsetindex)->GetName());
        surfnorm->Execute();

        eavlCellToNodeRecenterMutator *cell2node = new eavlCellToNodeRecenterMutator;
        cell2node->SetDataSet(data);
        cell2node->SetField("surface_normals");
        cell2node->SetCellSet(data->GetCellSet(cellsetindex)->GetName());
        cell2node->Execute();

        if (argc == 3)
        {
            cerr << "\n\n-- done with surface normal, writing to file --\n";	
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
        cerr << "\nUsage: "<<argv[0]<<" <infile.vtk> [<outfile.vtk>]\n";
        return 1;
    }


    return 0;
}
