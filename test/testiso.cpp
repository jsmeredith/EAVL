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

#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlCoordinates.h"
#include "eavlLogicalStructureRegular.h"


eavlDataSet *GenerateRectXY(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension2D(ni, nj);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, float(i)/float(ni-1));

    eavlFloatArray *y = new eavlFloatArray("y", 1, nj);
    for (int j=0; j<nj; ++j)
        y->SetValue(j, float(j)/float(nj-1));

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));

    // create a radial nodal field
    eavlFloatArray *rad = new eavlFloatArray("rad", 1, npts);
    for (int j=0; j<nj; ++j)
        for (int i=0; i<ni; ++i)
            rad->SetValue(j*ni + i, i*i + j*j);
    data->AddField(new eavlField(1,rad,eavlField::ASSOC_POINTS));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    data->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    data->AddCellSet(cells);

    return data;
}


eavlDataSet *GenerateRectXYZ(int ni, int nj, int nk)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj * nk;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension3D(ni, nj, nk);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, float(i)/float(ni-1));

    eavlFloatArray *y = new eavlFloatArray("y", 1, nj);
    for (int j=0; j<nj; ++j)
        y->SetValue(j, float(j)/float(nj-1));

    eavlFloatArray *z = new eavlFloatArray("z", 1, nk);
    for (int k=0; k<nk; ++k)
        z->SetValue(k, float(k)/float(nk-1));

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));
    data->AddField(new eavlField(1, z, eavlField::ASSOC_LOGICALDIM, 2));

    // create a radial nodal field
    eavlFloatArray *rad = new eavlFloatArray("rad", 1, npts);
    for (int k=0; k<nk; ++k)
        for (int j=0; j<nj; ++j)
            for (int i=0; i<ni; ++i)
                rad->SetValue(k*ni*nj + j*ni + i, i*i + j*j + k*k);
    data->AddField(new eavlField(1,rad,eavlField::ASSOC_POINTS));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y,
                                            eavlCoordinatesCartesian::Z);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    data->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    data->AddCellSet(cells);

    return data;
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

        if (argc != 4 && argc != 5 &&
            argc != 6 && argc != 7)
            THROW(eavlException,"Incorrect number of arguments");

        char *tmp;
        double value = strtod(argv[1], &tmp);
        if (tmp == argv[1])
            THROW(eavlException,"Expected a value for first argument");

        const char *outfile = NULL;

        if (argc == 5)
            outfile = argv[4];
        else if (argc == 7)
            outfile = argv[6];

        const char *fieldname = argv[2];
    
        // Read the input
        eavlDataSet *data = NULL;
        if (argc == 4 || argc == 5)
        {
            data = ReadWholeFile(argv[3]);
        }
        else
        {
            int ni = atoi(argv[3]);
            int nj = atoi(argv[4]);
            int nk = atoi(argv[5]);
            if (ni<2 || nj<2 || nk<2)
                THROW(eavlException, "Expected ni,nj,nk >= 2");
            //eavlDataSet *data = GenerateRectXY(ni,nj);
            cerr << "Forcing field name to 'rad'" << endl;
            fieldname = "rad";
            data = GenerateRectXYZ(ni,nj,nk);
        }

        //cout << "\n\n-- summary of data set input --\n";	
        //data->PrintSummary(cout);

        //WriteToVTKFile(data, "test.vtk", 0);

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

        if (outfile)
        {
            cerr << "\n\n-- done isosurfacing, writing to file --\n";	
            WriteToVTKFile(iso->GetOutput(), outfile, 0);
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
        cerr << "        "<<argv[0]<<" <value> <fieldname> <nx> <ny> <nz> [<outfile.vtk>]\n";
        return 1;
    }


    return 0;
}
