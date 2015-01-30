// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlVTKDataSet.h"

#include "eavl.h"
#include "eavlException.h"
#include "eavlVTKImporter.h"
#include "eavlVTKExporter.h"
#include "eavlTimer.h"

#ifdef HAVE_VTK

#include "eavlDataSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"

#include <vtkUnstructuredGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkCellArray.h>
#include <vtkCellTypes.h>
#include <vtkCellData.h>
#include <vtkPointData.h>

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
static vtkDataSet *ConvertEAVLToVTK_Fallback(eavlDataSet *in)
{
    ostringstream stream;
    eavlVTKExporter exporter(in);
    exporter.Export(stream);
    string str = stream.str();

    // Note: VisIt does this: (I ask because we're getting a 1-byte
    // invalid read in valgrind; maybe this fixes it?):
    //vtkCharArray *charArray = vtkCharArray::New();
    //int iOwnIt = 1;  // 1 means we own it -- you don't delete it.
    //charArray->SetArray((char *) asCharTmp, asCharLengthTmp, iOwnIt);
    //reader->SetReadFromInputString(1);
    //reader->SetInputArray(charArray);

    vtkDataSetReader *rdr = vtkDataSetReader::New();
    rdr->ReadAllScalarsOn();
    rdr->ReadAllVectorsOn();
    rdr->ReadAllTensorsOn();
    rdr->SetReadFromInputString(1);
    rdr->SetInputString(str.c_str());

    vtkDataSet *out = rdr->GetOutput();
    rdr->Update();
    out->Register(NULL);
    rdr->Delete();
    return out;
}

vtkDataSet *ConvertEAVLToVTK(eavlDataSet *in)
{
    int th = eavlTimer::Start();
    vtkDataSet *result = NULL;
    result = ConvertEAVLToVTK_Fallback(in);
    eavlTimer::Stop(th, "EAVL -> VTK");
    return result;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
static eavlDataSet *ConvertVTKToEAVL_Fallback(vtkDataSet *in)
{
    vtkDataSetWriter *wrtr = vtkDataSetWriter::New();
    wrtr->WriteToOutputStringOn();
    wrtr->SetFileTypeToBinary();
    wrtr->SetInputData(in);
    wrtr->Write();

    eavlVTKImporter importer(wrtr->GetOutputString(),
                             wrtr->GetOutputStringLength());
    int chunk = 0; // only one domain, of course
    string meshname = "mesh"; // unused for VTK importer; one mesh per file
    eavlDataSet *out = importer.GetMesh(meshname,chunk);
    vector<string> allvars = importer.GetFieldList(meshname);
    for (unsigned int i=0; i<allvars.size(); i++)
        out->AddField(importer.GetField(allvars[i], meshname, chunk));

    wrtr->Delete();
    return out;
}

static eavlDataSet *V2E_Explicit(vtkUnstructuredGrid *ug,
                                 vector<int> &cell_to_cell_splitmap)
{
    eavlDataSet *ds = new eavlDataSet;

    eavlCellSetExplicit *cells[4];
    eavlExplicitConnectivity newconn[4];
    for (int e=0; e<4; e++)
    {
        cells[e] = new eavlCellSetExplicit(string("UnstructuredGridCells")
                                           + char('0'+e) + "D", e);
    }

    vtkCellArray *ug_cells = ug->GetCells();
    vtkIdType npts, *pts, cellId;
    for (cellId=0, ug_cells->InitTraversal();
         ug_cells->GetNextCell(npts,pts);
         cellId++)
    {
        int ug_type = ug->GetCellType(cellId);

        eavlCellShape st = EAVL_OTHER;
        int d = -1;
        switch (ug_type)
        {
          //case VTK_EMPTY_CELL:   d=0; st = EAVL_OTHER??;  break;
          case VTK_VERTEX:         d=0; st = EAVL_POINT;    break;
          //case VTK_POLY_VERTEX   d=0; st = EAVL_OTHER??;  break;
          case VTK_LINE:           d=1; st = EAVL_BEAM;     break;
          //case VTK_POLY_LINE:    d=1; st = EAVL_OTHER??;  break;
          case VTK_TRIANGLE:       d=2; st = EAVL_TRI;      break;
          case VTK_TRIANGLE_STRIP: d=2; st = EAVL_TRISTRIP; break;
          case VTK_POLYGON:        d=2; st = EAVL_POLYGON;  break;
          case VTK_PIXEL:          d=2; st = EAVL_PIXEL;    break;
          case VTK_QUAD:           d=2; st = EAVL_QUAD;     break;
          case VTK_TETRA:          d=3; st = EAVL_TET;      break;
          case VTK_VOXEL:          d=3; st = EAVL_VOXEL;    break;
          case VTK_HEXAHEDRON:     d=3; st = EAVL_HEX;      break;
          case VTK_WEDGE:          d=3; st = EAVL_WEDGE;    break;
          case VTK_PYRAMID:        d=3; st = EAVL_PYRAMID;  break;
        }

        if (d >= 0)
            newconn[d].AddElement(st,  npts, pts);

        // we need to split fields by dimensionality
        cell_to_cell_splitmap.push_back(d);
    }
    for (int e=0; e<4; e++)
    {
        if (newconn[e].GetNumElements() > 0)
        {
            cells[e]->SetCellNodeConnectivity(newconn[e]);
            ds->AddCellSet(cells[e]);
        }
        else
        {
            delete cells[e];
        }
    }

    return ds;
}

static eavlDataSet *V2E_Structured(int *vtkdims)
{
    eavlDataSet *ds = new eavlDataSet;

    int dim = 0;
    int eavldims[3];
    if (vtkdims[0] > 1)
        eavldims[dim++] = vtkdims[0];
    if (vtkdims[1] > 1)
        eavldims[dim++] = vtkdims[1];
    if (vtkdims[2] > 1)
        eavldims[dim++] = vtkdims[2];

    eavlRegularStructure reg; 
    reg.SetNodeDimension(dim, eavldims);
    eavlLogicalStructureRegular *log = new eavlLogicalStructureRegular(dim,reg);
    ds->SetLogicalStructure(log);

    eavlCellSetAllStructured *cells =
        new eavlCellSetAllStructured("StructuredGridCells", reg);
    ds->AddCellSet(cells);

    return ds;
}

static eavlDataSet *V2E_Rectilinear(vtkRectilinearGrid *rg)
{
    eavlDataSet *ds = new eavlDataSet;

    vtkDataArray *vtkx = rg->GetXCoordinates();
    vtkDataArray *vtky = rg->GetYCoordinates();
    vtkDataArray *vtkz = rg->GetZCoordinates();

    vector< vector<double> > coords(3);
    for (int i=0; i<vtkx->GetNumberOfTuples(); ++i)
        coords[0].push_back(vtkx->GetTuple1(i));
    for (int i=0; i<vtky->GetNumberOfTuples(); ++i)
        coords[1].push_back(vtky->GetTuple1(i));
    for (int i=0; i<vtkz->GetNumberOfTuples(); ++i)
        coords[2].push_back(vtkz->GetTuple1(i));

    vector<string> coordNames;
    coordNames.push_back("xcoord");
    coordNames.push_back("ycoord");
    coordNames.push_back("zcoord");

    AddRectilinearMesh(ds, coords, coordNames, true, "RectilinearGridCells");

    return ds;
}

static void V2E_AddPoints(eavlDataSet *ds, vtkDataSet *in)
{
    int npts = in->GetNumberOfPoints();

    ds->SetNumPoints(npts);

    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(
                                              ds->GetLogicalStructure(),
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    ds->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("xcoord",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("ycoord",0));
    coords->SetAxis(2,new eavlCoordinateAxisField("zcoord",0));

    eavlArray *axisValues[3] = {
        new eavlFloatArray("xcoord",1, npts),
        new eavlFloatArray("ycoord",1, npts),
        new eavlFloatArray("zcoord",1, npts)
    };
    for (int i=0; i<npts; i++)
    {
        double *p = in->GetPoint(i);
        axisValues[0]->SetComponentFromDouble(i, 0, p[0]);
        axisValues[1]->SetComponentFromDouble(i, 0, p[1]);
        axisValues[2]->SetComponentFromDouble(i, 0, p[2]);
    }

    for (int d=0; d<3; d++)
    {
        eavlField *field = new eavlField(1, axisValues[d], eavlField::ASSOC_POINTS);
        ds->AddField(field);
    }
}


static void V2E_AddAllPointFields(eavlDataSet *ds, vtkDataSet *in)
{
    vtkPointData *pd = in->GetPointData();
    if (pd)
    {
        int narrays = pd->GetNumberOfArrays();
        for (int i=0; i<narrays; ++i)
        {
            vtkDataArray *vtkarray = pd->GetArray(i);
            string name = vtkarray->GetName();
            int nc = vtkarray->GetNumberOfComponents();
            int nt = vtkarray->GetNumberOfTuples();
            eavlFloatArray *eavlarray = new eavlFloatArray(name, nc, nt);
            for (int j=0; j<nc; j++)
            {
                for (int k=0; k<nt; k++)
                {
                    eavlarray->
                        SetComponentFromDouble(k,j,
                                               vtkarray->GetComponent(k,j));
                }
            }
            ds->AddField(new eavlField(1, eavlarray, eavlField::ASSOC_POINTS));
        }
    }
}

static void V2E_AddWholeMeshFields(eavlDataSet *ds, vtkDataSet *in)
{
    vtkFieldData *fd = in->GetAttributesAsFieldData(vtkDataObject::FIELD);
    if (fd)
    {
        int narrays = fd->GetNumberOfArrays();
        for (int i=0; i<narrays; ++i)
        {
            vtkDataArray *vtkarray = fd->GetArray(i);
            string name = vtkarray->GetName();
            int nc = vtkarray->GetNumberOfComponents();
            int nt = vtkarray->GetNumberOfTuples();
            eavlFloatArray *eavlarray = new eavlFloatArray(name, nc, nt);
            for (int j=0; j<nc; j++)
            {
                for (int k=0; k<nt; k++)
                {
                    eavlarray->
                        SetComponentFromDouble(k,j,
                                               vtkarray->GetComponent(k,j));
                }
            }
            ds->AddField(new eavlField(0, eavlarray, eavlField::ASSOC_WHOLEMESH));
        }
    }
}

static void V2E_AddAllCellFields(eavlDataSet *ds, vtkDataSet *in)
{
    vtkCellData *cd = in->GetCellData();
    if (cd)
    {
        int narrays = cd->GetNumberOfArrays();
        for (int i=0; i<narrays; ++i)
        {
            vtkDataArray *vtkarray = cd->GetArray(i);
            string name = vtkarray->GetName();
            int ncomp = vtkarray->GetNumberOfComponents();
            int ntuples = vtkarray->GetNumberOfTuples();
            int cellset = 0;
            string csname = ds->GetCellSet(cellset)->GetName();
            eavlFloatArray *eavlarray = new eavlFloatArray(name, ncomp, ntuples);
            for (int j=0; j<ncomp; j++)
            {
                for (int k=0; k<ntuples; k++)
                {
                    double val = vtkarray->GetComponent(k,j);
                    eavlarray->SetComponentFromDouble(k,j,val);
                }
            }
            ds->AddField(new eavlField(0, eavlarray, eavlField::ASSOC_CELL_SET, csname));
        }
    }
}

static void V2E_AddCellFields(eavlDataSet *ds, vtkDataSet *in,
                              vector<int> &cell_to_cell_splitmap)
{
    vtkCellData *cd = in->GetCellData();
    if (cd)
    {
        int narrays = cd->GetNumberOfArrays();
        for (int i=0; i<narrays; ++i)
        {
            vtkDataArray *vtkarray = cd->GetArray(i);
            string name = vtkarray->GetName();
            int ncomp = vtkarray->GetNumberOfComponents();
            int ntuples = vtkarray->GetNumberOfTuples();
            for (int c=0; c<ds->GetNumCellSets(); ++c)
            {
                eavlCellSet *cs = ds->GetCellSet(c);
                string csname = cs->GetName();
                int ncells = cs->GetNumCells();
                eavlFloatArray *eavlarray = new eavlFloatArray(name, ncomp, ncells);
                for (int j=0; j<ncomp; j++)
                {
                    int outctr = 0;
                    for (int k=0; k<ntuples; k++)
                    {
                        if (cell_to_cell_splitmap[k] == c)
                        {
                            double val = vtkarray->GetComponent(k,j);
                            eavlarray->SetComponentFromDouble(outctr,j,val);
                            outctr++;
                        }
                    }
                }
                ds->AddField(new eavlField(0, eavlarray, eavlField::ASSOC_CELL_SET, csname));
            }
        }
    }
}

eavlDataSet *ConvertVTKToEAVL(vtkDataSet *in)
{
    int th = eavlTimer::Start();
    eavlDataSet *result = NULL;

    if (in->GetDataObjectType() == VTK_UNSTRUCTURED_GRID)
    {
        vtkUnstructuredGrid *ug = dynamic_cast<vtkUnstructuredGrid*>(in);
        if (!ug)
        {
            THROW(eavlException, "Logic error: failed to cast VTK "
                  "unstructured grid");
        }

        vector<int> cell_to_cell_splitmap;
        result = V2E_Explicit(ug, cell_to_cell_splitmap);
        V2E_AddPoints(result, in);
        V2E_AddAllPointFields(result, in);
        V2E_AddCellFields(result, in, cell_to_cell_splitmap);
        V2E_AddWholeMeshFields(result, in);
    }
    else if (in->GetDataObjectType() == VTK_STRUCTURED_GRID)
    {
        vtkStructuredGrid *sg = dynamic_cast<vtkStructuredGrid*>(in);
        if (!sg)
        {
            THROW(eavlException, "Logic error: failed to cast VTK "
                  "structured grid");
        }

        result = V2E_Structured(sg->GetDimensions());
        V2E_AddPoints(result, in);
        V2E_AddAllPointFields(result, in);
        V2E_AddAllCellFields(result, in);
        V2E_AddWholeMeshFields(result, in);
    }
    else if (in->GetDataObjectType() == VTK_RECTILINEAR_GRID)
    {
        vtkRectilinearGrid *rg = dynamic_cast<vtkRectilinearGrid*>(in);
        if (!rg)
        {
            THROW(eavlException, "Logic error: failed to cast VTK "
                  "rectilinear grid");
        }

        result = V2E_Rectilinear(rg);
        V2E_AddAllPointFields(result, in);
        V2E_AddAllCellFields(result, in);
        V2E_AddWholeMeshFields(result, in);
    }
    else
    {
        // slower string fallback mostly just for polydata now....
        result = ConvertVTKToEAVL_Fallback(in);
    }
    result->PrintSummary(cerr);
    eavlTimer::Stop(th, "VTK -> EAVL");
    return result;
}

#else

vtkDataSet *ConvertEAVLToVTK(eavlDataSet *in)
{
    THROW(eavlException, "Cannot call EAVL<->VTK conversion; VTK library not compiled in.");
}

eavlDataSet *ConvertVTKToEAVL(vtkDataSet *in)
{
    THROW(eavlException, "Cannot call EAVL<->VTK conversion; VTK library not compiled in.");
}

#endif

