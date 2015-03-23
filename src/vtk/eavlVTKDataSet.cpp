// Copyright 2010-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
#include <vtkPolyData.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkCellArray.h>
#include <vtkCellTypes.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>

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

vtkDataSet *E2V_Structured(eavlDataSet *in)
{
    //
    // First, see if this counts as rectilinear
    //
    eavlCoordinates *coords = in->GetCoordinateSystem(0);
    int ndims = coords->GetDimension();
    if (ndims != 3)
    {
        THROW(eavlException, "Unimplemented: ndims != 3");
    }
    bool rectilinear = true;
    for (int axis=0; axis<ndims; ++axis)
    {
        eavlCoordinateAxis *ax = coords->GetAxis(axis);
        eavlCoordinateAxisField *axf = dynamic_cast<eavlCoordinateAxisField*>(ax);
        if (!axf)
        {
            ///\todo: not quite true, this would be rectilinear, but 
            /// instead of implementing this, we're going to
            /// be removing the rectilinear axis class soon
            /// (instead using implicit arrays with normal field axes)
            rectilinear = false;
            break;
        }
        string fn = axf->GetFieldName();
        eavlField *f = in->GetField(fn);
        if (f->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        {
            continue;
        }

        ///\todo: this check is probably too strict:
        if (f->GetAssociation() == eavlField::ASSOC_LOGICALDIM &&
            f->GetAssocLogicalDim() == axis)
            continue;

        // nope, not easily convertable to vtk rectilinear
        rectilinear = false;
        break;
    }


    //
    // Get the dimensions
    //
    eavlLogicalStructureRegular *log =
        dynamic_cast<eavlLogicalStructureRegular*>(in->GetLogicalStructure());
    eavlRegularStructure &reg = log->GetRegularStructure();
    if (!log)
    {
        THROW(eavlException, "Expected eavlLogicalStructureRegular");
    }
    int dims[3] = {reg.nodeDims[0],
                   reg.dimension < 2 ? 1 : reg.nodeDims[1],
                   reg.dimension < 3 ? 1 : reg.nodeDims[2]};

    //
    // Convert
    //
    if (rectilinear)
    {
        vtkRectilinearGrid *rg = vtkRectilinearGrid::New();
        rg->SetDimensions(dims);
        for (int i=0; i<ndims; ++i)
        {
            vtkFloatArray *val = vtkFloatArray::New();

            eavlCoordinateAxis *ax = coords->GetAxis(i);
            eavlCoordinateAxisField *axf = dynamic_cast<eavlCoordinateAxisField*>(ax);
            string fn = axf->GetFieldName();
            eavlField *f = in->GetField(fn);
            eavlArray *a = f->GetArray();

            int n = a->GetNumberOfTuples();
            val->SetNumberOfTuples(n);
            for (int j=0; j<n; ++j)
                val->SetComponent(j, 0, a->GetComponentAsDouble(j, 0));
            if (i==0)
                rg->SetXCoordinates(val);
            else if (i==1)
                rg->SetYCoordinates(val);
            else
                rg->SetZCoordinates(val);
            val->Delete();
        }
        return rg;
    }
    else
    {
        vtkStructuredGrid *sg = vtkStructuredGrid::New();
        sg->SetDimensions(dims);
        return sg;
    }
}

vtkDataSet *E2V_Explicit(eavlDataSet *in, eavlCellSetExplicit *cs)
{
    vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();

    int ncells = cs->GetNumCells();
    ug->Allocate(ncells);
    for (int i=0; i<ncells; ++i)
    {
        eavlCell c = cs->GetCellNodes(i);
        int st;
        switch (c.type)
        {
          case EAVL_POINT:   st = VTK_VERTEX;     break;
          case EAVL_BEAM:    st = VTK_LINE;       break;
          case EAVL_TRI:     st = VTK_TRIANGLE;   break;
          case EAVL_POLYGON: st = VTK_POLYGON;    break;
          case EAVL_PIXEL:   st = VTK_PIXEL;      break;
          case EAVL_QUAD:    st = VTK_QUAD;       break;
          case EAVL_TET:     st = VTK_TETRA;      break;
          case EAVL_VOXEL:   st = VTK_VOXEL;      break;
          case EAVL_HEX:     st = VTK_HEXAHEDRON; break;
          case EAVL_WEDGE:   st = VTK_WEDGE;      break;
          case EAVL_PYRAMID: st = VTK_PYRAMID;    break;
          default:           st = VTK_EMPTY_CELL; break;
        }
        vtkIdType ptIds[12];
        for (int j=0; j<c.numIndices; ++j)
            ptIds[j] = c.indices[j];
        ug->InsertNextCell(st, c.numIndices, ptIds);
    }

    return ug;
}

static void E2V_AddPoints_Copy(vtkPointSet *ds, eavlDataSet *in)
{
    int npts = in->GetNumPoints();

    vtkPoints *pts = vtkPoints::New();
    ds->SetPoints(pts);

    pts->SetNumberOfPoints(npts);
    for (int i=0; i<npts; ++i)
    {
        float x = in->GetPoint(i,0);
        float y = in->GetPoint(i,1);
        float z = in->GetPoint(i,2);
        pts->SetPoint(i, x, y, z);
    }
}


static void E2V_AddPoints_ZeroCopy(vtkPointSet *ds, eavlDataSet *in)
{
    int npts = in->GetNumPoints();
    eavlCoordinates *coords = in->GetCoordinateSystem(0);
    int ndims = coords->GetDimension();
    if (ndims != 3)
    {
        // assuming 3 dimensional for zerocopy
        E2V_AddPoints_Copy(ds,in);
        return;
    }
    string name = "";
    for (int axis=0; axis<ndims; ++axis)
    {
        eavlCoordinateAxis *ax = coords->GetAxis(axis);
        eavlCoordinateAxisField *axf = dynamic_cast<eavlCoordinateAxisField*>(ax);
        if (!axf || axf->GetComponent() != axis)
        {
            // expected axis fields to be components 0,1,2 of the field, in that order
            E2V_AddPoints_Copy(ds,in);
            return;
        }
        if (name == "")
        {
            name = axf->GetFieldName();
        }
        else
        {
            if (name != axf->GetFieldName())
            {
                // and they'd better all be the same name
                E2V_AddPoints_Copy(ds,in);
                return;
            }
        }
    }
    if (name == "")
    {
        // something went wrong; didn't get an actual field name
        E2V_AddPoints_Copy(ds,in);
        return;
    }

    // Okay, we're safe to use zerocopy now
    eavlField *field = in->GetField(name);
    eavlArray *array = field->GetArray();
    vtkFloatArray *f = vtkFloatArray::New();
    f->SetNumberOfComponents(ndims);
    f->SetArray((float*)dynamic_cast<eavlFloatArray*>(array)->GetHostArray(), 3*npts, 1);

    vtkPoints *pts = vtkPoints::New();
    ds->SetPoints(pts);
    pts->SetData(f);
}

static void E2V_AddAllFields_ZeroCopy(vtkDataSet *ds, eavlDataSet *in, eavlCellSet *cs)
{
    for (int f=0; f<in->GetNumFields(); ++f)
    {
        eavlField *field = in->GetField(f);
        eavlArray *array = field->GetArray();
        if (field->GetAssociation() == eavlField::ASSOC_POINTS || 
            (field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
             field->GetAssocCellSet() == cs->GetName()))
        {   
            eavlFloatArray *farray = dynamic_cast<eavlFloatArray*>(array);
            if (farray)
            {
                vtkFloatArray *v = vtkFloatArray::New();
                int nc = array->GetNumberOfComponents();
                int nt = array->GetNumberOfTuples();
                v->SetNumberOfComponents(nc);
                //v->SetNumberOfTuples(nt); //<- no, this allocates it
                v->SetName(array->GetName().c_str());
                v->SetArray((float*)farray->GetHostArray(), nc*nt, 1);
                if (field->GetAssociation() == eavlField::ASSOC_POINTS)
                    ds->GetPointData()->AddArray(v);
                else
                    ds->GetCellData()->AddArray(v);
            }
        }
    }
}

static void E2V_AddAllFields_Copy(vtkDataSet *ds, eavlDataSet *in, eavlCellSet *cs)
{
    for (int f=0; f<in->GetNumFields(); ++f)
    {
        eavlField *field = in->GetField(f);
        eavlArray *array = field->GetArray();
        if (field->GetAssociation() == eavlField::ASSOC_POINTS || 
            (field->GetAssociation() == eavlField::ASSOC_CELL_SET &&
             field->GetAssocCellSet() == cs->GetName()))
        {   
            eavlFloatArray *farray = dynamic_cast<eavlFloatArray*>(array);
            if (farray)
            {
                vtkFloatArray *v = vtkFloatArray::New();
                int nc = array->GetNumberOfComponents();
                int nt = array->GetNumberOfTuples();
                v->SetNumberOfComponents(nc);
                v->SetNumberOfTuples(nt);
                v->SetName(array->GetName().c_str());
                float *ptr = (float*)v->GetVoidPointer(0);
                for (int k=0; k<nt; k++)
                {
                    for (int j=0; j<nc; j++)
                    {
                        ptr[k*nc+j] = farray->GetComponentAsDouble(k,j);
                    }
                }
                if (field->GetAssociation() == eavlField::ASSOC_POINTS)
                    ds->GetPointData()->AddArray(v);
                else
                    ds->GetCellData()->AddArray(v);
            }
        }
    }
}

vtkDataSet *ConvertEAVLToVTK(eavlDataSet *in)
{
    int th = eavlTimer::Start();
    vtkDataSet *result = NULL;

    int cellsetindex = 0;

    if (cellsetindex < 0 || cellsetindex > in->GetNumCellSets())
        THROW(eavlException, "Logic error: cell set index out of range");

    eavlCellSet *cs = in->GetCellSet(cellsetindex);
    if (dynamic_cast<eavlCellSetAllStructured*>(cs))
    {
        result = E2V_Structured(in);

        // for non-rectilinear, still need to add points
        if (dynamic_cast<vtkPointSet*>(result))
            E2V_AddPoints_Copy(dynamic_cast<vtkPointSet*>(result), in);

        E2V_AddAllFields_Copy(result, in, cs);
    }
    else if (dynamic_cast<eavlCellSetExplicit*>(cs))
    {
        result = E2V_Explicit(in, dynamic_cast<eavlCellSetExplicit*>(cs));

        E2V_AddPoints_Copy(dynamic_cast<vtkPointSet*>(result), in);

        E2V_AddAllFields_Copy(result, in, cs);
    }
    else
    {
        // slower string fallback -- mostly to compare performance,
        // but used in some unusual cases as well
        result = ConvertEAVLToVTK_Fallback(in);
    }

    eavlTimer::Stop(th, "EAVL -> VTK");

    // debug -- double check the output
    if (false)
    {
        vtkDataSetWriter *wrtr = vtkDataSetWriter::New();
        wrtr->SetFileName("output.vtk");
        wrtr->SetFileTypeToASCII();
        wrtr->SetInputData(result);
        wrtr->Write();
        cout << wrtr->GetOutputString() << endl;
    }

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

static eavlDataSet *V2E_Unstructured(vtkUnstructuredGrid *ug,
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

static eavlDataSet *V2E_PolyData(vtkPolyData *pd,
                                 vector<int> &cell_to_cell_splitmap)
{
    eavlDataSet *ds = new eavlDataSet;

    eavlCellSetExplicit *cells[4] = {
        new eavlCellSetExplicit("PolyDataVertices", 0),
        new eavlCellSetExplicit("PolyDataLines", 1),
        new eavlCellSetExplicit("PolyDataPolygons", 2),
        new eavlCellSetExplicit("PolyDataTriStrips", 2)
    };
    eavlExplicitConnectivity newconn[4];

    for (int pass = 0 ; pass < 3 ; ++pass)
    {
        eavlCellShape st = EAVL_OTHER;
        vtkCellArray *pd_cells = NULL;
        int index = -1;
        switch (pass)
        {
          case 0: index = 0; pd_cells = pd->GetVerts(); st = EAVL_POINT; break;
          case 1: index = 1; pd_cells = pd->GetLines(); st = EAVL_BEAM;  break;
          case 2: index = 2; pd_cells = pd->GetPolys(); st = EAVL_POLYGON; break;
        }
        vtkIdType npts, *pts, cellId;
        for (cellId=0, pd_cells->InitTraversal();
             pd_cells->GetNextCell(npts,pts);
             cellId++)
        {
            newconn[index].AddElement(st,  npts, pts);
            
            // we need to split fields by dimensionality
            cell_to_cell_splitmap.push_back(index);
        }
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


static void V2E_AddPoints_Copy(eavlDataSet *ds, vtkPointSet *in)
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
        new eavlFloatArray("xcoord", 1, npts),
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


static void V2E_AddPoints_ZeroCopy(eavlDataSet *ds, vtkPointSet *in)
{
    int npts = in->GetNumberOfPoints();

    vtkPoints *pts = in->GetPoints();
    vtkDataArray *data = pts->GetData();
    float *fptr = dynamic_cast<vtkFloatArray*>(data) ? dynamic_cast<vtkFloatArray*>(data)->GetPointer(0) : NULL;
    double *dptr = dynamic_cast<vtkDoubleArray*>(data) ? dynamic_cast<vtkDoubleArray*>(data)->GetPointer(0) : NULL;
    eavlArray *axisValues = NULL;
    if (fptr)
    {
        axisValues = new eavlFloatArray(eavlArray::HOST, fptr, "coords", 3, npts);
    }
    else if (dptr)
    {
        //cerr << "Error: no double array support yet.....\n";
        // Fall back to copy
        V2E_AddPoints_Copy(ds,in);
        return;
    }
    else
    {
        //cerr << "Error: expected float or double coords....\n";
        // Fall back to copy
        V2E_AddPoints_Copy(ds,in);
        return;
    }

    ds->SetNumPoints(npts);

    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(
                                              ds->GetLogicalStructure(),
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    ds->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("coords",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("coords",1));
    coords->SetAxis(2,new eavlCoordinateAxisField("coords",2));

    eavlField *field = new eavlField(1, axisValues, eavlField::ASSOC_POINTS);
    ds->AddField(field);
}



static void V2E_AddAllPointFields_ZeroCopy(eavlDataSet *ds, vtkDataSet *in)
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
            if (dynamic_cast<vtkFloatArray*>(vtkarray))
            {
                float *fptr = dynamic_cast<vtkFloatArray*>(vtkarray)->GetPointer(0);
                eavlFloatArray *eavlarray = new eavlFloatArray(eavlArray::HOST,
                                                               fptr, name, nc, nt);
                ds->AddField(new eavlField(1, eavlarray, eavlField::ASSOC_POINTS));
            }
            else
            {
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
}

static void V2E_AddAllPointFields_Copy(eavlDataSet *ds, vtkDataSet *in)
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

static void V2E_AddAllCellFields_ZeroCopy(eavlDataSet *ds, vtkDataSet *in)
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
            if (dynamic_cast<vtkFloatArray*>(vtkarray))
            {
                float *fptr = dynamic_cast<vtkFloatArray*>(vtkarray)->GetPointer(0);
                eavlFloatArray *eavlarray = new eavlFloatArray(eavlArray::HOST,
                                                               fptr, name, ncomp, ntuples);
                ds->AddField(new eavlField(0, eavlarray, eavlField::ASSOC_CELL_SET, csname));
            }
            else
            {
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
}

static void V2E_AddAllCellFields_Copy(eavlDataSet *ds, vtkDataSet *in)
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

static void V2E_AddCellFields_splitmap(eavlDataSet *ds, vtkDataSet *in,
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
        result = V2E_Unstructured(ug, cell_to_cell_splitmap);
        V2E_AddPoints_Copy(result, ug);
        V2E_AddAllPointFields_Copy(result, in);
        V2E_AddCellFields_splitmap(result, in, cell_to_cell_splitmap);
        V2E_AddWholeMeshFields(result, in);
    }
    else if (in->GetDataObjectType() == VTK_POLY_DATA)
    {
        vtkPolyData *pd = dynamic_cast<vtkPolyData*>(in);
        if (!pd)
        {
            THROW(eavlException, "Logic error: failed to cast VTK "
                  "unstructured grid");
        }

        vector<int> cell_to_cell_splitmap;
        result = V2E_PolyData(pd, cell_to_cell_splitmap);
        V2E_AddPoints_Copy(result, pd);
        V2E_AddAllPointFields_Copy(result, in);
        V2E_AddCellFields_splitmap(result, in, cell_to_cell_splitmap);
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
        V2E_AddPoints_Copy(result, sg);
        V2E_AddAllPointFields_Copy(result, in);
        V2E_AddAllCellFields_Copy(result, in);
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
        V2E_AddAllPointFields_Copy(result, in);
        V2E_AddAllCellFields_Copy(result, in);
        V2E_AddWholeMeshFields(result, in);
    }
    else
    {
        // slower string fallback mostly just for polydata now....
        result = ConvertVTKToEAVL_Fallback(in);
    }

    // debug -- double check the output
    //result->PrintSummary(cerr);
    //WriteToVTKFile(result, "output_eavl.vtk", 0);

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

