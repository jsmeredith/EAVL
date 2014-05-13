// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlVTKExporter.h"
#include "eavlCellSetAllStructured.h"
#include "eavlCoordinates.h"

#include <iostream>

void
eavlVTKExporter::Export(ostream &out)
{
    out<<"# vtk DataFile Version 3.0"<<endl;
    out<<"vtk output"<<endl;
    out<<"ASCII"<<endl;

    eavlCellSet *cs = NULL;
    if (cellSetIndex >= 0 && cellSetIndex < data->GetNumCellSets())
        cs = data->GetCellSet(cellSetIndex);
    if (dynamic_cast<eavlCellSetAllStructured*>(cs))
    {
        ExportStructured(out);
    }
    else
    {
        ExportUnstructured(out);
    }
}


void
eavlVTKExporter::ExportStructured(ostream &out)
{
    eavlCoordinates *coords = data->GetCoordinateSystem(0);
    int ndims = coords->GetDimension();
    bool rectilinear = true;
    for (int axis=0; axis<ndims; ++axis)
    {
        eavlCoordinateAxis *ax = coords->GetAxis(axis);
        eavlCoordinateAxisField *axf = dynamic_cast<eavlCoordinateAxisField*>(ax);
        if (!axf)
        {
            ///\todo: not true, this would be rectilinear, but 
            /// instead of implementing this, we're going to
            /// be removing the rectilinear axis class soon
            /// (instead using implicit arrays with normal field axes)
            rectilinear = false;
            break;
        }
        string fn = axf->GetFieldName();
        eavlField *f = data->GetField(fn);
        if (f->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
            continue;

        ///\todo: this check is probably too strict:
        if (f->GetAssociation() == eavlField::ASSOC_LOGICALDIM &&
            f->GetAssocLogicalDim() == axis)
            continue;

        // nope, not easily convertable to vtk rectilinear
        rectilinear = false;
        break;
    }


    if (rectilinear)
    {
        out<<"DATASET RECTILINEAR_GRID"<<endl;
    }
    else
    {
        out<<"DATASET STRUCTURED_GRID"<<endl;
    }

    ExportGlobalFields(out);

    ExportStructuredCells(out);

    if (rectilinear)
        ExportRectilinearCoords(out);
    else
        ExportPoints(out);

    ExportFields(out);
}

void
eavlVTKExporter::ExportRectilinearCoords(ostream &out)
{
    eavlCellSetAllStructured *cs =
        dynamic_cast<eavlCellSetAllStructured*>(data->GetCellSet(cellSetIndex));
    if (!cs)
        return;

    eavlRegularStructure &reg = cs->GetRegularStructure();

    eavlCoordinates *coords = data->GetCoordinateSystem(0);
    int ndims = coords->GetDimension();
    const char *axnames[3] = {"X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"};
    for (int axis=0; axis<3; ++axis)
    {
        if (axis >= ndims)
        {
            out << axnames[axis] << " 1 float" << endl;
            out << "0" << endl;
            continue;
        }

        eavlCoordinateAxis *ax = coords->GetAxis(axis);
        eavlCoordinateAxisField *axf = dynamic_cast<eavlCoordinateAxisField*>(ax);
        // assert axf!=NULL
        string fn = axf->GetFieldName();
        eavlField *f = data->GetField(fn);
        eavlArray *arr = f->GetArray();

        int n = (axis >= reg.dimension) ? 1 : reg.nodeDims[axis];
        out << axnames[axis] << " " << n << " " << "float" << endl;
        if (f->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        {
            for (int i=0; i<n; ++i)
                out << arr->GetComponentAsDouble(0, 0) << " ";
            out << endl;
        }
        else if (f->GetAssociation() == eavlField::ASSOC_LOGICALDIM &&
                 f->GetAssocLogicalDim() == axis)
        {
            for (int i=0; i<n; ++i)
                out << arr->GetComponentAsDouble(i, 0) << " ";
            out << endl;
        }
    }

}

void
eavlVTKExporter::ExportUnstructured(ostream &out)
{
    out<<"DATASET UNSTRUCTURED_GRID"<<endl;
    ExportGlobalFields(out);
    ExportPoints(out);

    if (cellSetIndex >= 0 && cellSetIndex < data->GetNumCellSets())
        ExportUnstructuredCells(out);
    else
        ExportPointCells(out);
    ExportFields(out);
}

void
eavlVTKExporter::ExportPointCells(ostream &out)
{
    int nCells = data->GetNumPoints();

    out<<"CELLS "<<nCells<<" "<<nCells*2<<endl;
    for (int i = 0; i < nCells; i++)
    {
        out << "1 " << i << endl;
    }
    out<<"CELL_TYPES "<<nCells<<endl;
    for (int i = 0; i < nCells; i++)
    {
        out<<CellTypeToVTK(EAVL_POINT)<<endl;
    }
}

void
eavlVTKExporter::ExportStructuredCells(ostream &out)
{
    eavlCellSetAllStructured *cs =
        dynamic_cast<eavlCellSetAllStructured*>(data->GetCellSet(cellSetIndex));
    if (!cs)
        return;

    eavlRegularStructure &reg = cs->GetRegularStructure();
    out << "DIMENSIONS ";
    out << reg.nodeDims[0] << " ";
    out << ((reg.dimension > 1) ? reg.nodeDims[1] : 1) << " ";
    out << ((reg.dimension > 2) ? reg.nodeDims[2] : 1) << " ";
    out << endl;
}


void
eavlVTKExporter::ExportUnstructuredCells(ostream &out)
{
    int nCells = data->GetCellSet(cellSetIndex)->GetNumCells();

    int sz = 0;
    for (int i = 0; i < nCells; i++)
    {
        sz += data->GetCellSet(cellSetIndex)->GetCellNodes(i).numIndices;
        sz += 1;
    }

    out<<"CELLS "<<nCells<<" "<<sz<<endl;
    for (int i = 0; i < nCells; i++)
    {
        int nVerts = data->GetCellSet(cellSetIndex)->GetCellNodes(i).numIndices;
        eavlCell cell = data->GetCellSet(cellSetIndex)->GetCellNodes(i);
        out<<nVerts<<" ";
        for (int j = 0; j < nVerts; j++)
            out<<cell.indices[j]<<" ";
        out<<endl;
    }
    out<<"CELL_TYPES "<<nCells<<endl;
    for (int i = 0; i < nCells; i++)
    {
        eavlCell cell = data->GetCellSet(cellSetIndex)->GetCellNodes(i);
        out<<CellTypeToVTK(cell.type)<<endl;
    }
}

void
eavlVTKExporter::ExportGlobalFields(ostream &out)
{
    int count = 0;
    for (int f = 0; f < data->GetNumFields(); f++)
    {
        if (data->GetField(f)->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
            ++count;
    }
    bool wrote_global_field_header = false;
    for (int f = 0; f < data->GetNumFields(); f++)
    {
        int ntuples = data->GetField(f)->GetArray()->GetNumberOfTuples();
        int ncomp = data->GetField(f)->GetArray()->GetNumberOfComponents();
        
        if (ncomp > 4)
            continue;

        if (data->GetField(f)->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        {
            if (!wrote_global_field_header)
                out<<"FIELD FieldData "<<count<<endl;
            wrote_global_field_header = true;
            out<<data->GetField(f)->GetArray()->GetName()<<" "<<ncomp<<" "<<ntuples<<" float"<<endl;
            for (int i = 0; i < ntuples; i++)
            {
                for (int j = 0; j < ncomp; j++)
                    out<<data->GetField(f)->GetArray()->GetComponentAsDouble(i,j)<<endl;
            }
        }
    }
}

void
eavlVTKExporter::ExportFields(ostream &out)
{
    // do point data
    bool wrote_point_header = false;
    for (int f = 0; f < data->GetNumFields(); f++)
    {
        int ntuples = data->GetField(f)->GetArray()->GetNumberOfTuples();
        int ncomp = data->GetField(f)->GetArray()->GetNumberOfComponents();
        
        if (ncomp > 4)
            continue;

        if (data->GetField(f)->GetAssociation() == eavlField::ASSOC_POINTS)
        {
            if (!wrote_point_header)
                out<<"POINT_DATA "<<ntuples<<endl;
            wrote_point_header = true;
            out<<"SCALARS "<<data->GetField(f)->GetArray()->GetName()<<" float "<< ncomp<<endl;
            out<<"LOOKUP_TABLE default"<<endl;
            for (int i = 0; i < ntuples; i++)
            {
                for (int j = 0; j < ncomp; j++)
                    out<<data->GetField(f)->GetArray()->GetComponentAsDouble(i,j)<<endl;
            }
        }
    }

    // do cell data
    if (cellSetIndex < 0 || cellSetIndex >= data->GetNumCellSets())
        return;

    bool wrote_cell_header = false;
    for (int f = 0; f < data->GetNumFields(); f++)
    {
        int ntuples = data->GetField(f)->GetArray()->GetNumberOfTuples();
        int ncomp = data->GetField(f)->GetArray()->GetNumberOfComponents();
        
        if (ncomp > 4)
            continue;

        if (data->GetField(f)->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            data->GetField(f)->GetAssocCellSet() == data->GetCellSet(cellSetIndex)->GetName())
        {
            if (!wrote_cell_header)
                out<<"CELL_DATA "<<ntuples<<endl;
            wrote_cell_header = true;
            out<<"SCALARS "<<data->GetField(f)->GetArray()->GetName()<<" float "<< ncomp<<endl;
            out<<"LOOKUP_TABLE default"<<endl;
            for (int i = 0; i < ntuples; i++)
            {
                for (int j = 0; j < ncomp; j++)
                    out<<data->GetField(f)->GetArray()->GetComponentAsDouble(i,j)<<endl;
            }
        }
    }
}


void
eavlVTKExporter::ExportPoints(ostream &out)
{
    out<<"POINTS "<<data->GetNumPoints()<<" float"<<endl;

    int npts = data->GetNumPoints();
    for (int i = 0; i < npts; i++)
    {
        out<<(float)data->GetPoint(i, 0)<<" ";
        out<<(float)data->GetPoint(i, 1)<<" ";
        out<<(float)data->GetPoint(i, 2)<<endl;
    }
}

#define VTK_EMPTY_CELL     0
#define VTK_VERTEX         1
#define VTK_POLY_VERTEX    2
#define VTK_LINE           3
#define VTK_POLY_LINE      4
#define VTK_TRIANGLE       5
#define VTK_TRIANGLE_STRIP 6
#define VTK_POLYGON        7
#define VTK_PIXEL          8
#define VTK_QUAD           9
#define VTK_TETRA         10
#define VTK_VOXEL         11
#define VTK_HEXAHEDRON    12
#define VTK_WEDGE         13
#define VTK_PYRAMID       14
#define VTK_PENTAGONAL_PRISM 15
#define VTK_HEXAGONAL_PRISM  16

int
eavlVTKExporter::CellTypeToVTK(eavlCellShape type)
{
    int vtkType = -1;
    switch(type)
    {
      case EAVL_POINT:
        vtkType = VTK_VERTEX;
        break;
      case EAVL_BEAM:
        vtkType = VTK_LINE;
        break;
      case EAVL_TRI:
        vtkType = VTK_TRIANGLE;
        break;
      case EAVL_QUAD:
        vtkType = VTK_QUAD;
        break;
      case EAVL_PIXEL:
        vtkType = VTK_PIXEL;
        break;
      case EAVL_TET:
        vtkType = VTK_TETRA;
        break;
      case EAVL_PYRAMID:
        vtkType = VTK_PYRAMID;
        break;
      case EAVL_WEDGE:
        vtkType = VTK_WEDGE;
        break;
      case EAVL_HEX:
        vtkType = VTK_HEXAHEDRON;
        break;
      case EAVL_VOXEL:
        vtkType = VTK_VOXEL;
        break;
      case EAVL_TRISTRIP:
        vtkType = VTK_TRIANGLE_STRIP;
        break;
      case EAVL_POLYGON:
        vtkType = VTK_POLYGON;
        break;
      case EAVL_OTHER:
        break;
    }
    
    return vtkType;
}
