// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlVTKExporter.h"
#include <iostream>

void
eavlVTKExporter::Export(ostream &out)
{
    out<<"# vtk DataFile Version 3.0"<<endl;
    out<<"vtk output"<<endl;
    out<<"ASCII"<<endl;

    ExportUnstructured(out);
}


void
eavlVTKExporter::ExportUnstructured(ostream &out)
{
    out<<"DATASET UNSTRUCTURED_GRID"<<endl;
    ExportPoints(out);

    ExportCells(out);
    ExportFields(out);
}

void
eavlVTKExporter::ExportCells(ostream &out)
{
    if (data->cellsets.size() == 0)
        return;

    int nCells = data->cellsets[cellSetIndex]->GetNumCells();

    int sz = 0;
    for (int i = 0; i < nCells; i++)
    {
        sz += data->cellsets[cellSetIndex]->GetCellNodes(i).numIndices;
        sz += 1;
    }

    out<<"CELLS "<<nCells<<" "<<sz<<endl;
    for (int i = 0; i < nCells; i++)
    {
        int nVerts = data->cellsets[cellSetIndex]->GetCellNodes(i).numIndices;
        eavlCell cell = data->cellsets[cellSetIndex]->GetCellNodes(i);
        out<<nVerts<<" ";
        for (int j = 0; j < nVerts; j++)
            out<<cell.indices[j]<<" ";
        out<<endl;
    }
    out<<"CELL_TYPES "<<nCells<<endl;
    for (int i = 0; i < nCells; i++)
    {
        eavlCell cell = data->cellsets[cellSetIndex]->GetCellNodes(i);
        out<<CellTypeToVTK(cell.type)<<endl;
    }
}

void
eavlVTKExporter::ExportFields(ostream &out)
{
    // do point data
    bool wrote_point_header = false;
    for (unsigned int f = 0; f < data->fields.size(); f++)
    {
        int ntuples = data->fields[f]->GetArray()->GetNumberOfTuples();
        int ncomp = data->fields[f]->GetArray()->GetNumberOfComponents();
        
        if (ncomp > 4)
            continue;

        if (data->fields[f]->GetAssociation() == eavlField::ASSOC_POINTS)
        {
            if (!wrote_point_header)
                out<<"POINT_DATA "<<ntuples<<endl;
            wrote_point_header = true;
            out<<"SCALARS "<<data->fields[f]->GetArray()->GetName()<<" float "<< ncomp<<endl;
            out<<"LOOKUP_TABLE default"<<endl;
            for (int i = 0; i < ntuples; i++)
            {
                for (int j = 0; j < ncomp; j++)
                    out<<data->fields[f]->GetArray()->GetComponentAsDouble(i,j)<<endl;
            }
        }
    }

    // do cell data
    bool wrote_cell_header = false;
    for (unsigned int f = 0; f < data->fields.size(); f++)
    {
        int ntuples = data->fields[f]->GetArray()->GetNumberOfTuples();
        int ncomp = data->fields[f]->GetArray()->GetNumberOfComponents();
        
        if (ncomp > 4)
            continue;

        if (data->fields[f]->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            data->fields[f]->GetAssocCellSet() == cellSetIndex)
        {
            if (!wrote_cell_header)
                out<<"CELL_DATA "<<ntuples<<endl;
            wrote_cell_header = true;
            out<<"SCALARS "<<data->fields[f]->GetArray()->GetName()<<" float "<< ncomp<<endl;
            out<<"LOOKUP_TABLE default"<<endl;
            for (int i = 0; i < ntuples; i++)
            {
                for (int j = 0; j < ncomp; j++)
                    out<<data->fields[f]->GetArray()->GetComponentAsDouble(i,j)<<endl;
            }
        }
    }
}


void
eavlVTKExporter::ExportPoints(ostream &out)
{
    out<<"POINTS "<<data->npoints<<" float"<<endl;

    int dim = data->coordinateSystems[0]->GetDimension();
    int npts = data->npoints;
    for (int i = 0; i < npts; i++)
    {
        out<<(float)data->GetPoint(i, 0)<<" ";
        out<<(dim >=2 ? (float)data->GetPoint(i, 1) : 0.0)<<" ";
        out<<(dim >=3 ? (float)data->GetPoint(i, 2) : 0.0)<<endl;
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
eavlVTKExporter::CellTypeToVTK(eavlCellShape &type)
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
