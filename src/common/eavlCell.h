// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_H
#define EAVL_CELL_H

enum eavlCellShape
{
    EAVL_POINT,
    EAVL_BEAM,
    EAVL_TRI,
    EAVL_QUAD,
    EAVL_PIXEL,
    EAVL_TET,
    EAVL_PYRAMID,
    EAVL_WEDGE,
    EAVL_HEX,
    EAVL_VOXEL,
    EAVL_TRISTRIP, ///<\todo: only needed for VTK polydata support? can remove?
    EAVL_POLYGON,  ///<\todo: added for VTK polydata support; really necessary?
    EAVL_OTHER
};

// ****************************************************************************
// Class:  eavlCell
//
// Purpose:
///   A single cell.
///   \todo: horribly inefficient.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************

class eavlCell
{
  public:
    eavlCellShape    type;
    int              numIndices;
    int              indices[12]; ///< \todo: bad idea

    void PrintSummary(ostream &out)
    {
        out<<"Cell: ";
        if      (type == EAVL_POINT)    out<<"EAVL_POINT";
        else if (type == EAVL_BEAM)     out<<"EAVL_BEAM";
        else if (type == EAVL_TRI)      out<<"EAVL_TRI";
        else if (type == EAVL_QUAD)     out<<"EAVL_QUAD";
        else if (type == EAVL_PIXEL)    out<<"EAVL_PIXEL";
        else if (type == EAVL_TET)      out<<"EAVL_TET";
        else if (type == EAVL_PYRAMID)  out<<"EAVL_PYRAMID";
        else if (type == EAVL_WEDGE)    out<<"EAVL_WEDGE";
        else if (type == EAVL_HEX)      out<<"EAVL_HEX";
        else if (type == EAVL_VOXEL)    out<<"EAVL_VOXEL";
        else if (type == EAVL_VOXEL)    out<<"EAVL_VOXEL";
        else if (type == EAVL_TRISTRIP) out<<"EAVL_TRISTRIP";
        else if (type == EAVL_POLYGON)  out<<"EAVL_POLYGON";
        else if (type == EAVL_OTHER)    out<<"EAVL_OTHER";
        out<<" numIndices= "<<numIndices<<" [";
        for (int i = 0; i < numIndices; i++)
            out<<indices[i]<<" ";
        out<<"]";
        out<<endl;
    }
};


#endif
