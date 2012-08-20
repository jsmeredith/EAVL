// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_H
#define EAVL_CELL_SET_H

#include "eavlArray.h"
#include "eavlUtility.h"
#include "eavlLogicalStructureRegular.h"
#include "eavlCell.h"

// ****************************************************************************
// Class:  eavlCellSet
//
// Purpose:
///   Encapsulate the connectivity and fields of cells.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************

class eavlCellSet
{
  protected:
    string              name;           ///< e.g. atoms, cells, nodes, faces
    int                 dimensionality; ///< e.g. 0, 1, 2, 3, (more?)
  public:
    eavlCellSet(const string &n, int d) : name(n), dimensionality(d) { }
    virtual string GetName() { return name; }
    virtual int GetDimensionality() { return dimensionality; }
    virtual int GetNumCells() = 0;
    virtual int GetNumFaces() { return 0; }
    virtual int GetNumEdges() { return 0; }
    virtual eavlCell GetCellNodes(int i) = 0;
    virtual eavlCell GetCellFaces(int)
    {
        eavlCell c;
        c.type=EAVL_OTHER;
        c.numIndices = 0;
        return c;
    }
    virtual eavlCell GetCellEdges(int)
    {
        eavlCell c;
        c.type=EAVL_OTHER;
        c.numIndices = 0;
        return c;
    }
    virtual void PrintSummary(ostream&) = 0;
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(string) + name.length() * sizeof(char);
        mem += sizeof(int); // dimensionality
        mem += sizeof(int); // nCells;
        return mem;
    }
};


#endif

