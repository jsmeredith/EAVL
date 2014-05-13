// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
//   Jeremy Meredith, Fri Oct 19 16:54:36 EDT 2012
//   Added reverse connectivity (i.e. get cells attached to a node).
//
// ****************************************************************************

class eavlCellSet
{
  protected:
    string              name;           ///< e.g. atoms, cells, nodes, faces
    int                 dimensionality; ///< e.g. 0, 1, 2, 3, (more?)

    int                 dataset_numpoints; ///< the number of points in the container data set
  public:
    eavlCellSet(const string &n, int d) : name(n), dimensionality(d), dataset_numpoints(0) { }
    virtual string className() const {return "eavlCellSet";}
    virtual eavlStream& serialize(eavlStream &s) const;
    virtual eavlStream& deserialize(eavlStream &s);
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
    virtual eavlCell GetNodeCells(int)
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
    /// Hack.  A cell set doesn't know about the parent (i.e. containing)
    /// data set.  Turns out this is normally not a problem, with one exception:
    /// if you are trying to create call-to-node connectivity 
    /// (e.g. reverse of the normal node-to-cell connectivity),
    /// for a cell set that does not touch every point in the
    /// original data set, how do you know how many untouched
    /// points should be added to the end?  (Concrete example:
    /// Data set has 4 points.  Cell set is a single triangle
    /// referencing points 0, 1, and 2.  When you create the
    /// reverse connectivity, you know pt 0 touches cell 0,
    /// pt 1 touches cell 0, and pt 2 touches cell 0.  But 
    /// you don't even know that there's a pt 3, because it's 
    /// not referenced by your original connectivity.  One
    /// could argue you simply shouldn't try to access pt 3,
    /// but what happens if you're trying to recenter a cell
    /// array to the nodes: step one is create a node-length
    /// output array, which in this case is correctly created
    /// of length 4, not of length 3, so I think the right thing
    /// is to create an empty entry for that extra point (as we
    /// have to do for other missing points already if they don't
    /// happen to be at the tail end of the list).
    virtual void SetDSNumPoints(int n)
    {
        dataset_numpoints = n;
    }

    static eavlCellSet * CreateObjFromName(const string &nm);
};


inline eavlStream& eavlCellSet::serialize(eavlStream &s) const
{
    s << name << dimensionality << dataset_numpoints;
    return s;
}

inline eavlStream& eavlCellSet::deserialize(eavlStream &s)
{
    s >> name >> dimensionality >> dataset_numpoints;
    return s;
}


#endif
