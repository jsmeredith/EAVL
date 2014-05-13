// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_ALL_STRUCTURED_H
#define EAVL_CELL_SET_ALL_STRUCTURED_H

#include "eavlCellSet.h"
#include "eavlRegularStructure.h"

// ****************************************************************************
// Class:  eavlCellSetAllStructured
//
// Purpose:
///   A set of all cells on a structured grid.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// Modifications:
//   Jeremy Meredith, Fri Oct 19 16:54:36 EDT 2012
//   Added reverse connectivity (i.e. get cells attached to a node).
//
// ****************************************************************************

class eavlCellSetAllStructured : public eavlCellSet
{
  protected:
    eavlRegularStructure regularStructure;
  public:
    eavlCellSetAllStructured() : eavlCellSet("", 0) {}
    eavlCellSetAllStructured(const string &n, eavlRegularStructure r)
        : eavlCellSet(n, r.dimension), regularStructure(r) { }
    
    virtual string className() const {return "eavlCellSetAllStructured";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlCellSet::serialize(s);
	regularStructure.serialize(s);
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlCellSet::deserialize(s);
	regularStructure.deserialize(s);
	return s;
    }

    eavlRegularStructure &GetRegularStructure() { return regularStructure; }
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetAllStructured:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;
        out << "        zdims[] = ";
        PrintVectorSummary(out, regularStructure.cellDims, dimensionality);
        out << endl;
    }
    virtual int GetNumCells()
    {
        return regularStructure.GetNumCells();
    }
    virtual int GetNumFaces()
    {
        return regularStructure.GetNumFaces();
    }
    virtual int GetNumEdges()
    {
        return regularStructure.GetNumEdges();
    }
    virtual eavlCell GetCellEdges(int index)
    {
        eavlCell c;
        c.type = (eavlCellShape)regularStructure.GetCellEdges(index,
                                                              c.numIndices,
                                                              c.indices);
        return c;
    }
    virtual eavlCell GetCellFaces(int index)
    {
        eavlCell c;
        c.type = (eavlCellShape)regularStructure.GetCellFaces(index,
                                                              c.numIndices,
                                                              c.indices);
        return c;
    }
    virtual eavlCell GetCellNodes(int index)
    {
        eavlCell c;
        c.type = (eavlCellShape)regularStructure.GetCellNodes(index,
                                                              c.numIndices,
                                                              c.indices);
        return c;
    }
    virtual eavlCell GetNodeCells(int index)
    {
        eavlCell c;
        c.type = (eavlCellShape)regularStructure.GetNodeCells(index,
                                                              c.numIndices,
                                                              c.indices);
        return c;
    }
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        return mem + eavlCellSet::GetMemoryUsage();
    }
};

#endif
