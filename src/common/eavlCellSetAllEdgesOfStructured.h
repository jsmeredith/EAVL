// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_ALL_EDGES_OF_STRUCTURED_H
#define EAVL_CELL_SET_ALL_EDGES_OF_STRUCTURED_H

#include "eavlCellSetAllStructured.h"

// ****************************************************************************
// Class:  eavlCellSetAllEdgesOfStructured
//
// Purpose:
///   A set of all edges on a structured grid.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    December 29, 2011
//
// ****************************************************************************

class eavlCellSetAllEdgesOfStructured : public eavlCellSet
{
  protected:
    eavlRegularStructure regularStructure;
  public:
    eavlCellSetAllEdgesOfStructured(eavlCellSetAllStructured *p)
        : eavlCellSet(string("edges_of_")+p->GetName(), 1),
          regularStructure(p->GetRegularStructure())
    {
    }
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetAllEdgesOfStructured:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;

        out << "        zdims[] = ";
        PrintVectorSummary(out, regularStructure.cellDims, dimensionality);
        out << endl;
    }
    virtual int GetNumCells()
    {
        return regularStructure.GetNumEdges();
    }
    virtual eavlCell GetCellNodes(int edgeindex)
    {
        eavlCell e;
        e.type = (eavlCellShape)regularStructure.GetEdgeNodes(edgeindex,
                                                                 e.numIndices,
                                                                 e.indices);
        return e;
    }
};

#endif
