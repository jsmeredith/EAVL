// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_ALL_FACES_OF_EXPLICIT_H
#define EAVL_CELL_SET_ALL_FACES_OF_EXPLICIT_H

#include "eavlCellComponents.h"
#include "eavlCellSetExplicit.h"
// ****************************************************************************
// Class:  eavlCellSetAllFacesOfExplicit
//
// Purpose:
///   A set of all faces on an explicit set of cells.
///   \todo: question: do we create faces for a 2D polygon mesh?
///   \todo: BIGGER question: don't we need to support an "AllEdges" of
///          this cell set as well?  Or more specifically, provide a GetEdge
///          method?  How about a GetFace method?
///          It's not so bad because we can just defer any edge-related stuff
///          to the parent cell set, I think....
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    January  3, 2012
//
// ****************************************************************************

class eavlCellSetAllFacesOfExplicit : public eavlCellSet
{
  protected:
    eavlCellSetExplicit *parent; ///<\todo: should this really be a pointer, or index, or what?
  public:
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetAllFacesOfExplicit:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;
        out << "        parent = "<<parent<<endl;
    }
    eavlCellSetAllFacesOfExplicit(eavlCellSetExplicit *p)
        : eavlCellSet(string("faces_of_")+p->GetName(), 2), parent(p)
    {
        if (!p)
            throw;
        if (parent->GetNumCells() <= 0)
            throw;
    }
    virtual int GetNumCells()
    {
        return parent->GetNumFaces();
    }
    virtual eavlCell GetCellNodes(int faceindex)
    {
        const eavlExplicitConnectivity &conn = 
            parent->GetConnectivity(EAVL_NODES_OF_FACES);
        eavlCell cell;
        cell.type = eavlCellShape(conn.GetElementComponents(faceindex,
                                                            cell.numIndices,
                                                            cell.indices));
        return cell;
    }
};

#endif
