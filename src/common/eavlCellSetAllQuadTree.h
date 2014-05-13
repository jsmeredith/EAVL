// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_ALL_QUADTREE_H
#define EAVL_CELL_SET_ALL_QUADTREE_H

#include "eavlLogicalStructureQuadTree.h"

// ****************************************************************************
// Class:  eavlCellSetAllQuadTree
//
// Purpose:
///   All cells in a quadtree.
///   \todo: this assumes no points are shared across quadtree cells.
//
// Programmer:  Jeremy Meredith
// Creation:    July 26, 2012
//
// Modifications:
// ****************************************************************************
class eavlCellSetAllQuadTree : public eavlCellSet
{
  protected:
    eavlLogicalStructureQuadTree *log;
  public:
    eavlCellSetAllQuadTree(const string &n, eavlLogicalStructureQuadTree *l)
        : eavlCellSet(n, 2), log(l)
    {
    }
    virtual string className() const {return "eavlCellSetAllQuadTree";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlCellSet::serialize(s);
	log->serialize(s);
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlCellSet::deserialize(s);
	log->deserialize(s);
	return s;
    }
    virtual int GetNumCells()
    {
        return log->root.GetNumCells(true);
    }
    virtual eavlCell GetCellNodes(int index)
    {
        //cerr << "ASKING FOR CELL "<<index<<endl;
        eavlCell cell;
        cell.type = EAVL_PIXEL;
        cell.numIndices = 4;
        cell.indices[0] = index*4 + 0;
        cell.indices[1] = index*4 + 1;
        cell.indices[2] = index*4 + 2;
        cell.indices[3] = index*4 + 3;
        //eavlLogicalStructureQuadTree::QuadTreeCell *qcell = log->celllist[index];
        //cerr << "  width="<<(qcell->xmax-qcell->xmin)<<" level="<<qcell->lvl<<endl;
        return cell;
    }
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetAllQuadTree:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;
    }
};

#endif
