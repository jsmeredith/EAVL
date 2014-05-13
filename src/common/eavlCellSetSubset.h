// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_SUBSET_H
#define EAVL_CELL_SET_SUBSET_H

#include "eavlCellSet.h"
// ****************************************************************************
// Class:  eavlCellSetSubset
//
// Purpose:
///   This set of cells is an explicit subset of other another cell set
///
///\todo: the existence of this in our current data model raises one issue:
///       do we need to make an explicit downselection of our cell variables
///       when we do the subselection?  if we indirected the cell variable
///       like we do GetCell, though the eavlCellSetSubset, then we
///       wouldn't need to copy it at all -- just look up the corresponding
///       value from the original array directly!  Still, maybe when
///       we create a subset it's more efficient to downselect it anyway
///       (it has almost no chance of causing a memory expansion, unlike
///       the mesh structure might e.g. from rect -> unstructured).
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************


class eavlCellSetSubset : public eavlCellSet
{
  protected:
    eavlCellSet *parent; ///<\todo: should this really be a pointer, or index, or what?
  public:
    vector<int> subset;
  public:
    eavlCellSetSubset(eavlCellSet *p)
        : eavlCellSet(string("subset_of_")+p->GetName(), p->GetDimensionality()),
          parent(p)
    {
    }

    virtual string className() const {return "eavlCellSetSubset";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlCellSet::serialize(s);
	parent->serialize(s);
	s << subset;
	throw; //fix the parent serialization.
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlCellSet::deserialize(s);
	parent->deserialize(s);
	s >> subset;
	throw; //fix the parent deserialization.
	return s;
    }

    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetSubset:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;
        out << "        parent = "<<parent<<endl;
        out << "        subset["<<subset.size()<<"] = ";
        PrintVectorSummary(out, subset);
        out << endl;
    }
    virtual int GetNumCells()
    {
        return subset.size();
    }
    virtual eavlCell GetCellNodes(int index)
    {
        return parent->GetCellNodes(subset[index]);
    }
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(eavlCellSet*);
        mem += sizeof(vector<int>);
        mem += subset.size() * sizeof(int);
        return mem + eavlCellSet::GetMemoryUsage();
    }
};

#endif
