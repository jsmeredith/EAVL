// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_LOGICAL_STRUCTURE_H
#define EAVL_LOGICAL_STRUCTURE_H

#include "STL.h"
#include "eavlSerialize.h"

// ****************************************************************************
// Class:  eavlLogicalStructure
//
// Purpose:
///   Defines the logical structure of the layout of a mesh.  For
///   example, a 1D array of points for an N-dimensional unstructured
///   grid, a 3D array of points for a 3D structured grid.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March  1, 2011
//
// ****************************************************************************
class eavlLogicalStructure
{
  protected:
    int logicalDimension;
  public:
    eavlLogicalStructure(int dim) : logicalDimension(dim) { }

    virtual string className() const {return "eavlLogicalStructure";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << logicalDimension;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	s >> logicalDimension;
	return s;
    }

    virtual void PrintSummary(ostream &out) = 0;
    int GetDimension() const { return logicalDimension; }

    static eavlLogicalStructure* CreateObjFromName(const string &nm);
};


#endif
