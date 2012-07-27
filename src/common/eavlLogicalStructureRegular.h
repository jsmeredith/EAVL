// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_LOGICAL_STRUCTURE_REGULAR_H
#define EAVL_LOGICAL_STRUCTURE_REGULAR_H

#include "eavlTopology.h"
#include "eavlLogicalStructure.h"
#include "eavlRegularStructure.h"
#include "eavlCell.h"

// ****************************************************************************
// Class:  eavlLogicalStructureRegular
//
// Purpose:
///   Defines a regular logical structure of the layout of a mesh.
///   Note that irregular meshes use this as well, but only have
//    a 1D logical structure because you only index them with a 1D index.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March  1, 2011
//
// ****************************************************************************
class eavlLogicalStructureRegular : public eavlLogicalStructure
{
  protected:
    eavlRegularStructure reg;
  public:
    ///\todo: I think we want to enforce that none of the dims[]
    ///       values here can be '1'; if it's 1 in some dimension,
    ///       you shouldn't have included that in the logical dims at all.
    eavlLogicalStructureRegular(int dim,
                                eavlRegularStructure rs=eavlRegularStructure())
        : eavlLogicalStructure(dim), reg(rs)
    {
    }
    eavlRegularStructure &GetRegularStructure()
    {
        return reg;
    }
    eavlRegularConnectivity GetConnectivity(eavlTopology topology)
    {
        return eavlRegularConnectivity(reg,topology);
    }
    virtual void PrintSummary(ostream &out)
    {
        out << "   eavlLogicalStructureRegular:"<<endl;
        out << "     logicalDimension = "<<logicalDimension<<endl;
        out << "     logicalDims["<<reg.dimension<<"] = ";
        for (unsigned int i=0; i<reg.dimension; i++)
        {
            out << reg.nodeDims[i] << " ";
        }
        out << endl;
    }
};


#endif
