// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_LOGICAL_STRUCTURE_REGULAR_H
#define EAVL_LOGICAL_STRUCTURE_REGULAR_H

#include "eavlTopology.h"
#include "eavlLogicalStructure.h"
#include "eavlRegularStructure.h"
#include "eavlCell.h"
#include "eavlSerialize.h"

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
    eavlLogicalStructureRegular() : eavlLogicalStructure(0) {}
    eavlLogicalStructureRegular(int dim,
                                eavlRegularStructure rs=eavlRegularStructure())
        : eavlLogicalStructure(dim), reg(rs)
    {
    }
    
    virtual string className() const {return "eavlLogicalStructureRegular";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlLogicalStructure::serialize(s);
	reg.serialize(s);
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlLogicalStructure::deserialize(s);
	reg.deserialize(s);
	return s;
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
        for (int i=0; i<reg.dimension; i++)
        {
            out << reg.nodeDims[i] << " ";
        }
        out << endl;
    }
};


#endif
