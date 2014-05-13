// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_ALL_FACES_OF_STRUCTURED_H
#define EAVL_CELL_SET_ALL_FACES_OF_STRUCTURED_H

#include "eavlCellSetAllStructured.h"

// ****************************************************************************
// Class:  eavlCellSetAllFacesOfStructured
//
// Purpose:
///   A set of all faces on a structured grid.
///   \todo: question: don't we need to support an "AllEdges" of
///          this cell set as well?  Or more specifically, provide a GetEdge
///          method?  How about a GetFace method?
///          It's not so bad because we can just defer any edge-related stuff
///          to the parent cell set, I think....
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    December 29, 2011
//
// ****************************************************************************

class eavlCellSetAllFacesOfStructured : public eavlCellSet
{
  protected:
    eavlRegularStructure regularStructure;
  public:
    eavlCellSetAllFacesOfStructured(eavlCellSetAllStructured *p)
        : eavlCellSet(string("faces_of_")+p->GetName(), 2),
          regularStructure(p->GetRegularStructure())
    {
    }

    virtual string className() const {return "eavlCellSetAllFacesOfStructured";}
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
    
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetAllFacesOfStructured:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;

        out << "        zdims[] = ";
        PrintVectorSummary(out, regularStructure.cellDims, dimensionality);
        out << endl;
    }
    virtual int GetNumCells()
    {
        return regularStructure.GetNumFaces();
    }
    virtual eavlCell GetCellNodes(int faceindex)
    {
        eavlCell e;
        e.type = (eavlCellShape)regularStructure.GetFaceNodes(faceindex,
                                                                 e.numIndices,
                                                                 e.indices);
        return e;
    }
};

#endif
