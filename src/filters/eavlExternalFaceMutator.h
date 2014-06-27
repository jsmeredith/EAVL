// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_EXTERNAL_FACE_MUTATOR_H
#define EAVL_EXTERNAL_FACE_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlExternalFaceMutator
//
// Purpose:
///   Extract non-duplicated (external) faces from a topologically 3D data set.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March 14, 2011
//
// ****************************************************************************
class eavlExternalFaceMutator : public eavlMutator
{
  protected:
    string cellsetname;
  public:
    eavlExternalFaceMutator();
    virtual ~eavlExternalFaceMutator();
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    
    virtual void Execute();
};

#endif
