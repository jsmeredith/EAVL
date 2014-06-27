// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SURFACE_NORMAL_MUTATOR_H
#define EAVL_SURFACE_NORMAL_MUTATOR_H

#include "STL.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlSurfaceNormalMutator
//
// Purpose:
///   Add new cell field which is the face surface normal of those cells.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    September 2, 2011
//
// ****************************************************************************
class eavlSurfaceNormalMutator : public eavlMutator
{
  protected:
    string cellsetname;
  public:
    eavlSurfaceNormalMutator();
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    
    virtual void Execute();
};

#endif

