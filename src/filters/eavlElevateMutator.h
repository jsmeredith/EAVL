// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ELEVATE_MUTATOR_H
#define EAVL_ELEVATE_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlElevateMutator
//
// Purpose:
///  Add a field as the third spatial dimension of a spatially 2D grid.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 3, 2012
//
// ****************************************************************************
class eavlElevateMutator : public eavlMutator
{
  protected:
    string fieldname;
    string cellsetname;
  public:
    eavlElevateMutator();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }

    virtual void Execute();
};

#endif
