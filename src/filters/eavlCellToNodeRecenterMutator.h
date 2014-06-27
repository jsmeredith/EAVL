// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_TO_NODE_RECENTER_MUTATOR_H
#define EAVL_CELL_TO_NODE_RECENTER_MUTATOR_H
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlCellToNodeRecenterMutator
//
// Purpose:
///   Recenters a field from a cellset to the mesh points.
//
// Programmer:  Jeremy Meredith
// Creation:    November 28, 2012
//
// Modifications:
// ****************************************************************************
class eavlCellToNodeRecenterMutator : public eavlMutator
{
  public:
    eavlCellToNodeRecenterMutator();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    virtual void Execute();
  protected:
    string fieldname;
    string cellsetname;
};

#endif
