// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_GRAPH_LAYOUT_MUTATOR_H
#define EAVL_2D_GRAPH_LAYOUT_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavl2DGraphLayoutMutator
//
// Purpose:
///  Add a field as the third spatial dimension of a spatially 2D grid.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 3, 2012
//
// ****************************************************************************
class eavl2DGraphLayoutMutator : public eavlMutator
{
  protected:
    string cellsetname;
    int niter;
    double startdist;
    double finaldist;
  public:
    eavl2DGraphLayoutMutator();
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    void SetNumIterations(int n)
    {
        niter = n;
    }
    void SetStartDist(double d)
    {
        startdist = d;
    }
    void SetFinalDist(double d)
    {
        finaldist = d;
    }

    virtual void Execute();
};

#endif
