// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_TESSELATOR_H
#define EAVL_2D_TESSELATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"
#include "eavlArray.h"

// ****************************************************************************
// Class:  eavlTesselate2DFilter
//
// Purpose:
///  Tesselate each quad into 4 new quads.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 2, 2012
//
// ****************************************************************************
class eavlTesselate2DFilter : public eavlFilter
{
  protected:
    string cellsetname;
  public:
    eavlTesselate2DFilter();
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    
    virtual void Execute();
};

#endif
