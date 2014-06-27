// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ISOSURFACE_FILTER_H
#define EAVL_ISOSURFACE_FILTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"
#include "eavlArray.h"

// ****************************************************************************
// Class:  eavlIsosurfaceFilter
//
// Purpose:
///  Generate a triangle-mesh isosurface from volumetric elements.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 3, 2012
//
// ****************************************************************************
class eavlIsosurfaceFilter : public eavlFilter
{
  protected:
    string fieldname;
    string cellsetname;
    double value;

    eavlByteArray *hiloArray;
    eavlByteArray *caseArray;
    eavlIntArray *numoutArray;
    eavlIntArray *outindexArray;
    eavlIntArray *totalout;
    eavlIntArray *edgeInclArray;
    eavlIntArray *outpointindexArray;
    eavlIntArray *totaloutpts;

  public:
    eavlIsosurfaceFilter();
    virtual ~eavlIsosurfaceFilter();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    void SetIsoValue(double val)
    {
        value = val;
    }
    
    virtual void Execute();
};

#endif
