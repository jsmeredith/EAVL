// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.

#ifndef EAVL_SCALAR_BIN_FILTER_H
#define EAVL_SCALAR_BIN_FILTER_H

#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlScalarBinFilter
//
// Purpose:
///   Essentially a histogram; bins a scalar and counts the frequency
///   across some set of equally spaced bins.
//
// Programmer:  Jeremy Meredith
// Creation:    January 17, 2013
//
// Modifications:
// ****************************************************************************
class eavlScalarBinFilter : public eavlFilter
{
  public:
    eavlScalarBinFilter();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void SetNumBins(int n)
    {
        nbins = n;
    }
    virtual void Execute();
  protected:
    int nbins;
    string fieldname;
};

#endif
