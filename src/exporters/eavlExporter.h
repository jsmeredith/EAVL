// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_EXPORTER_H
#define EAVL_EXPORTER_H

#include "eavlDataSet.h"

// ****************************************************************************
// Class :  eavlVTKExporter
//
// Programmer:  Dave Pugmire
// Creation:    May 17, 2011
//
// ****************************************************************************

class eavlExporter
{
  public:
    eavlExporter(eavlDataSet *data_)
    {
        data = data_;
    }

    virtual void Export(ostream &out) = 0;

  protected:
    eavlDataSet *data;
};

#endif
