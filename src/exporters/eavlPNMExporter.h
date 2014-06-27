// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_PNMEXPORTER_H
#define EAVL_PNMEXPORTER_H

#include "STL.h"
#include "eavlArray.h"

// ****************************************************************************
// Class :  eavlPNMExporter
//
// Programmer:  Rob Sisneros
// Creation:    Aug 3, 2011
//
// ****************************************************************************

class eavlPNMExporter
{
  public:
    void Export(ostream &out, int, int, eavlByteArray *);
    void ConvertAndExport(ostream &out, int, int, eavlFloatArray *);
  protected:
};

#endif
