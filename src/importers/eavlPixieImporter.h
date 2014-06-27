// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_PIXIE_IMPORTER_H
#define EAVL_PIXIE_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"

#ifdef HAVE_ADIOS

#include "eavlADIOSImporter.h"

// ****************************************************************************
// Class:  eavlPixieImporter
//
// Purpose:
///   Import a Pixie Silo file.
///   \todo: this appears to be a bit out of date
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    July 19, 2011
//
// ****************************************************************************

class eavlPixieImporter : public eavlADIOSImporter
{
  public:
    eavlPixieImporter(const string &filename);
    virtual ~eavlPixieImporter();

    vector<string> GetFieldList();
    eavlDataSet*   GetMesh(int);

  protected:
};

#else

class eavlPixieImporter : public eavlMissingImporter
{
  public:
    eavlPixieImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
