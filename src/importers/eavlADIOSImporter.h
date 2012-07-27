// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ADIOS_IMPORTER_H
#define EAVL_ADIOS_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#if HAVE_ADIOS

#include "ADIOSFileObject.h"

// ****************************************************************************
// Class:  eavlADIOSImporter
//
// Purpose:
///   Import a basic ADIOS file.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    July 19, 2011
//
// ****************************************************************************

class eavlADIOSImporter : public eavlImporter
{
  public:
    eavlADIOSImporter(const string &filename);
    virtual ~eavlADIOSImporter();

    virtual vector<string>  GetFieldList();
    virtual int             GetNumChunks() { return 1; }
    virtual eavlDataSet    *GetMesh(int);
    virtual eavlField      *GetField(int,string);

  protected:
    void               Import();
    virtual eavlDataSet    *CreateRectilinearGrid(const ADIOSVar &v);
    
    ADIOSFileObject *file;
};

#else

class eavlADIOSImporter : public eavlMissingImporter
{
  public:
    eavlADIOSImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
