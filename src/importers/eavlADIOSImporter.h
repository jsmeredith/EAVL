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

    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh) { return vector<string>(1,"RectilinearGridCells"); }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

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
