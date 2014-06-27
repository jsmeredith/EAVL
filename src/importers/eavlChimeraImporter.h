// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CHIMERA_IMPORTER_H
#define EAVL_CHIMERA_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#ifdef HAVE_SILO

#include "silo.h"

// ****************************************************************************
// Class:  eavlChimeraImporter
//
// Purpose:
///   Import a Chimera Silo file.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    July 19, 2011
//
// ****************************************************************************

class eavlChimeraImporter : public eavlImporter
{
  public:
    eavlChimeraImporter(const string &filename);
    ~eavlChimeraImporter();

    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh) { return vector<string>(1,"chimera_Cells"); }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

  private:
    void Import();
    
    DBfile *file;
    eavlDataSet *data;
};


#else

class eavlChimeraImporter : public eavlMissingImporter
{
  public:
    eavlChimeraImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
