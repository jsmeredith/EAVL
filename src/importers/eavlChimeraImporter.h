// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
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

    vector<string>      GetFieldList();
    int                 GetNumChunks() { return 1; }
    eavlDataSet      *GetMesh(int);
    eavlField *GetField(int,string);

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
