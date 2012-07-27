// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_IMPORTER_H
#define EAVL_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"

// ****************************************************************************
// Class:  eavlImporter
//
// Purpose:
///   Base class for a data source importer.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    July  8, 2011
//
// ****************************************************************************
class eavlImporter
{
  public:
    //virtual vector<string> GetMeshList()       = 0;
    virtual vector<string> GetFieldList()      = 0;
    virtual int            GetNumChunks()      = 0;

    virtual vector<string> GetDiscreteDimNames() { return vector<string>(); }
    virtual vector<int>    GetDiscreteDimLengths() { return vector<int>(); }
    virtual void           SetDiscreteDim(int d, int i) { }

    virtual eavlDataSet      *GetMesh(int)   = 0;
    virtual eavlField *GetField(int,string) = 0;
};

class eavlMissingImporter : public eavlImporter
{
  public:
    virtual vector<string> GetFieldList() { throw; }
    virtual int            GetNumChunks() { throw; }

    virtual eavlDataSet      *GetMesh(int) { throw; }
    virtual eavlField *GetField(int,string) { throw; }
};

#endif
