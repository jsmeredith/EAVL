// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    eavlImporter() { }
    virtual ~eavlImporter() { }
    virtual vector<string> GetMeshList() { return vector<string>(1,"mesh"); }
    virtual vector<string> GetFieldList(const std::string &mesh) = 0;
    virtual vector<string> GetCellSetList(const std::string &mesh) = 0;
    virtual int            GetNumChunks(const std::string &mesh) = 0;

    virtual vector<string> GetDiscreteDimNames() { return vector<string>(); }
    virtual vector<int>    GetDiscreteDimLengths() { return vector<int>(); }
    virtual void           SetDiscreteDim(int, int) { }

    virtual eavlDataSet *GetMesh(const string &name,
                                 int chunk) = 0;
    virtual eavlField   *GetField(const string &name, const string &mesh,
                                  int chunk) = 0;
};

class eavlMissingImporter : public eavlImporter
{
  public:
    virtual vector<string> GetFieldList(const std::string &) { throw; }
    virtual int            GetNumChunks(const std::string &) { throw; }

    virtual eavlDataSet *GetMesh(const string&, int) { throw; }
    virtual eavlField   *GetField(const string&, const string&, int) { throw; }
};

#endif
