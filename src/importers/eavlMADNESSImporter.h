// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_MADNESS_IMPORTER_H
#define EAVL_MADNESS_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"
#include "eavlArray.h"

class eavlLogicalStructureQuadTree;

// ****************************************************************************
// Class:  eavlMADNESSImporter
//
// Purpose:
///   Import MADNESS data.
//
// Programmer:  Jeremy Meredith
// Creation:    January 31, 2012
//
// ****************************************************************************
class eavlMADNESSImporter : public eavlImporter
{
  public:
    eavlMADNESSImporter(const string &filename);
    eavlMADNESSImporter(const char *data, size_t len);
    ~eavlMADNESSImporter();
    int                 GetNumChunks(const std::string &mesh);
    vector<string>      GetFieldList(const std::string &mesh);
    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

  protected:
    eavlLogicalStructureQuadTree *log;
};

#endif