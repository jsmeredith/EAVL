// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_PNG_IMPORTER_H
#define EAVL_PNG_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"


// ****************************************************************************
// Class:  eavlPNGImporter
//
// Purpose:
///   Import PNG files as regular grids.
//
// Programmer:  Jeremy Meredith
// Creation:    January  7, 2013
// ****************************************************************************
class eavlPNGImporter : public eavlImporter
{
  public:
    eavlPNGImporter(const string &filename);
    eavlPNGImporter(const unsigned char *buffer, long long size);
    ~eavlPNGImporter();

    int                 GetNumChunks(const std::string &) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh);

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

  protected:
    unsigned int width, height;
    std::vector<unsigned char> rgba;
};


#endif
