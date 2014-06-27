// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_NETCDF_IMPORTER_H
#define EAVL_NETCDF_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#ifdef HAVE_NETCDF

#include <netcdfcpp.h>

// ****************************************************************************
// Class:  eavlNetCDFImporter
//
// Purpose:
///   Import NetCDF files as eavlDataSets.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    May  5, 2011
//
// ****************************************************************************
class eavlNetCDFImporter : public eavlImporter
{
  public:
    eavlNetCDFImporter(const string &filename);
    ~eavlNetCDFImporter();

    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh) { return vector<string>(1,"RectilinearGridCells"); }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);
  protected:
    NcFile *file;
    vector<NcVar*> vars;
    vector<NcDim*> dims;
    char buff[4096];
    char bufforig[4096];
};

#else

class eavlNetCDFImporter : public eavlMissingImporter
{
  public:
    eavlNetCDFImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
