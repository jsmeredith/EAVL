// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_NETCDF_DECOMPOSING_IMPORTER_H
#define EAVL_NETCDF_DECOMPOSING_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#ifdef HAVE_NETCDF

#include <netcdfcpp.h>

// ****************************************************************************
// Class:  eavlNetCDFDecomposingImporter
//
// Purpose:
///   Import NetCDF files as eavlDataSets.
///   This version automatically splits the single mesh into
///   a number of smaller chunks (i.e. automatic domain decomposition).
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    May  5, 2011
//
// ****************************************************************************
class eavlNetCDFDecomposingImporter : public eavlImporter
{
  public:
    eavlNetCDFDecomposingImporter(int numdomains,
                                  const string &filename);
    ~eavlNetCDFDecomposingImporter();

    int                 GetNumChunks(const std::string &mesh);
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh) { return vector<string>(1,"RectilinearGridCells"); }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);
  protected:
    int numchunks;
    NcFile *file;
    vector<NcVar*> vars;
    vector<NcDim*> dims;
    char buff[4096];
    char bufforig[4096];
};

#else

class eavlNetCDFDecomposingImporter : public eavlMissingImporter
{
  public:
    eavlNetCDFDecomposingImporter(int numdomains,
                             const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
