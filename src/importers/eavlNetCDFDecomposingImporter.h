// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    int                 GetNumChunks();
    vector<string>      GetFieldList();
    eavlDataSet      *GetMesh(int);
    eavlField *GetField(int,string);
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
