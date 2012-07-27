// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    int                 GetNumChunks() { return 1; }
    vector<string>      GetFieldList();
    eavlDataSet      *GetMesh(int);
    eavlField *GetField(int,string);
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
