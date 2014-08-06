// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ADIOS_IMPORTER_H
#define EAVL_ADIOS_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#if HAVE_ADIOS
//NOTE: #include <mpi.h> *MUST* become before the adios includes.
#ifdef HAVE_MPI
#include <mpi.h>
#else
#define _NOMPI
#endif

extern "C"
{
#include <adios_read.h>
}

//#include "ADIOSFileObject.h"

// ****************************************************************************
// Class:  eavlADIOSImporter
//
// Purpose:
///   Import a basic ADIOS file.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    July 19, 2011
//
// ****************************************************************************

class eavlADIOSImporter : public eavlImporter
{
  public:
    eavlADIOSImporter(const string &filename);
    virtual ~eavlADIOSImporter();

    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetMeshList();
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh)
    { return vector<string>(1,"RectilinearGridCells"); }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

  protected:
    void Initialize();
    
    bool Supported(ADIOS_VARINFO *avi);
    int NumBytes(ADIOS_VARINFO *avi);
    int NumTuples(ADIOS_VARINFO *avi);
    string MeshName(ADIOS_VARINFO *avi);

    struct meshData {
	string nm;
	int dim;
	int dims[3];
	vector<string> vars;
    };

    map<string, meshData> metadata;
    ADIOS_FILE *fp;
};

#else

class eavlADIOSImporter : public eavlMissingImporter
{
  public:
    eavlADIOSImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
