// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_XGC_IMPORTER_H
#define EAVL_XGC_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#ifdef HAVE_ADIOS

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

// ****************************************************************************
// Class:  eavlXGCImporter
//
// Purpose:
///   Import a XGC ADIOS file.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    July 19, 2011
//
// ****************************************************************************

class eavlXGCImporter : public eavlImporter
{
  public:
    eavlXGCImporter(const string &filename);
    eavlXGCImporter(   const string &filename,
                                                    ADIOS_READ_METHOD method,
                                                    MPI_Comm communicator,
                                                    ADIOS_LOCKMODE mode,
                                                    int timeout_sec,
                                                    int turnOnMesh
                                                );
    virtual ~eavlXGCImporter();

    int            GetNumChunks(const std::string &mesh) {return 1;}
    vector<string> GetMeshList();
    vector<string> GetFieldList(const std::string &mesh);
    vector<string> GetCellSetList(const std::string &mesh);
    
    eavlDataSet*   GetMesh(const string &name, int chunk);
    int            GetTimeStep();
    int            GetNumPlanes();
    int            GetMyPlaneCount();
    int            AdvanceTimeStep(int step, int timeout_sec);
    void           ReleaseTimeStep();
    void           SetReaderOptions(int turnOnPrintReadSizes);
    eavlField*     GetField(const string &name, const string &mesh, int chunk);

  protected:
    void Initialize();
    bool Supported(ADIOS_VARINFO *avi);
    ADIOS_SELECTION *MakeSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c);
    ADIOS_SELECTION *MakeLimitedSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c, int threeD);
    ADIOS_FILE *fp;
    ADIOS_FILE *mesh_fp;
    MPI_Comm        comm;
    int           numMPITasks, mpiRank, startIndex, endIndex, timestep, myTotalPlanes;
    int nNodes, nPlanes, nElems, meshOn, printReadSizes;

    map<string, ADIOS_VARINFO *> variables, mesh;
    ADIOS_VARINFO *points, *cells, *nextNode;
};

#else

class eavlXGCImporter : public eavlMissingImporter
{
  public:
    eavlXGCImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif //HAVE_ADIOS
#endif //EAVL_XGC_IMPORTER_H
