// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_XGC_PARTICLE_IMPORTER_H
#define EAVL_XGC_PARTICLE_IMPORTER_H

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
// Class:  eavlXGCParticleImporter
//
// Purpose:
///   Import a XGC ADIOS restart file.
//
// Programmer:  James Kress
// Creation:    July 1, 2014
//
// ****************************************************************************

class eavlXGCParticleImporter : public eavlImporter
{
  public:
    eavlXGCParticleImporter(const string &filename);
    eavlXGCParticleImporter(const string &filename, 
                            ADIOS_READ_METHOD method, 
                            MPI_Comm comm, 
                            ADIOS_LOCKMODE mode, 
                            int timeout_sec,
                            int fromDataspaces
                           );
    virtual ~eavlXGCParticleImporter();

	int			   GetTimeStep();
    int            GetNumChunks(const std::string &mesh) {return 1;}
    int			   AdvanceTimeStep(int step, int timeout_sec);
    vector<string> GetMeshList();
    vector<string> GetFieldList(const std::string &mesh);
    vector<string> GetCellSetList(const std::string &mesh);
    
    eavlDataSet*   GetMesh(const string &name, int chunk);
    eavlField*     GetField(const string &name, const string &mesh, int chunk);
	
  protected:
    void Initialize();
    ADIOS_SELECTION *MakeSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c);
    ADIOS_FILE *fp;
	
    int             maxnum, enphase, inphase, nvars, retVal, numMPITasks, mpiRank;
    int	            totalIParticles, totalEParticles, timestep;
    double          time;
    MPI_Comm        comm;
    long long       emaxgid, imaxgid;    

    map<string, ADIOS_VARINFO *> iphase, igid, ephase, egid;
};

#else

class eavlXGCParticleImporter : public eavlMissingImporter
{
  public:
    eavlXGCParticleImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif //HAVE_ADIOS
#endif //EAVL_XGC_PARTICLE_IMPORTER_H
