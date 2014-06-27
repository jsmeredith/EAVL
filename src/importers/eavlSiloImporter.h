// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SILO_IMPORTER_H
#define EAVL_SILO_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

#ifdef HAVE_SILO

#include "silo.h"

// ****************************************************************************
// Class:  eavlSiloImporter
//
// Purpose:
///   Import a Silo file.
//
// Programmer:  Dave Pugmire
// Creation:    July 19, 2011
//
// Modifications:
//   Jeremy Meredith, Mon Jul 30 15:12:07 EDT 2012
//   Support for multiple meshes in a file.
//   Better support for multimeshes.
//
// ****************************************************************************

class eavlSiloImporter : public eavlImporter
{
  public:
    eavlSiloImporter(const string &filename);
    ~eavlSiloImporter();

    
    vector<string>      GetMeshList();
    int                 GetNumChunks(const std::string &mesh);
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh);

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

  private:
    void Import();

    void ReadFile(DBfile *file, string dir);
    void ReadQuadMeshes(DBfile *file, DBtoc *toc, string &dir);
    void ReadUCDMeshes(DBfile *file, DBtoc *toc, string &dir);
    void ReadPointMeshes(DBfile *file, DBtoc *toc, string &dir);
    void ReadMultiMeshes(DBfile *file, DBtoc *toc, string &dir);
    
    void ReadQuadVars(DBfile *file, DBtoc *toc, string &dir);
    void ReadUCDVars(DBfile *file, DBtoc *toc, string &dir);
    void ReadPointVars(DBfile *file, DBtoc *toc, string &dir);
    void ReadMultiVars(DBfile *file, DBtoc *toc, string &dir);
    void Print();

    eavlDataSet *GetMultiMesh(string nm, int chunk);
    eavlDataSet *GetQuadMesh(string nm);
    eavlDataSet *GetUCDMesh(string nm);
    eavlDataSet *GetPtMesh(string nm);
    
    DBfile *file;
    map<string, vector<string> > multiMeshes, multiVars;
    vector<string> quadMeshes, ucdMeshes, ptMeshes;
    vector<string> quadVars, ucdVars, ptVars;
    map<string, string> meshForVar;
    set<string> meshesToHide;

    ///\todo: a bit of a hack; maybe we want to change the importers
    ///       to explicitly work this way, though.  for example,
    ///       set a mesh first, then read it, or a var
    eavlByteArray *ghosts_for_latest_mesh;
};


#else

class eavlSiloImporter : public eavlMissingImporter
{
  public:
    eavlSiloImporter(const string &) : eavlMissingImporter() { throw; }
};

#endif

#endif
