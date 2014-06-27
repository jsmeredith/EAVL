// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_PDB_IMPORTER_H
#define EAVL_PDB_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

// ****************************************************************************
// Class:  eavlPDBImporter
//
// Purpose:
///   Import Protein Data Bank files.
//
// Programmer:  Jeremy Meredith
// Creation:    July 26, 2012
//
// Modifications:
// ****************************************************************************
class eavlPDBImporter : public eavlImporter
{
  public:
    eavlPDBImporter(const string &filename);
    ~eavlPDBImporter();

    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh) { return vector<string>(1,"bonds"); }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);
  protected:
    struct Atom
    {
        int   serial;
        char  name[5];
        char  altloc;
        char  resname[4];
        char  chainid;
        int   resseq;
        char  icode;
        float x;
        float y;
        float z;
        float occupancy;
        float tempfactor;
        char  segid[5];
        char  element[3];
        char  charge[3];

        int   atomicnumber;
        int   residuenumber;
        bool  backbone;

        int   compound;

        Atom(const char *line, int compound=0);
        void Print(ostream &out);
    };


    struct ConnectRecord
    {
        int a;
        int b[4];

        ConnectRecord(const char *line);
        void Print(ostream &out);
    };

    ifstream in;

    bool metadata_read;
    int  nmodels;
    std::vector< std::vector<Atom> >    allatoms;
    std::vector< std::pair<int, int> >  bonds;

    std::vector<ConnectRecord>       connect;
    std::vector<std::string>         compoundNames;

    std::string filename;
    std::string dbTitle;


    static bool AtomsShouldBeBonded(const vector<Atom> &atoms, int a1, int a2);
    void OpenFileAtBeginning();
    void ReadAllMetaData();
    void ReadAtomsForModel(int);
    void CreateBondsFromModel(int);
    void CreateBondsFromModel_Slow(int);
    void CreateBondsFromModel_Fast(int);
};

#endif
