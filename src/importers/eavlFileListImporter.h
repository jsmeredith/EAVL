// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_FILE_LIST_IMPORTER_H
#define EAVL_FILE_LIST_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"
#include "eavlChimeraImporter.h"
#include "eavlVTKImporter.h"
#include "eavlNetCDFImporter.h"
#include "eavlNetCDFDecomposingImporter.h"
#include "eavlSiloImporter.h"
#include "eavlArray.h"

static string right(string &s, int len)
{
    if (len >= s.length())
        return s;
    return s.substr(s.length() - len);
}

// ****************************************************************************
// Class:  eavlFileListImporter
//
// Purpose:
///   Takes a pile of single-timestep files and treats them as a single
///   time-varying sequence.
//
// Programmer:  Jeremy Meredith
// Creation:    December 28. 2011
//
// Modifications:
// ****************************************************************************
class eavlFileListImporter : public eavlImporter
{
  public:
    eavlFileListImporter(const int ndom, // need domains for netcdf
                         const vector<string> &files)
    {
        numdomains = ndom;
        filenames = files;
        importer = NULL;
    }
    virtual vector<string> GetDiscreteDimNames()
    {
        vector<string> retval;
        retval.push_back("time");
    }
    virtual vector<int>    GetDiscreteDimLengths()
    {
        vector<int> retval;
        retval.push_back(filenames.size());
    }
    vector<string>      GetFieldList()
    {
        return importer->GetFieldList();
    }
    int                 GetNumChunks()
    {
        return importer->GetNumChunks();
    }
    eavlDataSet      *GetMesh(int c)
    {
        return importer->GetMesh(c);
    }
    eavlField *GetField(int c,string v)
    {
        return importer->GetField(c,v);
    }

    virtual void           SetDiscreteDim(int d, int i)
    {
        if (d != 0) // only one discrete dim for now: time
            throw; 
        if (importer)
            delete importer;

        string filename = filenames[i];
        if (right(filename,4) == ".vtk")
        {
            importer = new eavlVTKImporter(filename);
        }
#ifdef HAVE_NETCDF
        else if (right(filename,3) == ".nc")
        {
            //importer = new eavlNetCDFImporter(filename.toStdString());
            importer = new eavlNetCDFDecomposingImporter(numdomains,
                                                         filename);
        }
#endif
#ifdef HAVE_HDF5
        else if (right(filename,3) == ".h5")
        {
            cerr << "Error: HDF5 not implemented yet\n";
            return;
        }
#endif
#ifdef HAVE_SILO
        else if (right(filename,5) == ".silo")
        {
            importer = new eavlSiloImporter(filename);
        }
        else if (right(filename,4) == ".chi")
        {
            importer = new eavlChimeraImporter(filename);
        }
#endif
        else
        {
            cerr << "Error: unknown file extension\n";
            return;
        }

    }

  protected:
    int numdomains;
    vector<string> filenames;
    eavlImporter *importer;
};

#endif
