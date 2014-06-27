// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CURVE_IMPORTER_H
#define EAVL_CURVE_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"

// ****************************************************************************
// Class:  eavlCurveImporter
//
// Purpose:
///   Import X/Y files -- two data columns and #'s as names/delimiters.
//
// Programmer:  Jeremy Meredith
// Creation:    February 11, 2013
//
// ****************************************************************************
class eavlCurveImporter : public eavlImporter
{
  public:
    eavlCurveImporter(const string &filename);
    ~eavlCurveImporter();

    vector<string>      GetMeshList();
    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh);

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);
  protected:
    std::map<std::string, std::vector<double> > curveX;
    std::map<std::string, std::vector<double> > curveY;
};

#endif
