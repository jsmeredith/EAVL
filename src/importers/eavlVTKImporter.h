// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VTK_IMPORTER_H
#define EAVL_VTK_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"
#include "eavlArray.h"

// ****************************************************************************
// Class:  eavlVTKImporter
//
// Purpose:
///   Import binary and ASCII VTK "legacy" (*.vtk) data files into
///   an eavlDataSet.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 17, 2011
//
// ****************************************************************************
class eavlVTKImporter : public eavlImporter
{
  public:
    eavlVTKImporter(const string &filename);
    eavlVTKImporter(const char *data, size_t len);
    ~eavlVTKImporter();
    int                 GetNumChunks(const std::string &mesh) { return 1; }
    vector<string>      GetFieldList(const std::string &mesh);
    vector<string>      GetCellSetList(const std::string &mesh);

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);
  protected:
    enum DataType
    {
        dt_bit,
        dt_unsigned_char,
        dt_char,
        dt_unsigned_short,
        dt_short,
        dt_unsigned_int,
        dt_int,
        dt_unsigned_long,
        dt_long,
        dt_float,
        dt_double
    };

    enum DataSetType
    {
        DS_UNKNOWN,
        DS_STRUCTURED_POINTS,
        DS_STRUCTURED_GRID,
        DS_RECTILINEAR_GRID,
        DS_POLYDATA,
        DS_UNSTRUCTURED_GRID
    };

    istream *is;
    char buff[4096];
    char bufforig[4096];
    enum Location { LOC_DATASET, LOC_CELLS, LOC_POINTS };
    string      comment;
    bool        binary;
    DataSetType structure;

  protected:
    void Import();
    void ParseVersion();
    void ParseHeader();
    void ParseFormat();
    void ParseStructure();
    void ParseAttributes();

    void ParseFieldData(Location);
    void ParseScalars(Location);
    void ParseVectors(Location);
    void ParseNormals(Location);
    void ParsePoints(eavlLogicalStructureRegular*);

    void Parse_Structured_Points();
    void Parse_Structured_Grid();
    void Parse_Rectilinear_Grid();
    void Parse_Polydata();
    void Parse_Unstructured_Grid();
    void AddArray(eavlFloatArray *arr, eavlVTKImporter::Location loc);

    DataType DataTypeFromString(const string &s);
    string StringFromDataType(eavlVTKImporter::DataType dt);
    DataSetType DataSetTypeFromString(const string &s);
    string StringFromDataSetType(DataSetType dst);

    template <class T>
    void ReadIntoVector(int,DataType,vector<T>&);
    void ReadIntoArray(DataType, eavlArray *);
    bool GetNextLine();
  protected:
    vector<int> cell_to_cell_splitmap;
    eavlDataSet *data;
    map<string,eavlField*> vars;
};

#endif
