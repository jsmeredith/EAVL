// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BOV_IMPORTER_H
#define EAVL_BOV_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"
#include "eavlArray.h"

// ****************************************************************************
// Class:  eavlBOVImporter
//
// Purpose:
///   Import BOV data.
//
// Programmer:  Dave Pugmire
// Creation:    Febuary 9, 2012
//
// ****************************************************************************
class eavlBOVImporter : public eavlImporter
{
  public:
    eavlBOVImporter(const string &filename);
    ~eavlBOVImporter();

    vector<string>      GetFieldList();
    int                 GetNumChunks();
    eavlDataSet      *GetMesh(int);
    eavlField *GetField(int,string);

  private:
    void                ReadTOC(const string &filename);
    string              DataFileFromChunk(int);
    size_t              SizeOfDataType();
    
    int dataSize[3], brickSize[3], numComponents, numChunks;
    float brickOrigin[3], brickXAxis[3], brickYAxis[3], brickZAxis[3];
    string dataFilePattern, filePath, variable;
    bool nodalCentering, swapBytes, hasBoundaries;

    enum dataType
    {
        FLOAT, DOUBLE, BYTE, INT, SHORT
    };
    dataType dataT;

};

#endif

