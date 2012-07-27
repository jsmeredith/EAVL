// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VTKEXPORTER_H
#define EAVL_VTKEXPORTER_H

#include "STL.h"
#include "eavlExporter.h"
#include "eavlDataSet.h"

// ****************************************************************************
// Class :  eavlVTKExporter
//
// Programmer:  Dave Pugmire
// Creation:    May 17, 2011
//
// ****************************************************************************

class eavlVTKExporter : public eavlExporter
{
  public:
    eavlVTKExporter(eavlDataSet *data_, int which_cells = 0) :
        eavlExporter(data_), cellSetIndex(which_cells)
    {}
    virtual void Export(ostream &out);
    
  protected:

    int cellSetIndex;

    void ExportUnstructured(ostream &out);
    void ExportPoints(ostream &out);
    void ExportCells(ostream &out);
    void ExportFields(ostream &out);

    int  CellTypeToVTK(eavlCellShape &type);
    
};

#endif
