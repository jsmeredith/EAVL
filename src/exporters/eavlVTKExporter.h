// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
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

    void ExportStructured(ostream &out);
    void ExportUnstructured(ostream &out);

    void ExportPointCells(ostream &out);
    void ExportStructuredCells(ostream &out);
    void ExportUnstructuredCells(ostream &out);

    void ExportRectilinearCoords(ostream &out);
    void ExportPoints(ostream &out);

    void ExportFields(ostream &out);

    void ExportGlobalFields(ostream &out);

    int  CellTypeToVTK(eavlCellShape type);
    
};

#endif
