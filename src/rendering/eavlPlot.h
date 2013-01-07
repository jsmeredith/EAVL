#ifndef EAVL_PLOT_H
#define EAVL_PLOT_H

#include "eavl.h"
#include "eavlCamera.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"

struct eavlPlot
{
    eavlDataSet  *data;
    string        colortable;
    int           cellset_index;
    int           variable_fieldindex;
    //int           variable_cellindex;
    eavlRenderer *pcRenderer;
    eavlRenderer *meshRenderer;

    eavlPlot()
        : data(NULL),
          colortable("default"),
          cellset_index(-1),
          variable_fieldindex(-1),
          pcRenderer(NULL),
          meshRenderer(NULL)
    {
    }
};

#endif
