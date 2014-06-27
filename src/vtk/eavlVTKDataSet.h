// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VTK_DATASET_H
#define EAVL_VTK_DATASET_H

class eavlDataSet;
class vtkDataSet;

vtkDataSet *ConvertEAVLToVTK(eavlDataSet *in);
eavlDataSet *ConvertVTKToEAVL(vtkDataSet *in);

#endif
