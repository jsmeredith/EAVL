// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSurfaceNormalMutator.h"

#include "eavl.h"
#include "eavlFilter.h"
#include "eavlCellSet.h"
#include "eavlField.h"
#include "eavlDataSet.h"
#include "eavlSourceTopologyMapOp.h"
#include "eavlField.h"
#include "eavlVector3.h"
#include "eavlException.h"
#include "eavlExecutor.h"

class FaceNormalFunctor
{
  public:
    template <class IN>
    EAVL_FUNCTOR tuple<float,float,float> operator()(int shapeType, int n, int ids[],
                                                     const IN coords)
    {
        // should we treat EAVL_PIXEL differently here?
        typename collecttype<IN>::const_type pt0 = collect(ids[0], coords);
        typename collecttype<IN>::const_type pt1 = collect(ids[1], coords);
        typename collecttype<IN>::const_type pt2 = collect(ids[2], coords);
        eavlPoint3 p0(get<0>(pt0), get<1>(pt0), get<2>(pt0));
        eavlPoint3 p1(get<0>(pt1), get<1>(pt1), get<2>(pt1));
        eavlPoint3 p2(get<0>(pt2), get<1>(pt2), get<2>(pt2));
        eavlVector3 v01 = p1 - p0;
        eavlVector3 v12 = p2 - p1;
        eavlVector3 norm = v01 % v12;
        norm.normalize();
        return tuple<float,float,float>(norm.x, norm.y, norm.z);
    }
};

eavlSurfaceNormalMutator::eavlSurfaceNormalMutator()
{
}

void
eavlSurfaceNormalMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);

    // input arrays are from the coordinates
    eavlCoordinates *cs = dataset->GetCoordinateSystem(0);
    if (cs->GetDimension() != 3)
        THROW(eavlException,"eavlNodeToCellOp assumes 3D coordinates");

    eavlIndexable<eavlArray> i0 = dataset->GetIndexableAxis(0, cs);
    eavlIndexable<eavlArray> i1 = dataset->GetIndexableAxis(1, cs);
    eavlIndexable<eavlArray> i2 = dataset->GetIndexableAxis(2, cs);

    eavlFloatArray *out = new eavlFloatArray("surface_normals", 3,
                                             inCells->GetNumCells());

    eavlExecutor::AddOperation(new_eavlSourceTopologyMapOp(
                                      inCells, EAVL_NODES_OF_CELLS,
                                      eavlOpArgs(i0, i1, i2),
                                      eavlOpArgs(eavlIndexable<eavlFloatArray>(out,0),
                                                 eavlIndexable<eavlFloatArray>(out,1),
                                                 eavlIndexable<eavlFloatArray>(out,2)),
                                      FaceNormalFunctor()),
                               "surface normal");
    eavlExecutor::Go();

    eavlField *cellnormalfield = new eavlField(0, out,
                                               eavlField::ASSOC_CELL_SET,
                                               cellsetname);
    dataset->AddField(cellnormalfield);
};
