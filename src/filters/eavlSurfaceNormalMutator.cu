// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSurfaceNormalMutator.h"

#include "eavl.h"
#include "eavlFilter.h"
#include "eavlCellSet.h"
#include "eavlField.h"
#include "eavlDataSet.h"
#include "eavlSimpleTopologyMapOp.h"
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

    eavlCoordinateAxisField *axis0 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(0));
    eavlCoordinateAxisField *axis1 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(1));
    eavlCoordinateAxisField *axis2 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(2));

    if (!axis0 || !axis1 || !axis2)
        THROW(eavlException,"eavlNodeToCellOp expects only field-based coordinate axes");

    eavlField *field0 = dataset->GetField(axis0->GetFieldName());
    eavlField *field1 = dataset->GetField(axis1->GetFieldName());
    eavlField *field2 = dataset->GetField(axis2->GetFieldName());
    eavlArray *arr0 = field0->GetArray();
    eavlArray *arr1 = field1->GetArray();
    eavlArray *arr2 = field2->GetArray();
    if (!arr0 || !arr1 || !arr2)
    {
        THROW(eavlException,"eavlNodeToCellOp assumes single-precision float arrays");
    }

    eavlIndexable<eavlArray> i0(arr0, axis0->GetComponent());
    eavlIndexable<eavlArray> i1(arr1, axis1->GetComponent());
    eavlIndexable<eavlArray> i2(arr2, axis2->GetComponent());
    if (field0->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i0.indexer.mul = 0;
    if (field1->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i1.indexer.mul = 0;
    if (field2->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i2.indexer.mul = 0;
    
    eavlLogicalStructureRegular *logReg = dynamic_cast<eavlLogicalStructureRegular*>(dataset->GetLogicalStructure());
    if (logReg)
    {
        eavlRegularStructure &reg = logReg->GetRegularStructure();

        if (field0->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i0 = eavlIndexable<eavlArray>(arr0, axis0->GetComponent(), reg, field0->GetAssocLogicalDim());
        if (field1->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i1 = eavlIndexable<eavlArray>(arr1, axis1->GetComponent(), reg, field1->GetAssocLogicalDim());
        if (field2->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i2 = eavlIndexable<eavlArray>(arr2, axis2->GetComponent(), reg, field2->GetAssocLogicalDim());
    }

    eavlFloatArray *out = new eavlFloatArray("surface_normals", 3,
                                             inCells->GetNumCells());

    eavlExecutor::AddOperation(new_eavlSimpleTopologyMapOp(
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
                                               inCellSetIndex);
    dataset->AddField(cellnormalfield);
};
