// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSurfaceNormalMutator.h"

#include "eavl.h"
#include "eavlFilter.h"
#include "eavlCellSet.h"
#include "eavlField.h"
#include "eavlDataSet.h"
#include "eavlTopologyMapOp_3_0_3.h"
#include "eavlField.h"
#include "eavlVector3.h"
#include "eavlException.h"
#include "eavlExecutor.h"

class FaceNormalFunctor
{
  public:
    EAVL_FUNCTOR void operator()(int shapeType, int n,
                                 float x[], float y[], float z[],
                                 float &ox, float &oy, float &oz)
    {
        // should we treat EAVL_PIXEL differently here?
        eavlPoint3 p0(x[0], y[0], z[0]);
        eavlPoint3 p1(x[1], y[1], z[1]);
        eavlPoint3 p2(x[2], y[2], z[2]);
        eavlVector3 v01 = p1 - p0;
        eavlVector3 v12 = p2 - p1;
        eavlVector3 norm = v01 % v12;
        norm.normalize();
	    
        ox = norm.x;
        oy = norm.y;
        oz = norm.z;
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

    eavlArrayWithLinearIndex i0(arr0, axis0->GetComponent());
    eavlArrayWithLinearIndex i1(arr1, axis1->GetComponent());
    eavlArrayWithLinearIndex i2(arr2, axis2->GetComponent());
    if (field0->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i0.mul = 0;
    if (field1->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i1.mul = 0;
    if (field2->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i2.mul = 0;
    
    eavlLogicalStructureRegular *logReg = dynamic_cast<eavlLogicalStructureRegular*>(dataset->GetLogicalStructure());
    if (logReg)
    {
        eavlRegularStructure &reg = logReg->GetRegularStructure();

        if (field0->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i0 = eavlArrayWithLinearIndex(arr0, axis0->GetComponent(), reg, field0->GetAssocLogicalDim());
        if (field1->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i1 = eavlArrayWithLinearIndex(arr1, axis1->GetComponent(), reg, field1->GetAssocLogicalDim());
        if (field2->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i2 = eavlArrayWithLinearIndex(arr2, axis2->GetComponent(), reg, field2->GetAssocLogicalDim());
    }

    eavlFloatArray *out = new eavlFloatArray("surface_normals", 3,
                                             inCells->GetNumCells());

    eavlExecutor::AddOperation(new eavlTopologyMapOp_3_0_3<FaceNormalFunctor>(
                                      inCells, EAVL_NODES_OF_CELLS,
                                      i0, i1, i2,
                                      eavlArrayWithLinearIndex(out, 0),
                                      eavlArrayWithLinearIndex(out, 1),
                                      eavlArrayWithLinearIndex(out, 2),
                                      FaceNormalFunctor()),
                               "surface normal");
    eavlExecutor::Go();

    eavlField *cellnormalfield = new eavlField(0, out,
                                               eavlField::ASSOC_CELL_SET,
                                               inCellSetIndex);
    dataset->AddField(cellnormalfield);
};
