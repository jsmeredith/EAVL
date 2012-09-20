// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlTransformMutator.h"
#include "eavlException.h"

#include "eavlMapOp_3_3.h"

eavlTransformMutator::eavlTransformMutator() : eavlMutator(), transform(),
    transformCoordinates(false), coordinateSystemIndex(0)
{
}

void
eavlTransformMutator::SetTransform(const eavlMatrix4x4 &m)
{
   transform = m;
}

const eavlMatrix4x4 &
eavlTransformMutator::GetTransform() const
{
    return transform;
}

void
eavlTransformMutator::SetTransformCoordinates(bool val)
{
    transformCoordinates = val;
}

bool
eavlTransformMutator::GetTransformCoordinates() const
{
    return transformCoordinates;
}

void
eavlTransformMutator::SetCoordinateSystemIndex(int index)
{
    if(index >= 0)
        coordinateSystemIndex = index;
}

int
eavlTransformMutator::GetCoordinateSystemIndex() const
{
    return coordinateSystemIndex;
}

struct TransformFunctor
{
  private:
    eavlMatrix4x4 transform;
  public:
    TransformFunctor(const eavlMatrix4x4 &M) : transform(M) { }

    EAVL_FUNCTOR void operator()(float x, float y, float z,
                                 float &ox, float &oy, float &oz)
    {
        avtVector3 pt(x,y,z);
        avtVector3 result = transform * pt;
        ox = result.v[0];
        oy = result.v[1];
        oz = result.v[2];
    }
};


void
eavlTransformMutator::Execute()
{
    eavlCoordinatesCartesian *old_coords =
        dynamic_cast<eavlCoordinatesCartesian*>(dataset->GetCoordinateSystem(coordinateSystemIndex));
    if (!old_coords)
        THROW(eavlException,"for the moment, assuming we've got cartesian axes in elevate....");

    // This handles linear transforms with a matrix. If we were doing something
    // fancier, we could be installing a new eavlCoordinateSystem subclass.

    if(GetTransformCoordinates())
    {
#define HOLY_MOLY
#ifdef HOLY_MOLY
        // input arrays are from the coordinates
        eavlCoordinates *cs = old_coords;
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
#endif
        const char *transformName = "transformed_coords";
        eavlFloatArray *out = new eavlFloatArray(transformName, 3,
                                                 inCells->GetNumCells());

        eavlExecutor::AddOperation(new eavlMapOp_3_3<TransformFunctor>(
                                          i0, i1, i2,
                                          eavlArrayWithLinearIndex(out, 0),
                                          eavlArrayWithLinearIndex(out, 1),
                                          eavlArrayWithLinearIndex(out, 2),
                                          TransformFunctor(transform)),
                                   "transforming coordinates");
        eavlExecutor::Go();

        eavlField *coordField = new eavlField(1, out, eavlField::ASSOC_POINTS);
        dataset->AddField(coordField);

        coords->SetAxis(0, new eavlCoordinateAxisField(transformName, 0));
        coords->SetAxis(1, new eavlCoordinateAxisField(transformName, 1));
        coords->SetAxis(2, new eavlCoordinateAxisField(transformName, 2));

        dataset->SetCoordinateSystem(coordinateSystemIndex, coords);
    }
    else
    {
cerr << "Installing new transform object. transform=" << transform << endl;

        eavlCoordinatesCartesianWithTransform *coords = NULL;
        if (old_coords->GetDimension() == 1)
        {
            coords = new eavlCoordinatesCartesianWithTransform(
                             dataset->GetLogicalStructure(),
                             eavlCoordinatesCartesian::X);
            coords->SetAxis(0, old_coords->GetAxis(0));
            coords->SetTransform(transform);
        }
        else if (old_coords->GetDimension() == 2)
        {
            coords = new eavlCoordinatesCartesianWithTransform(
                             dataset->GetLogicalStructure(),
                             eavlCoordinatesCartesian::X,
                             eavlCoordinatesCartesian::Y);
            coords->SetAxis(0, old_coords->GetAxis(0));
            coords->SetAxis(1, old_coords->GetAxis(1));
            coords->SetTransform(transform);
        }
        else if (old_coords->GetDimension() == 3)
        {
            coords = new eavlCoordinatesCartesianWithTransform(
                             dataset->GetLogicalStructure(),
                             eavlCoordinatesCartesian::X,
                             eavlCoordinatesCartesian::Y,
                             eavlCoordinatesCartesian::Z);
            coords->SetAxis(0, old_coords->GetAxis(0));
            coords->SetAxis(1, old_coords->GetAxis(1));
            coords->SetAxis(2, old_coords->GetAxis(2));
            coords->SetTransform(transform);
        }
        else
        {
            THROW(eavlException,"Unexpected number of dimensions");
        }
        dataset->SetCoordinateSystem(coordinateSystemIndex, coords);
    }
}

