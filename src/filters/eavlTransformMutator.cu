// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlTransformMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"

#include "eavlMapOp.h"

#include "eavlMatrix4x4.h"
#include "eavlVector3.h"

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

struct TransformFunctor1
{
  private:
    eavlMatrix4x4 transform;
  public:
    TransformFunctor1(const eavlMatrix4x4 &M) : transform(M) { }

    EAVL_FUNCTOR float operator()(float x)
    {
        eavlVector3 pt(x,0.,0.);
        eavlVector3 result = transform * pt;
        return result.x;
    }
};

struct TransformFunctor2
{
  private:
    eavlMatrix4x4 transform;
  public:
    TransformFunctor2(const eavlMatrix4x4 &M) : transform(M) { }

    EAVL_FUNCTOR tuple<float,float> operator()(tuple<float,float> c)
    {
        eavlVector3 pt(get<0>(c), get<1>(c), 0.);
        eavlVector3 result = transform * pt;
        return tuple<float,float>(result.x, result.y);
    }
};

struct TransformFunctor3
{
  private:
    eavlMatrix4x4 transform;
  public:
    TransformFunctor3(const eavlMatrix4x4 &M) : transform(M) { }

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<float,float,float> c)
    {
        eavlVector3 pt(get<0>(c), get<1>(c), get<2>(c));
        eavlVector3 result = transform * pt;
        return tuple<float,float,float>(result.x, result.y, result.z);
    }
};

#if 0
void
Transform1D(eavlDataSet *dataset, eavlCoordinates *cs, const eavlMatrix4x4 transform)
{
    eavlCoordinateAxisField *axis0 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(0));
    
    if (!axis0)
        THROW(eavlException,"Transform1D expects only field-based coordinate axes");

    eavlField *field0 = dataset->GetField(axis0->GetFieldName());
    eavlArray *arr0 = field0->GetArray();
    if (!arr0)
    {
        THROW(eavlException,"Transform1D assumes single-precision float arrays");
    }

    eavlArrayWithLinearIndex i0(arr0, axis0->GetComponent());
    if (field0->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
        i0.mul = 0;

    eavlLogicalStructureRegular *logReg = dynamic_cast<eavlLogicalStructureRegular*>(dataset->GetLogicalStructure());
    if (logReg)
    {
        eavlRegularStructure &reg = logReg->GetRegularStructure();

        if (field0->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            i0 = eavlArrayWithLinearIndex(arr0, axis0->GetComponent(), reg, field0->GetAssocLogicalDim());
    }

    const char *transformName = "transformed_coords";
    eavlFloatArray *out = new eavlFloatArray(transformName, 1,
                                             dataset->GetNumPoints());

    eavlExecutor::AddOperation(new eavlMapOp_1_1<TransformFunctor1>(
                                      i0,
                                      eavlArrayWithLinearIndex(out, 0),
                                      TransformFunctor1(transform)),
                               "transforming 1d coordinates");
    eavlExecutor::Go();

    eavlField *coordField = new eavlField(1, out, eavlField::ASSOC_POINTS);
    dataset->AddField(coordField);

    cs->SetAxis(0, new eavlCoordinateAxisField(transformName, 0));
}
#endif

template <typename Functor>
void
Transform2D(Functor func, eavlDataSet *dataset, eavlCoordinates *cs, 
    eavlCoordinateAxisField *axis0, eavlCoordinateAxisField *axis1,
    eavlField *field0, eavlField *field1,
    eavlArray *arr0, eavlArray *arr1)
{
    eavlIndexable<eavlArray> i0 = dataset->GetIndexableAxis(0, cs);
    eavlIndexable<eavlArray> i1 = dataset->GetIndexableAxis(1, cs);

    const char *transformName = "transformed_coords";
    eavlFloatArray *out = new eavlFloatArray(transformName, 2,
                                             dataset->GetNumPoints());

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(i0, i1),
                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(out, 0),
                                                        eavlIndexable<eavlFloatArray>(out, 1)),
                                             func),
                               "transforming coordinates");
    eavlExecutor::Go();

    eavlField *coordField = new eavlField(1, out, eavlField::ASSOC_POINTS);
    dataset->AddField(coordField);

    cs->SetAxis(0, new eavlCoordinateAxisField(transformName, 0));
    cs->SetAxis(1, new eavlCoordinateAxisField(transformName, 1));
}

template <typename Functor>
void
Transform3D(Functor func, eavlDataSet *dataset, eavlCoordinates *cs, 
    eavlCoordinateAxisField *axis0, eavlCoordinateAxisField *axis1, eavlCoordinateAxisField *axis2,
    eavlField *field0, eavlField *field1, eavlField *field2,
    eavlArray *arr0, eavlArray *arr1, eavlArray *arr2)
{
    eavlIndexable<eavlArray> i0 = dataset->GetIndexableAxis(0, cs);
    eavlIndexable<eavlArray> i1 = dataset->GetIndexableAxis(1, cs);
    eavlIndexable<eavlArray> i2 = dataset->GetIndexableAxis(2, cs);

    const char *transformName = "transformed_coords";
    eavlFloatArray *out = new eavlFloatArray(transformName, 3,
                                             dataset->GetNumPoints());

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(i0, i1, i2),
                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(out, 0),
                                                        eavlIndexable<eavlFloatArray>(out, 1),
                                                        eavlIndexable<eavlFloatArray>(out, 2)),
                                             func),
                               "transforming coordinates");
    eavlExecutor::Go();

    eavlField *coordField = new eavlField(1, out, eavlField::ASSOC_POINTS);
    dataset->AddField(coordField);

    cs->SetAxis(0, new eavlCoordinateAxisField(transformName, 0));
    cs->SetAxis(1, new eavlCoordinateAxisField(transformName, 1));
    cs->SetAxis(2, new eavlCoordinateAxisField(transformName, 2));
}

void
eavlTransformMutator::Execute()
{
    eavlCoordinatesCartesian *cs =
        dynamic_cast<eavlCoordinatesCartesian*>(dataset->GetCoordinateSystem(coordinateSystemIndex));
    if (!cs)
        THROW(eavlException,"for the moment, assuming we've got cartesian axes in elevate....");

    // This handles linear transforms with a matrix. If we were doing something
    // fancier, we could be installing a new eavlCoordinateSystem subclass.

    if(GetTransformCoordinates())
    {
#if 0
        if (cs->GetDimension() == 1)
            Transform1D(dataset, cs, transform);
        else
#endif
        if (cs->GetDimension() == 2)
        {
            eavlCoordinateAxisField *axis0 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(0));
            eavlCoordinateAxisField *axis1 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(1));
    
            if (!axis0 || !axis1)
                THROW(eavlException,"Transform2D expects only field-based coordinate axes");

            eavlField *field0 = dataset->GetField(axis0->GetFieldName());
            eavlField *field1 = dataset->GetField(axis1->GetFieldName());
            eavlArray *arr0 = field0->GetArray();
            eavlArray *arr1 = field1->GetArray();
            if (!arr0 || !arr1)
            {
                THROW(eavlException,"Transform3D assumes single-precision float arrays");
            }

            Transform2D(TransformFunctor2(transform), dataset, cs,
                        axis0, axis1,
                        field0, field1,
                        arr0, arr1);
        }
        else if(cs->GetDimension() == 3)
        {
            eavlCoordinateAxisField *axis0 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(0));
            eavlCoordinateAxisField *axis1 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(1));
            eavlCoordinateAxisField *axis2 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(2));
    
            if (!axis0 || !axis1 || !axis2)
                THROW(eavlException,"Transform3D expects only field-based coordinate axes");

            eavlField *field0 = dataset->GetField(axis0->GetFieldName());
            eavlField *field1 = dataset->GetField(axis1->GetFieldName());
            eavlField *field2 = dataset->GetField(axis2->GetFieldName());
            eavlArray *arr0 = field0->GetArray();
            eavlArray *arr1 = field1->GetArray();
            eavlArray *arr2 = field2->GetArray();
            if (!arr0 || !arr1 || !arr2)
            {
                THROW(eavlException,"Transform3D assumes single-precision float arrays");
            }

            Transform3D(TransformFunctor3(transform), dataset, cs,
                        axis0, axis1, axis2,
                        field0, field1, field2,
                        arr0, arr1, arr2);
        }
    }
    else
    {
        eavlCoordinatesCartesianWithTransform *coords = NULL;
        if (cs->GetDimension() == 1)
        {
            coords = new eavlCoordinatesCartesianWithTransform(
                             dataset->GetLogicalStructure(),
                             eavlCoordinatesCartesian::X);
            coords->SetAxis(0, cs->GetAxis(0));
            coords->SetTransform(transform);
        }
        else if (cs->GetDimension() == 2)
        {
            coords = new eavlCoordinatesCartesianWithTransform(
                             dataset->GetLogicalStructure(),
                             eavlCoordinatesCartesian::X,
                             eavlCoordinatesCartesian::Y);
            coords->SetAxis(0, cs->GetAxis(0));
            coords->SetAxis(1, cs->GetAxis(1));
            coords->SetTransform(transform);
        }
        else if (cs->GetDimension() == 3)
        {
            coords = new eavlCoordinatesCartesianWithTransform(
                             dataset->GetLogicalStructure(),
                             eavlCoordinatesCartesian::X,
                             eavlCoordinatesCartesian::Y,
                             eavlCoordinatesCartesian::Z);
            coords->SetAxis(0, cs->GetAxis(0));
            coords->SetAxis(1, cs->GetAxis(1));
            coords->SetAxis(2, cs->GetAxis(2));
            coords->SetTransform(transform);
        }
        else
        {
            THROW(eavlException,"Unexpected number of dimensions");
        }
        dataset->SetCoordinateSystem(coordinateSystemIndex, coords);
    }
}

