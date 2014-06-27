// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl3X3AverageMutator.h"

#include "eavl.h"
#include "eavlFilter.h"
#include "eavlCellSet.h"
#include "eavlField.h"
#include "eavlDataSet.h"
#include "eavl3X3NodeStencilOp_1_1.h"
#include "eavlField.h"
#include "eavlVector3.h"
#include "eavlException.h"
#include "eavlExecutor.h"

class AverageFunctor
{
  public:
    EAVL_FUNCTOR float operator()(float x[])
    {
        float average = 0;
        for(int i = 0; i < 9; i++)
            average += x[i];

        return average/9.0;
    }
};

eavl3X3AverageMutator::eavl3X3AverageMutator()
{
}


void
eavl3X3AverageMutator::Execute()
{
    //eavlField *inField = dataset->GetField(fieldname);

    // input arrays are from the coordinates
    eavlCoordinates *cs = dataset->GetCoordinateSystem(0);
    if (cs->GetDimension() != 2)
        THROW(eavlException,"2D coordinate system required.");

    eavlRegularStructure reg;
    eavlLogicalStructureRegular *logReg = dynamic_cast<eavlLogicalStructureRegular*>(dataset->GetLogicalStructure());
    if (logReg && logReg->GetDimension() == 2)
        reg = logReg->GetRegularStructure();
    else
        THROW(eavlException,"Expected 2D regular grid.");

    eavlFloatArray *out = new eavlFloatArray("3X3Average", 1,
                                             field->GetArray()->GetNumberOfTuples());

    eavlExecutor::AddOperation(new eavl3X3NodeStencilOp_1_1<AverageFunctor>(
                                      field,
                                      reg,
                                      field->GetArray(),
                                      eavlArrayWithLinearIndex(out, 0),
                                      AverageFunctor()),
                               "3x3 node stencil");
    eavlExecutor::Go();

    eavlField *averagefield = new eavlField(0, out,
                                            eavlField::ASSOC_POINTS);
    dataset->AddField(averagefield);
};
