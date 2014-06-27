// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlBinaryMathMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlMapOp.h"

eavlBinaryMathMutator::eavlBinaryMathMutator()
{
}


void
eavlBinaryMathMutator::Execute()
{
    eavlField *field1 = dataset->GetField(fieldname1);
    eavlField *field2 = dataset->GetField(fieldname2);

    if (field1->GetArray()->GetNumberOfComponents() != 1 ||
        field2->GetArray()->GetNumberOfComponents() != 1)
    {
        THROW(eavlException,
              "eavlBinaryMathMutator expects single-component fields");
    }

    int n = field1->GetArray()->GetNumberOfTuples();

    if (n != field2->GetArray()->GetNumberOfTuples())
    {
        THROW(eavlException,
              "eavlBinaryMathMutator expects two arrays with same length");
    }

    eavlFloatArray *result = new eavlFloatArray(resultname, 1, n);

    switch (optype)
    {
      case Add:
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(field1->GetArray(),
                                                            field2->GetArray()),
                                                 eavlOpArgs(result),
                                                 eavlAddFunctor<float>()),
            "binary addition");
        break;
      case Subtract:
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(field1->GetArray(),
                                                            field2->GetArray()),
                                                 eavlOpArgs(result),
                                                 eavlSubFunctor<float>()),
            "binary subtraction");
        break;
      case Multiply:
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(field1->GetArray(),
                                                            field2->GetArray()),
                                                 eavlOpArgs(result),
                                                 eavlMulFunctor<float>()),
            "binary multiplication");
        break;
      case Divide:
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(field1->GetArray(),
                                                            field2->GetArray()),
                                                 eavlOpArgs(result),
                                                 eavlDivFunctor<float>()),
            "binary division");
        break;
    }
    eavlExecutor::Go();

    // copy association, order, etc. from first field
    eavlField *newfield = new eavlField(field1, result);
    dataset->AddField(newfield);
}
