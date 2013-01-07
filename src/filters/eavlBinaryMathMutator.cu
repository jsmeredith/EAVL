// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlBinaryMathMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlMapOp_2_1.h"

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

    eavlArray *result = new eavlFloatArray(resultname, 1, n);

    switch (optype)
    {
      case Add:
        eavlExecutor::AddOperation(
            new eavlMapOp_2_1<eavlAddFunctor<float> >(field1->GetArray(),
                                                      field2->GetArray(),
                                                      result,
                                                      eavlAddFunctor<float>()),
            "binary addition");
        break;
      case Subtract:
        eavlExecutor::AddOperation(
            new eavlMapOp_2_1<eavlSubFunctor<float> >(field1->GetArray(),
                                                      field2->GetArray(),
                                                      result,
                                                      eavlSubFunctor<float>()),
            "binary subtraction");
        break;
      case Multiply:
        eavlExecutor::AddOperation(
            new eavlMapOp_2_1<eavlMulFunctor<float> >(field1->GetArray(),
                                                      field2->GetArray(),
                                                      result,
                                                      eavlMulFunctor<float>()),
            "binary multiplication");
        break;
      case Divide:
        eavlExecutor::AddOperation(
            new eavlMapOp_2_1<eavlDivFunctor<float> >(field1->GetArray(),
                                                      field2->GetArray(),
                                                      result,
                                                      eavlDivFunctor<float>()),
            "binary division");
        break;
    }
    eavlExecutor::Go();

    // copy association, order, etc. from first field
    eavlField *newfield = new eavlField(field1, result);
    dataset->AddField(newfield);
}
