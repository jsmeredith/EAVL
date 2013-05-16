// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlUnaryMathMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlMapOp_1_1.h"

eavlUnaryMathMutator::eavlUnaryMathMutator()
{
}


void
eavlUnaryMathMutator::Execute()
{
    eavlField *field = dataset->GetField(fieldname);

    if (field->GetArray()->GetNumberOfComponents() != 1)
    {
        THROW(eavlException,
              "eavlUnaryMathMutator expects single-component fields");
    }

    int n = field->GetArray()->GetNumberOfTuples();

    eavlArray *result = new eavlFloatArray(resultname, 1, n);

    switch (optype)
    {
      case Negate:
        eavlExecutor::AddOperation(
	   new eavlMapOp_1_1<eavlNegateFunctor<float> >(field->GetArray(),
						     result,
						     eavlNegateFunctor<float>()),
	   "negate");
        break;

      case Square:
        eavlExecutor::AddOperation(
	   new eavlMapOp_1_1<eavlSquareFunctor<float> >(field->GetArray(),
						     result,
						     eavlSquareFunctor<float>()),
	   "square");
        break;

      case Cube:
        eavlExecutor::AddOperation(
	   new eavlMapOp_1_1<eavlCubeFunctor<float> >(field->GetArray(),
						     result,
						     eavlCubeFunctor<float>()),
	   "cube");
        break;

      case Log_10:
        eavlExecutor::AddOperation(
	   new eavlMapOp_1_1<eavlLog10Functor<float> >(field->GetArray(),
						     result,
						     eavlLog10Functor<float>()),
	   "Log10");
        break;

      case Log_2:
        eavlExecutor::AddOperation(
	   new eavlMapOp_1_1<eavlLog2Functor<float> >(field->GetArray(),
						     result,
						     eavlLog2Functor<float>()),
	   "Log2");
        break;

      case Ln:
        eavlExecutor::AddOperation(
	   new eavlMapOp_1_1<eavlLnFunctor<float> >(field->GetArray(),
						     result,
						     eavlLnFunctor<float>()),
	   "Ln");
        break;


    }
    eavlExecutor::Go();

    // copy association, order, etc. from first field
    eavlField *newfield = new eavlField(field, result);
    dataset->AddField(newfield);
}
