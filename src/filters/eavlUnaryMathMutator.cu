// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlUnaryMathMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlMapOp.h"

template<class T>
struct eavlNegateFunctor
{
    EAVL_FUNCTOR T operator()(T value) { return -value; }
};

template<class T>
struct eavlSquareFunctor
{
    EAVL_FUNCTOR T operator()(T value) { return value*value; }
};

template<class T>
struct eavlSquareRootFunctor
{
    EAVL_FUNCTOR T operator()(T value) { return sqrt(value); }
};

template<class T>
struct eavlCubeFunctor
{
    EAVL_FUNCTOR T operator()(T value) { return value*value*value; }
};

template<class T>
struct eavlLog10Functor
{
    EAVL_FUNCTOR T operator()(T value) { return log10(value); }
};

template<class T>
struct eavlLog2Functor
{
    EAVL_FUNCTOR T operator()(T value) { return log2(value); }
};

template<class T>
struct eavlLnFunctor
{
    EAVL_FUNCTOR T operator()(T value) { return log(value); }
};

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

    eavlFloatArray *result = new eavlFloatArray(resultname, 1, n);

    switch (optype)
    {
      case Negate:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlNegateFunctor<float>()),
           "negate");
        break;


      case Square:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlSquareFunctor<float>()),
           "square");
        break;

      case SquareRoot:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlSquareRootFunctor<float>()),
           "square_root");
        break;

      case Cube:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlCubeFunctor<float>()),
           "cube");
        break;

      case Log_10:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlLog10Functor<float>()),
           "log10");
        break;

      case Log_2:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlLog2Functor<float>()),
           "log2");
        break;

      case Ln:
        eavlExecutor::AddOperation(
           new_eavlMapOp(eavlOpArgs(field->GetArray()),
                         eavlOpArgs(result),
                         eavlLnFunctor<float>()),
           "ln");
        break;

        ///\todo: implement the other functors and remove the default:throw
      default:
        THROW(eavlException, "Unimplemented");

    }
    eavlExecutor::Go();

    // copy association, order, etc. from first field
    eavlField *newfield = new eavlField(field, result);
    dataset->AddField(newfield);
}
