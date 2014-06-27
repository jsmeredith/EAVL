// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_UNARY_MATH_MUTATOR_H
#define EAVL_UNARY_MATH_MUTATOR_H

#include "eavlDataSet.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlUnaryMathMutator
//
// Purpose:
///  Apply a unary math operation.
//
// Programmer:  Dave Pugmire
// Creation:    May 10, 2013
//
// ****************************************************************************

class eavlUnaryMathMutator : public eavlMutator
{
  public:
    enum OpType { Negate, Square, Cube, Log_10, Log_2, Ln, SquareRoot };
  public:
    eavlUnaryMathMutator();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void SetResultName(const string &name)
    {
        resultname = name;
    }
    void SetOperation(OpType op)
    {
        optype = op;
    }

    virtual void Execute();

  protected:
    string fieldname;
    string resultname;
    OpType optype;
};

#endif
