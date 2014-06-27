// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BINARY_MATH_MUTATOR_H
#define EAVL_BINARY_MATH_MUTATOR_H

#include "eavlDataSet.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlBinaryMathMutator
//
// Purpose:
///  Add a field as the third spatial dimension of a spatially 2D grid.
//
// Programmer:  Jeremy Meredith
// Creation:    September 10, 2012
//
// ****************************************************************************
class eavlBinaryMathMutator : public eavlMutator
{
  public:
    enum OpType { Add, Subtract, Multiply, Divide };
  public:
    eavlBinaryMathMutator();
    void SetField1(const string &name)
    {
        fieldname1 = name;
    }
    void SetField2(const string &name)
    {
        fieldname2 = name;
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
    string fieldname1;
    string fieldname2;
    string resultname;
    OpType optype;
};

#endif
