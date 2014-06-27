// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_3X3_AVERAGE_MUTATOR_H
#define EAVL_3X3_AVERAGE_MUTATOR_H

#include "STL.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavl3X3AverageMutator
//
// Purpose:
///   3X3 Stencil Average
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    June 25, 2012
//
// ****************************************************************************
class eavl3X3AverageMutator : public eavlMutator
{
  protected:
    eavlField *field;
  public:
    eavl3X3AverageMutator();
    void SetField(eavlField *inField)
    {
        field = inField;
    }
    
    virtual void Execute();
};

#endif

