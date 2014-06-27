// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_FILTER_H
#define EAVL_FILTER_H

#include "eavlCoordinates.h"
#include "eavlCellSet.h"
#include "eavlField.h"
#include "eavlDataSet.h"
#include "eavlException.h"

// ****************************************************************************
// Class:  eavlMutator
//
// Purpose:
///
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    April 11, 2012
//
// ****************************************************************************

class eavlMutator
{
  protected:
    eavlDataSet *dataset;
  public:
    eavlMutator() : dataset(NULL) { }
    virtual ~eavlMutator() { }
    virtual void SetDataSet(eavlDataSet *ds) { dataset = ds; }
    virtual void Execute() = 0;
};


// ****************************************************************************
// Class:  eavlFilter
//
// Purpose:
///
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    April 11, 2012
//
// ****************************************************************************

class eavlFilter
{
  protected:
    eavlDataSet *input;
    eavlDataSet *output;
  public:
    eavlFilter() : input(NULL), output(new eavlDataSet) { }
    virtual ~eavlFilter() { }
    virtual void SetInput(eavlDataSet *ds)
    {
        input = ds;
        ///\todo: this is probably not the best place for this!
        output->Clear();
    }
    virtual eavlDataSet *GetOutput(void)    { return output; }
    virtual void Execute() = 0;
};

class eavlMutatorFilter : public eavlFilter
{
  protected:
    eavlMutator *mutator;
  public:
    eavlMutatorFilter(eavlMutator *m)
        : eavlFilter()
    {
        mutator = m;
    }
    virtual void SetInput(eavlDataSet *)
    {
        THROW(eavlException,"unimplemented");
        // this is where we should make a read-only shallow copy of ds
        // and set that as the mutator's data set
    }
    virtual void Execute()
    {
        mutator->Execute();

        THROW(eavlException,"Unimplemented");
        // this is where we should swap the input data sets contents
        // into the output
    }
};

#endif
