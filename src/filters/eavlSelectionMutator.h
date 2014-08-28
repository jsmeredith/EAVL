// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SELECTION_MUTATOR_H
#define EAVL_SELECTION_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellComponents.h"
#include "eavlFilter.h"
#include "eavlArray.h"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

// ****************************************************************************
// Class:  eavlSelectionMutator
//
// Purpose:
///   Create a selection of an existing cell set based on values from
//    an exsting field and add them to an existing data set
///   as well as stripped copies of the vars.
///   Note: this version creates an explicit standalone cell set.
//
//    Use: Pass an array of unique values contained in the field that
//          you set with SetFie ld. Then a new cell set will be made that
//          corresponds to your selection. This operation is useful for
//          multi timestep files, or whenever tracking known elements is 
//          desired.
//
// Programmer:  James Kress
// Creation:    August 13, 2014
//
// ****************************************************************************
class eavlSelectionMutator : public eavlMutator
{
  protected:
    eavlIntArray *chosenElements;
    string fieldname, cellsetname;
    bool presorted;
  public:
    eavlSelectionMutator();
    void SetArray(eavlIntArray *selectionarray)
    {
        chosenElements = selectionarray;
    }
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    void SetInputPreSorted(bool val)
    {
        presorted = val;
    }
    
    virtual void Execute();
};

#endif

