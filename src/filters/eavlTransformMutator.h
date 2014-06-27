// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TRANSFORM_MUTATOR_H
#define EAVL_TRANSFORM_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"
#include "eavlMatrix4x4.h"

// ****************************************************************************
// Class:  eavlTransformMutator
//
// Purpose:
///  Add a field as the third spatial dimension of a spatially 2D grid.
//
// Programmer:  Brad Whitlock
// Creation:    September 19, 2012
//
// ****************************************************************************
class eavlTransformMutator : public eavlMutator
{
  protected:
    eavlMatrix4x4 transform; // we could build this from rotations, scales, translates but use a matrix for now.
    bool          transformCoordinates;
    int           coordinateSystemIndex;

  public:
    eavlTransformMutator();
    void SetTransform(const eavlMatrix4x4 &m);
    const eavlMatrix4x4 &GetTransform() const;
    void SetTransformCoordinates(bool);
    bool GetTransformCoordinates() const;
    void SetCoordinateSystemIndex(int);
    int  GetCoordinateSystemIndex() const;

    virtual void Execute();
};

#endif
