// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BOX_MUTATOR_H
#define EAVL_BOX_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellComponents.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlBoxMutator
//
// Purpose:
///   Do a threshold with a range based on the coordiantes.
///   Create stripped copies of the variables, too.
///   Note: this version creates an explicit standalone cell set.
//
// Programmer:  Jeremy Meredith
// Creation:    December 17, 2013
//
// ****************************************************************************
class eavlBoxMutator : public eavlMutator
{
  protected:
    int dim;
    int ni, nj, nk;
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;
    string fieldname, cellsetname;
  public:
    eavlBoxMutator();
    virtual ~eavlBoxMutator();
    void SetRange1D(double xn, double xx)
    {
        dim = 1;
        xmin = xn;
        xmax = xx;
    }
    void SetRange2D(double xn, double xx,
                    double yn, double yx)
    {
        dim = 2;
        xmin = xn;
        xmax = xx;
        ymin = yn;
        ymax = yx;
    }
    void SetRange3D(double xn, double xx,
                    double yn, double yx,
                    double zn, double zx)
    {
        dim = 3;
        xmin = xn;
        xmax = xx;
        ymin = yn;
        ymax = yx;
        zmin = zn;
        zmax = zx;
    }
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    
    virtual void Execute();
};

#endif

