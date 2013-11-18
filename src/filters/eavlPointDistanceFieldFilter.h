// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_POINT_DISTANCE_FIELD_FILTER_H
#define EAVL_POINT_DISTANCE_FIELD_FILTER_H

#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavlPointDistanceFieldFilter
//
// Purpose:
///   Find the distance field for a set of point locations.
//
// Programmer:  Jeremy Meredith
// Creation:    November 18, 2013
//
// Modifications:
// ****************************************************************************
class eavlPointDistanceFieldFilter : public eavlFilter
{
  public:
    eavlPointDistanceFieldFilter();
    void SetRange1D(int numi,
                    double xn, double xx)
    {
        dim = 1;
        ni = numi;
        xmin = xn;
        xmax = xx;
    }
    void SetRange2D(int numi, int numj,
                    double xn, double xx,
                    double yn, double yx)
    {
        dim = 2;
        ni = numi;
        nj = numj;
        xmin = xn;
        xmax = xx;
        ymin = yn;
        ymax = yx;
    }
    void SetRange3D(int numi, int numj, int numk,
                    double xn, double xx,
                    double yn, double yx,
                    double zn, double zx)
    {
        dim = 3;
        ni = numi;
        nj = numj;
        nk = numk;
        xmin = xn;
        xmax = xx;
        ymin = yn;
        ymax = yx;
        zmin = zn;
        zmax = zx;
    }
    virtual void Execute();
  protected:
    int dim;
    int ni, nj, nk;
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;
};

#endif
