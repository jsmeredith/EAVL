// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_GRAPH_LAYOUT_FORCE_MUTATOR_H
#define EAVL_2D_GRAPH_LAYOUT_FORCE_MUTATOR_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlFilter.h"

// ****************************************************************************
// Class:  eavl2DGraphLayoutForceMutator
//
// Purpose:
///  Perform Fruchterman-Reingold style force-directed graph layout, where
///  all vertices have a repulsive force that diminishes with distance, an
///  attractive force that increases with the square of distance, and a
///  "gravity" (not mentioned in the 1990 paper) force that increases
///  linearly with distance to keep e.g. isolated subgraphs or nodes from
///  moving to infinity.  Since we have no hard constraints on final
///  coordinate extent, gravity works better than the hard bounding box
///  algorithms suggested in the original paper, allowing these isolated
///  components to retain a natural shape while keeping a distance from
///  other components.
///
///  The current "cooling" strategy is to provide an start distance
///  (limiting nodal motion during the first iteration) and a final
///  distance (limiting nodal motion during the last iteration), and
///  the algorithm will exponentially decrease the limiting distance
///  during the given number of iterations from the start to the final
///  distance.
///
///  This version of the algorithm is only 2D, and overwrites or
///  adds a 2D Cartesian coordinate system.
//
// Programmer:  Jeremy Meredith
// Creation:    May 28, 2013
//
// ****************************************************************************
class eavl2DGraphLayoutForceMutator : public eavlMutator
{
  protected:
    string cellsetname;
    int niter;
    double startdist;
    double finaldist;
    double areaconstant;
    double gravityconstant;
  public:
    eavl2DGraphLayoutForceMutator();
    void SetCellSet(const string &name)
    {
        cellsetname = name;
    }
    void SetNumIterations(int n)
    {
        niter = n;
    }
    void SetStartDist(double d)
    {
        startdist = d;
    }
    void SetFinalDist(double d)
    {
        finaldist = d;
    }
    void SetAreaConstant(double a)
    {
        areaconstant = a;
    }
    void SetGravityConstant(double g)
    {
        gravityconstant = g;
    }

    virtual void Execute();
};

#endif
