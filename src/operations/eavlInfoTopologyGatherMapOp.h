// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_INFO_TOPOLOGY_GATHER_MAP_OP_H
#define EAVL_INFO_TOPOLOGY_GATHER_MAP_OP_H

#include "eavlCUDA.h"
#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlOperation.h"
#include "eavlTopology.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

#ifndef DOXYGEN

template <class CONN>
struct eavlInfoTopologyGatherMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int *sparseindices = get<0>(indices).array;

        for (int denseindex = 0; denseindex < nitems; ++denseindex)
        {
            int sparseindex = sparseindices[denseindex];
            int shapeType = conn.GetShapeType(sparseindex);
            collect(denseindex, outputs) = functor(shapeType, collect(sparseindex, inputs));
        }
    }
};

#if defined __CUDACC__

template <class CONN>
struct eavlInfoTopologyGatherMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN0 inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        cerr << "IMPLEMENT ME!\n";
        ///\todo: implement!
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlInfoTopologyGatherMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with
///   input arrays on the source topology (at sparsely indexed locations)
///   and with outputs on the destination topology.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************
template <class I, class O, class INDEX, class F>
class eavlInfoTopologyGatherMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    I            inputs;
    O            outputs;
    INDEX        indices;
    F            functor;
  public:
    eavlInfoTopologyGatherMapOp(eavlCellSet *c, eavlTopology t,
                            I i, O o, INDEX ind, F f)
        : cells(c), topology(t), inputs(i), outputs(o), indices(ind), functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.array->GetNumberOfTuples();
        if (elExp)
        {
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);
            eavlOpDispatch<eavlInfoTopologyGatherMapOp_CPU<eavlExplicitConnectivity> >(n, conn, inputs, outputs, indices, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlInfoTopologyGatherMapOp_CPU<eavlRegularConnectivity> >(n, conn, inputs, outputs, indices, functor);
        }
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.array->GetNumberOfTuples();
        if (elExp)
        {
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);

            conn.shapetype.NeedOnDevice();
            conn.connectivity.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();

            eavlOpDispatch<eavlInfoTopologyGatherMapOp_GPU<eavlExplicitConnectivity> >(n, conn, inputs, outputs, indices, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlInfoTopologyGatherMapOp_GPU<eavlRegularConnectivity> >(n, conn, inputs, outputs, indices, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class INDEX, class F>
eavlInfoTopologyGatherMapOp<I,O,INDEX,F> *new_eavlInfoTopologyGatherMapOp(eavlCellSet *c, eavlTopology t,
                                                                   I i, O o, INDEX indices, F f) 
{
    return new eavlInfoTopologyGatherMapOp<I,O,INDEX,F>(c,t,i,o,indices,f);
}


#endif
