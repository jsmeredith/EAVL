// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_DESTINATION_TOPOLOGY_MAP_OP_H
#define EAVL_DESTINATION_TOPOLOGY_MAP_OP_H

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
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

template <class CONN>
struct eavlDestinationTopologyMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs, F &functor)
    {
        int ids[MAX_LOCAL_TOPOLOGY_IDS];
#pragma omp parallel for private(ids)
        for (int index = 0; index < nitems; ++index)
        {
            int nids;
            int shapeType = conn.GetElementComponents(index, nids, ids);

            collect(index, outputs) = functor(shapeType, nids, ids, collect(index, inputs));
        }
    }
};

#if defined __CUDACC__

template <class CONN, class F, class IN, class OUT>
__global__ void
eavlDestinationTopologyMapOp_kernel(int nitems, CONN conn,
                  const IN inputs, OUT outputs, F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int ids[MAX_LOCAL_TOPOLOGY_IDS];
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int nids;
        int shapeType = conn.GetElementComponents(index, nids, ids);

        collect(index, outputs) = functor(shapeType, nids, ids, collect(index, inputs));
    }
}

template <class CONN>
struct eavlDestinationTopologyMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlDestinationTopologyMapOp_kernel<<< blocks, threads >>>(nitems, conn,
                                                            inputs, outputs, functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif

// ****************************************************************************
// Class:  eavlDestinationTopologyMapOp
//
// Purpose:
///   Map from input to output arrays on the same topology type.
///
///   Much like a standard map, in that it does an element-wise map
///   between arrays of the same length, but the fields are known to
///   be on some sort of topology type -- typically a cell.  Or, you
///   could instead think of it like an eavlTopologyMap, but both the
///   inputs and outputs are on the same topology type.  (Or like an
///   eavlCombinedTopologyMap, but without a source topology.)
///
///   Essentially, this just adds a "shapetype" to the functor call of
///   a standard map operation.  For example, a cell-to-cell map would
///   be a simple map, but with the shape type (e.g. EAVL_HEX or
///   EAVL_TET) passed along with every functor call.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************
template <class I, class O, class F>
class eavlDestinationTopologyMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    I            inputs;
    O            outputs;
    F            functor;
  public:
    eavlDestinationTopologyMapOp(eavlCellSet *c, eavlTopology t,
                          I i, O o, F f)
        : cells(c), topology(t), inputs(i), outputs(o), functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.length();
        if (elExp)
        {
            eavlExplicitDestination &conn = elExp->GetDestination(topology);
            eavlOpDispatch<eavlDestinationTopologyMapOp_CPU<eavlExplicitDestination> >(n, conn, inputs, outputs, functor);
        }
        else if (elStr)
        {
            eavlRegularDestination conn = eavlRegularDestination(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlDestinationTopologyMapOp_CPU<eavlRegularDestination> >(n, conn, inputs, outputs, functor);
        }
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.length();
        if (elExp)
        {
            eavlExplicitDestination &conn = elExp->GetDestination(topology);

            conn.shapetype.NeedOnDevice();
            conn.destination.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();

            eavlOpDispatch<eavlDestinationTopologyMapOp_GPU<eavlExplicitDestination> >(n, conn, inputs, outputs, functor);

            conn.shapetype.NeedOnHost();
            conn.destination.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularDestination conn = eavlRegularDestination(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlDestinationTopologyMapOp_GPU<eavlRegularDestination> >(n, conn, inputs, outputs, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class F>
eavlDestinationTopologyMapOp<I,O,F> *new_eavlDestinationTopologyMapOp(eavlCellSet *c, eavlTopology t,
                                                         I i, O o, F f) 
{
    return new eavlDestinationTopologyMapOp<I,O,F>(c,t,i,o,f);
}

#endif
