// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_INFO_TOPOLOGY_MAP_OP_H
#define EAVL_INFO_TOPOLOGY_MAP_OP_H

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
struct eavlInfoTopologyMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs, F &functor)
    {
#pragma omp parallel for
        for (int index = 0; index < nitems; ++index)
        {
            int shapeType = conn.GetShapeType(index);
            collect(index, outputs) = functor(shapeType, collect(index, inputs));
        }
    }
};

#if defined __CUDACC__

template <class CONN, class F, class IN, class OUT>
__global__ void
eavlInfoTopologyMapOp_kernel(int nitems, CONN conn,
                  const IN inputs, OUT outputs, F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int shapeType = conn.GetShapeType(index);
        collect(index, outputs) = functor(shapeType, collect(index, inputs));
    }
}

template <class CONN>
struct eavlInfoTopologyMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlInfoTopologyMapOp_kernel<<< blocks, threads >>>(nitems, conn,
                                                            inputs, outputs, functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif

// ****************************************************************************
// Class:  eavlInfoTopologyMapOp
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
class eavlInfoTopologyMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    I            inputs;
    O            outputs;
    F            functor;
  public:
    eavlInfoTopologyMapOp(eavlCellSet *c, eavlTopology t,
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
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);
            eavlOpDispatch<eavlInfoTopologyMapOp_CPU<eavlExplicitConnectivity> >(n, conn, inputs, outputs, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlInfoTopologyMapOp_CPU<eavlRegularConnectivity> >(n, conn, inputs, outputs, functor);
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
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);

            conn.shapetype.NeedOnDevice();
            conn.connectivity.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();

            eavlOpDispatch<eavlInfoTopologyMapOp_GPU<eavlExplicitConnectivity> >(n, conn, inputs, outputs, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlInfoTopologyMapOp_GPU<eavlRegularConnectivity> >(n, conn, inputs, outputs, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class F>
eavlInfoTopologyMapOp<I,O,F> *new_eavlInfoTopologyMapOp(eavlCellSet *c, eavlTopology t,
                                                         I i, O o, F f) 
{
    return new eavlInfoTopologyMapOp<I,O,F>(c,t,i,o,f);
}

#endif
