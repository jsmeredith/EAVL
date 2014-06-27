// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SOURCE_TOPOLOGY_MAP_OP_H
#define EAVL_SOURCE_TOPOLOGY_MAP_OP_H

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
struct eavlSourceTopologyMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs, F &functor)
    {
        int ids[MAX_LOCAL_TOPOLOGY_IDS];
#pragma omp parallel for private(ids)
        for (int index = 0; index < nitems; ++index)
        {
            int nids;
            int shapeType = conn.GetElementComponents(index, nids, ids);

            collect(index, outputs) = functor(shapeType, nids, ids, s_inputs);
        }
    }
};

#if defined __CUDACC__

template <class CONN, class F, class IN, class OUT>
__global__ void
eavlSourceTopologyMapOp_kernel(int nitems, CONN conn,
                               const IN s_inputs, OUT outputs, F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int ids[MAX_LOCAL_TOPOLOGY_IDS];
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int nids;
        int shapeType = conn.GetElementComponents(index, nids, ids);

        collect(index, outputs) = functor(shapeType, nids, ids, s_inputs);
    }
}

template <class CONN>
struct eavlSourceTopologyMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlSourceTopologyMapOp_kernel<<< blocks, threads >>>(nitems, conn,
                                                              s_inputs, outputs, functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif

// ****************************************************************************
// Class:  eavlSourceTopologyMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with
///   input arrays on the source topology and with outputs
///   on the destination topology.  (If you need inputs on the
///   destination topology as well, use eavlCombinedTopologyMap.)
//
// Programmer:  Jeremy Meredith
// Creation:    July 26, 2013
//
// Modifications:
// ****************************************************************************
template <class IS, class O, class F>
class eavlSourceTopologyMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    O            outputs;
    F            functor;
  public:
    eavlSourceTopologyMapOp(eavlCellSet *c, eavlTopology t,
                      IS is, O o, F f)
        : cells(c), topology(t), s_inputs(is), outputs(o), functor(f)
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
            eavlOpDispatch<eavlSourceTopologyMapOp_CPU<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlSourceTopologyMapOp_CPU<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, functor);
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

            eavlOpDispatch<eavlSourceTopologyMapOp_GPU<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlSourceTopologyMapOp_GPU<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class IS, class O, class F>
eavlSourceTopologyMapOp<IS,O,F> *new_eavlSourceTopologyMapOp(eavlCellSet *c, eavlTopology t,
                                                 IS is, O o, F f) 
{
    return new eavlSourceTopologyMapOp<IS,O,F>(c,t,is,o,f);
}


#endif
