// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COMBINED_TOPOLOGY_PACKED_MAP_OP_H
#define EAVL_COMBINED_TOPOLOGY_PACKED_MAP_OP_H

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
struct eavlCombinedTopologyPackedMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN0, class IN1, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, const IN1 d_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int *sparseindices = get<0>(indices).array;

        int ids[MAX_LOCAL_TOPOLOGY_IDS]; // these are effectively our src indices
#pragma omp parallel for private(ids)
        for (int denseindex = 0; denseindex < nitems; ++denseindex)
        {
            int sparseindex = sparseindices[get<0>(indices).indexer.index(denseindex)];

            int nids;
            int shapeType = conn.GetElementComponents(sparseindex, nids, ids);

            typename collecttype<IN1>::const_type in_d(collect(denseindex, d_inputs));
            typename collecttype<OUT>::type out(collect(denseindex, outputs));

            out = functor(shapeType, nids, ids, s_inputs, in_d);
        }
    }
};

#if defined __CUDACC__

template <class CONN, class F, class IN0, class IN1, class OUT, class INDEX>
__global__ void
eavlCombinedTopologyPackedMapOp_kernel(int nitems, CONN conn,
                                       const IN0 s_inputs, const IN1 d_inputs, OUT outputs,
                                       INDEX indices, F functor)
{
    int *sparseindices = get<0>(indices).array;

    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int ids[MAX_LOCAL_TOPOLOGY_IDS];
    for (int denseindex = threadID; denseindex < nitems; denseindex += numThreads)
    {
        int sparseindex = sparseindices[get<0>(indices).indexer.index(denseindex)];

        int nids;
        int shapeType = conn.GetElementComponents(sparseindex, nids, ids);

        collect(denseindex, outputs) = functor(shapeType, nids, ids, s_inputs,
                                          collect(denseindex, d_inputs));
    }
}


template <class CONN>
struct eavlCombinedTopologyPackedMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN0, class IN1, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, const IN1 d_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlCombinedTopologyPackedMapOp_kernel<<< blocks, threads >>>(nitems, conn,
                                                                      s_inputs, d_inputs, outputs,
                                                                      indices, functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif

// ****************************************************************************
// Class:  eavlCombinedTopologyPackedMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with input
///   arrays on the source topology (at sparsely indexed locations) and the
///   destination topology, and with outputs on the destination topology.
///   In this packed version of the operation, the inputs on the destination
///   topology and the outputs are both compacted, i.e. densely indexed from
///   0 to n-1.
//
// Programmer:  Jeremy Meredith
// Creation:    August  2, 2013
//
// Modifications:
// ****************************************************************************
template <class IS, class ID, class O, class INDEX, class F>
class eavlCombinedTopologyPackedMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    ID           d_inputs;
    O            outputs;
    INDEX        indices;
    F            functor;
  public:
    eavlCombinedTopologyPackedMapOp(eavlCellSet *c, eavlTopology t,
                                    IS is, ID id, O o, INDEX ind, F f)
        : cells(c), topology(t), s_inputs(is), d_inputs(id), outputs(o), indices(ind), functor(f)
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
            eavlOpDispatch<eavlCombinedTopologyPackedMapOp_CPU<eavlExplicitConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlCombinedTopologyPackedMapOp_CPU<eavlRegularConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);
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

            eavlOpDispatch<eavlCombinedTopologyPackedMapOp_GPU<eavlExplicitConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlCombinedTopologyPackedMapOp_GPU<eavlRegularConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class IS, class ID, class O, class INDEX, class F>
eavlCombinedTopologyPackedMapOp<IS,ID,O,INDEX,F> *new_eavlCombinedTopologyPackedMapOp(eavlCellSet *c, eavlTopology t,
                                                                                      IS is, ID id, O o, INDEX indices, F f) 
{
    return new eavlCombinedTopologyPackedMapOp<IS,ID,O,INDEX,F>(c,t,is,id,o,indices,f);
}


#endif
