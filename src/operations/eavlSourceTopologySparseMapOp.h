// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SOURCE_TOPOLOGY_SPARSE_MAP_OP_H
#define EAVL_SOURCE_TOPOLOGY_SPARSE_MAP_OP_H

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
struct eavlSourceTopologySparseMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs,
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

            typename collecttype<OUT>::type out(collect(sparseindex, outputs));

            out = functor(shapeType, nids, ids, s_inputs);
        }
    }
};

#if defined __CUDACC__

template <class CONN, class F, class IN, class OUT, class INDEX>
__global__ void
eavlSourceTopologyGatherMapOp_kernel(int nitems, CONN conn,
                                     const IN s_inputs, OUT outputs,
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

        collect(sparseindex, outputs) = functor(shapeType, nids, ids, s_inputs);
    }
}


template <class CONN>
struct eavlSourceTopologyGatherMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlSourceTopologyGatherMapOp_kernel<<< blocks, threads >>>(nitems, conn,
                                                                    s_inputs, outputs,
                                                                    indices, functor);
        CUDA_CHECK_ERROR();
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlSourceTopologySparseMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with
///   input arrays on the source topology and with outputs on the destination
///   topology.  All input and output arrays are indexed sparsely as
///   specified by the index array.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************
template <class IS, class O, class INDEX, class F>
class eavlSourceTopologySparseMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    O            outputs;
    INDEX        indices;
    F            functor;
  public:
    eavlSourceTopologySparseMapOp(eavlCellSet *c, eavlTopology t,
                            IS is, O o, INDEX ind, F f)
        : cells(c), topology(t), s_inputs(is), outputs(o), indices(ind), functor(f)
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
            eavlOpDispatch<eavlSourceTopologySparseMapOp_CPU<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, indices, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlSourceTopologySparseMapOp_CPU<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, indices, functor);
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

            eavlOpDispatch<eavlSourceTopologySparseMapOp_GPU<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, indices, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlSourceTopologySparseMapOp_GPU<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, indices, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class IS, class O, class INDEX, class F>
eavlSourceTopologySparseMapOp<IS,O,INDEX,F> *new_eavlSourceTopologySparseMapOp(eavlCellSet *c, eavlTopology t,
                                                                   IS is, O o, INDEX indices, F f) 
{
    return new eavlSourceTopologySparseMapOp<IS,O,INDEX,F>(c,t,is,o,indices,f);
}


#endif
