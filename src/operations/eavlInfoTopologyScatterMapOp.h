// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_INFO_TOPOLOGY_SCATTER_MAP_OP_H
#define EAVL_INFO_TOPOLOGY_SCATTER_MAP_OP_H

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
struct eavlInfoTopologyScatterMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int *sparseindices = get<0>(indices).array;

#pragma omp parallel for
        for (int denseindex = 0; denseindex < nitems; ++denseindex)
        {
            int sparseindex = sparseindices[get<0>(indices).indexer.index(denseindex)];
            int shapeType = conn.GetShapeType(sparseindex);
            collect(sparseindex, outputs) = functor(shapeType, collect(denseindex, inputs));
        }
    }
};

#if defined __CUDACC__

template <class CONN, class F, class IN, class OUT, class INDEX>
__global__ void
eavlInfoTopologyScatterMapOp_kernel(int nitems, CONN conn,
                                   const IN inputs, OUT outputs,
                                   INDEX indices, F functor)
{
    int *sparseindices = get<0>(indices).array;

    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int denseindex = threadID; denseindex < nitems; denseindex += numThreads)
    {
        int sparseindex = sparseindices[get<0>(indices).indexer.index(denseindex)];
        int shapeType = conn.GetShapeType(sparseindex);
        collect(sparseindex, outputs) = functor(shapeType, collect(denseindex, inputs));
    }
}


template <class CONN>
struct eavlInfoTopologyScatterMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlInfoTopologyScatterMapOp_kernel<<< blocks, threads >>>(nitems, conn,
                                                                  inputs, outputs,
                                                                  indices, functor);
        CUDA_CHECK_ERROR();
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlInfoTopologyScatterMapOp
//
// Purpose:
///   Map from one element in a mesh to the same element, with
///   topological information passed along to the functor.
///   In this scatter version of the operation, the inputs on the destination
///   topology are densely indexed (0 to n-1), and the outputs are
///   sparsely indexed by the index array.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************
template <class I, class O, class INDEX, class F>
class eavlInfoTopologyScatterMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    I            inputs;
    O            outputs;
    INDEX        indices;
    F            functor;
  public:
    eavlInfoTopologyScatterMapOp(eavlCellSet *c, eavlTopology t,
                            I i, O o, INDEX ind, F f)
        : cells(c), topology(t), inputs(i), outputs(o), indices(ind), functor(f)
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
            eavlOpDispatch<eavlInfoTopologyScatterMapOp_CPU<eavlExplicitConnectivity> >(n, conn, inputs, outputs, indices, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlInfoTopologyScatterMapOp_CPU<eavlRegularConnectivity> >(n, conn, inputs, outputs, indices, functor);
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

            eavlOpDispatch<eavlInfoTopologyScatterMapOp_GPU<eavlExplicitConnectivity> >(n, conn, inputs, outputs, indices, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlInfoTopologyScatterMapOp_GPU<eavlRegularConnectivity> >(n, conn, inputs, outputs, indices, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class INDEX, class F>
eavlInfoTopologyScatterMapOp<I,O,INDEX,F> *new_eavlInfoTopologyScatterMapOp(eavlCellSet *c, eavlTopology t,
                                                                   I i, O o, INDEX indices, F f) 
{
    return new eavlInfoTopologyScatterMapOp<I,O,INDEX,F>(c,t,i,o,indices,f);
}


#endif
