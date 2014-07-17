// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_GATHER_OP_H
#define EAVL_GATHER_OP_H

#include "eavlCUDA.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlOperation.h"
#include "eavlException.h"
#include <time.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

struct eavlGatherOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, int,
                     const IN inputs, OUT outputs,
                     INDEX indices, F&)
    {
        int *sparseindices = get<0>(indices).array;

#pragma omp parallel for
        for (int denseindex = 0; denseindex < nitems; ++denseindex)
        {
            int sparseindex = sparseindices[get<0>(indices).indexer.index(denseindex)];
            // can't use operator= because it's ambiguous when only
            // one input and one output array (without a functor that
            // would force a cast to a known type situation).
            collect(denseindex, outputs).CopyFrom(collect(sparseindex, inputs));
        }
    }
};

#if defined __CUDACC__

template <class IN, class OUT, class INDEX>
__global__ void
eavlGatherOp_kernel(int nitems,
                    const IN inputs, OUT outputs,
                    INDEX indices)
{
    int *sparseindices = get<0>(indices).array;

    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int denseindex = threadID; denseindex < nitems; denseindex += numThreads)
    {
        int sparseindex = sparseindices[get<0>(indices).indexer.index(denseindex)];
        // can't use operator= because it's ambiguous when only
        // one input and one output array (without a functor that
        // would force a cast to a known type situation).
        collect(denseindex, outputs).CopyFrom(collect(sparseindex, inputs));
    }
}


struct eavlGatherOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, int,
                     const IN inputs, OUT outputs,
                     INDEX indices, F&)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        eavlGatherOp_kernel<<< blocks, threads >>>(nitems,
                                                   inputs, outputs,
                                                   indices);
        CUDA_CHECK_ERROR();
    }
};


#endif

#endif

#include "eavlExplicitConnectivity.h"
// ****************************************************************************
// Class:  eavlGatherOp
//
// Purpose:
///   A simple gather operation on a single input and output array; copies
///   the values specified by the indices array from the source array to
///   the destination array.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications: Matt Larsen 7/10/14 Added ability to only process a subset of 
//                the output length. This allows the output array length to be 
//                larger than the index array length.
// ****************************************************************************
template <class I, class O, class INDEX>
class eavlGatherOp : public eavlOperation
{
  protected:
    DummyFunctor functor;
    I            inputs;
    O            outputs;
    INDEX        indices;
    int          nitems;
  public:
    eavlGatherOp(I i, O o, INDEX ind)
        : inputs(i), outputs(o), indices(ind), nitems(-1)
    {
    }
    eavlGatherOp(I i, O o, INDEX ind, int itemsToProcess)
        : inputs(i), outputs(o), indices(ind), nitems(itemsToProcess)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
        int n=0;
        if( nitems > 0 ) n = nitems;
        else n = outputs.first.length();
        eavlOpDispatch<eavlGatherOp_CPU>(n, dummy, inputs, outputs, indices, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        int dummy;
        int n=0;
        if( nitems > 0 ) n = nitems;
        else n = outputs.first.length();
        eavlOpDispatch<eavlGatherOp_GPU>(n, dummy, inputs, outputs, indices, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class INDEX>
eavlGatherOp<I,O,INDEX> *new_eavlGatherOp(I i, O o, INDEX indices) 
{
    return new eavlGatherOp<I,O,INDEX>(i,o,indices);
}

template <class I, class O, class INDEX>
eavlGatherOp<I,O,INDEX> *new_eavlGatherOp(I i, O o, INDEX indices, int itemsToProcess) 
{
    return new eavlGatherOp<I,O,INDEX>(i,o,indices, itemsToProcess);
}


#endif
