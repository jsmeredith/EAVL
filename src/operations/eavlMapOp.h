// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_MAP_OP_H
#define EAVL_MAP_OP_H

#include "eavlCUDA.h"
#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlException.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

struct eavlMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int, const IN inputs, OUT outputs, F &functor)
    {
#pragma omp parallel for schedule(dynamic)
        for (int index = 0; index < nitems; ++index)
        {
            typename collecttype<IN>::const_type in(collect(index, inputs));
            typename collecttype<OUT>::type out(collect(index, outputs));
            out = functor(in);
            //int tid = omp_get_thread_num();
            //cerr<<index<< " "<<endl;
            // or more simply:
            //collect(index, outputs) = functor(collect(index, inputs));
        }
    }
};


#if defined __CUDACC__

template <class F, class IN, class OUT>
__global__ void
mapKernel(int nitems, const IN inputs, OUT outputs, F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < nitems; index += numThreads)
    { 
        //if (threadID<nitems)  
        collect(index, outputs) = functor(collect(index, inputs));
    }
} 

struct eavlMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }

    template <class F, class IN, class OUT>
    static void call(int nitems, int, const IN inputs, OUT outputs, F &functor)
    {
        
        int numThreads = 128;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (240,           1, 1);
        cudaFuncSetCacheConfig(mapKernel<F, IN,OUT>, cudaFuncCachePreferL1);
        mapKernel<<< blocks, threads >>>(nitems, inputs, outputs, functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlMapOp
//
// Purpose:
///   A simple operation which takes one set of input arrays, applies a functor
///   to them, and places the results matching locations in the output arrays.
//
// Programmer:  Jeremy Meredith
// Creation:    July 25, 2013
//
// Modifications:
// ****************************************************************************
template <class I, class O, class F>
class eavlMapOp : public eavlOperation
{
  protected:
    I  inputs;
    O  outputs;
    F  functor;
    int nitems;
  public:
    eavlMapOp(I i, O o, F f) : inputs(i), outputs(o), functor(f), nitems(-1)
    {
    }
    eavlMapOp(I i, O o, F f, int nItems) : inputs(i), outputs(o), functor(f), nitems(nItems)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
        int n;
        if( nitems > 0 ) n = nitems;
        else n = outputs.first.length();
        eavlOpDispatch<eavlMapOp_CPU>(n, dummy, inputs, outputs, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        int dummy;
        int n;
        if( nitems > 0 ) n = nitems;
        else n = outputs.first.length();
        eavlOpDispatch<eavlMapOp_GPU>(n, dummy, inputs, outputs, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class F>
eavlMapOp<I,O,F> *new_eavlMapOp(I i, O o, F f) 
{
    return new eavlMapOp<I,O,F>(i,o,f);
}
template <class I, class O, class F>
eavlMapOp<I,O,F> *new_eavlMapOp(I i, O o, F f, int itemsToProcess) 
{
    return new eavlMapOp<I,O,F>(i,o,f, itemsToProcess);
}


#endif
