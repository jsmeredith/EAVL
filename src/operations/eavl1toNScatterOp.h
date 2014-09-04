// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_1_TO_N_SCATTER_OP_H
#define EAVL_1_TO_N_SCATTER_OP_H

#include "eavlCUDA.h"
#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlException.h"
#include <stdlib.h>
#include <time.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

static int rndm;

struct eavl1toNScatterOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int n, const IN inputs, OUT outputs, F &functor)
    {
        int seed=nitems*n;
        #pragma omp parallel for
        for (int index = 0; index < nitems; ++index)
        {
            typename collecttype<IN>::const_type in(collect(index, inputs));
            for (int i = 0; i < n; i++)
            {
                // Passing random number and index into functor
                collect(index*n+i, outputs) =functor(collect(index, inputs),seed, i);    
            }
        }
    }
};


#if defined __CUDACC__

template < class F, class IN, class OUT>
__global__ void
oneToNScatterKernel(int nitems,int n, const IN inputs, OUT outputs, F functor, int rndm)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < nitems; index += numThreads)
    {   
        for(int i = 0; i < n; i++){
            collect(index * n + i, outputs) =functor(collect(index, inputs),rndm + threadID,i);
        }
    }
}

struct eavl1toNScatterOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }

    template <class F, class IN, class OUT>
    static void call(int nitems, int multiplyer, const IN inputs, OUT outputs, F &functor)
    {
        int numThreads = 128;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (64,           1, 1);
        oneToNScatterKernel<<< blocks, threads >>>(nitems,multiplyer, inputs, outputs, functor,rndm);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN
// ****************************************************************************
// Class:  eavl1toNScatterOp
//
// Purpose:
///   Used in conjuction with N to one gather for ray tracing. This operation
///   scatters n values from each input. To enable unique results for each 
///   functor output, a random value and the index (0-n) is passed into the 
///   functor.
//
// ****************************************************************************
template <class I, class O, class F>
class eavl1toNScatterOp : public eavlOperation
{
  protected:
    I  inputs;
    O  outputs;
    F  functor;
    int multiplyer;

  public:
    eavl1toNScatterOp(I i, O o,F f, int m) : inputs(i), outputs(o),functor(f), multiplyer(m)
    {
        
        rndm=rand();
    }
    virtual void GoCPU()
    {
        int n = inputs.first.length(); 
        rand();
        eavlOpDispatch<eavl1toNScatterOp_CPU>(n, multiplyer, inputs, outputs, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        
        int n = inputs.first.length();
        eavlOpDispatch<eavl1toNScatterOp_GPU>(n, multiplyer, inputs, outputs, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O,class F>
eavl1toNScatterOp<I,O,F> *new_eavl1toNScatterOp(I i, O o,F f, int width) 
{
    return new eavl1toNScatterOp<I,O,F>(i,o,f,width);
}


#endif
