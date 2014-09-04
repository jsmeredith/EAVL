// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_N_TO_1_GATHER_OP_H
#define EAVL_N_TO_1_GATHER_OP_H

#include "eavlCUDA.h"
#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlException.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN



struct eavlNto1GatherOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int n, const IN inputs, OUT outputs, F &functor)
    {
#pragma omp parallel for
        for (int index = 0; index < nitems; ++index)
        {
            typename collecttype<IN>::const_type in(collect(index, inputs));
            float acc = 0;
            tuple<float> p;
            for(int i = 0; i < n; i++){
               
                p=collect(index * n + i, inputs);
                acc += get<0>(p);
            }
            collect(index, outputs) = tuple<float>(acc/(float)n);
        }
    }
};


#if defined __CUDACC__

template < class F, class IN, class OUT>
__global__ void
nTo1GatherKernel(int nitems,int n, const IN inputs, OUT outputs, F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < nitems; index += numThreads)//why??
    {   
        typename collecttype<IN>::const_type in(collect(index, inputs));
        float acc = 0;
        tuple<float> p;
        for(int i = 0; i < n; i++){
            p = collect(index * n + i, inputs);
            acc += get<0>(p);
        }
        collect(index, outputs) =tuple<float>(acc/(float)n);        
    }
}

struct eavlNto1GatherOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }

    template <class F, class IN, class OUT>
    static void call(int nitems, int mutliplyer, const IN inputs, OUT outputs, F &functor)
    {
        int numThreads = 128;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (128,           1, 1);
        nTo1GatherKernel<<< blocks, threads >>>(nitems,mutliplyer, inputs, outputs,functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN
// ****************************************************************************
// Class:  eavlNto1GatherOp
//
// Purpose:
///   Used in conjuction with 1 to N scatter for ray tracing. This operation
///   gathers n contiguous values from the input array and averages them to
///   a single floating point value. Input should be n times larger than the 
///   output.
//
// ****************************************************************************
template <class I, class O>
class eavlNto1GatherOp : public eavlOperation
{
  protected:
    I  inputs;
    O  outputs;
    DummyFunctor  dummy;
    int mutliplyer;
  public:
    eavlNto1GatherOp(I i, O o, int m) : inputs(i), outputs(o), mutliplyer(m)
    {
    }
    virtual void GoCPU()
    {
        
        int n = outputs.first.length(); //if the inputs dim is n x m then output length is (n-1)x(m-1)
        eavlOpDispatch<eavlNto1GatherOp_CPU>(n, mutliplyer, inputs, outputs, dummy);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        
        int n = outputs.first.length();
        eavlOpDispatch<eavlNto1GatherOp_GPU>(n, mutliplyer, inputs, outputs, dummy);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O>
eavlNto1GatherOp<I,O> *new_eavlNto1GatherOp(I i, O o, int width) 
{
    return new eavlNto1GatherOp<I,O>(i,o,width);
}


#endif
