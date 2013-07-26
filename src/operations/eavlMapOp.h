// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_MAP_OP_H
#define EAVL_MAP_OP_H

#include "eavlCUDA.h"
#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlException.h"

#ifndef DOXYGEN

struct eavlMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int, const IN inputs, OUT outputs, F &functor)
    {
        for (int index = 0; index < nitems; ++index)
        {
            typename collecttype<IN>::const_type in(collect(index, inputs));
            typename collecttype<OUT>::type out(collect(index, outputs));
            out = functor(in);
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
        // if we want to be explicit:
        //typename collecttype<IN>::type in(collect(index, inputs));
        //typename collecttype<OUT>::type out(collect(index, outputs));
        //out = functor( cons<float,cons<float,nulltype> >(in) );

        // or more simply:
        collect(index, outputs) = functor(collect(index, inputs));
    }
}

struct eavlMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }

    template <class F, class IN, class OUT>
    static void call(int nitems, int, const IN inputs, OUT outputs, F &functor)
    {
        int numThreads = 64;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
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
  public:
    eavlMapOp(I i, O o, F f) : inputs(i), outputs(o), functor(f)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
        int n = outputs.first.array->GetNumberOfTuples();
        eavlOpDispatch<eavlMapOp_CPU>(n, dummy, inputs, outputs, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        int dummy;
        int n = outputs.first.array->GetNumberOfTuples();
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


#endif
