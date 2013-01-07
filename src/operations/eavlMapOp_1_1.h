// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_MAP_OP_1_1_H
#define EAVL_MAP_OP_1_1_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch_1_1.h"
#include "eavlException.h"

#ifndef DOXYGEN

template <class F,
          class I0, class O0>
struct cpuMapOp_1_1_function
{
    static void call(int n, int &dummy,
                     I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        for (int i=0; i<n; i++)
        {
            int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
            int index_o0 = i * o0mul + o0add;
            o0[index_o0] = functor(i0[index_i0]);
        }
    }
};

#if defined __CUDACC__

template <class F, class I0, class O0>
__global__ void
mapKernel_1_1(int n, int dummy,
              I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
              O0 *o0, int o0mul, int o0add,
              F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < n; index += numThreads)
    {
        int index_i0 = ((index / i0div) % i0mod) * i0mul + i0add;
        int index_o0 = index * o0mul + o0add;
        o0[index_o0] = functor(i0[index_i0]);
    }
}

template <class F, class I0, class O0>
struct gpuMapOp_1_1_function
{
    static void call(int n, int &dummy,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        mapKernel_1_1<<< blocks, threads >>>(n, dummy,
                                            d_i0, i0div, i0mod, i0mul, i0add,
                                            d_o0, o0mul, o0add,
                                            functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlMapOp_1_1
//
// Purpose:
///   A simple operation which takes one input array, applies a functor to
///   it, and places the result in a matching location in the output array.
//
// Programmer:  Jeremy Meredith
// Creation:    February 20, 2012
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlMapOp_1_1 : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    F           functor;
  public:
    eavlMapOp_1_1(eavlArrayWithLinearIndex in0,
                  eavlArrayWithLinearIndex out0,
                  F f)
        : inArray0(in0), outArray0(out0), functor(f)
    {
    }
    virtual void GoCPU()
    {
        int n = inArray0.array->GetNumberOfTuples();
        if (outArray0.array->GetNumberOfTuples() != n)
            THROW(eavlException,"Expecting same-length arrays in eavlMapOp_1_1");

        ///\todo: assert output array(s) aren't logical, i.e. check div,mod

        int dummy;
        eavlDispatch_1_1<cpuMapOp_1_1_function>(n, eavlArray::HOST, dummy,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        int n = inArray0.array->GetNumberOfTuples();
        if (outArray0.array->GetNumberOfTuples() != n)
            THROW(eavlException,"Expecting same-length arrays in eavlMapOp_1_1");

        ///\todo: assert output array(s) aren't logical, i.e. check div,mod
        int dummy;
        eavlDispatch_1_1<gpuMapOp_1_1_function>(n, eavlArray::DEVICE, dummy,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

#endif

