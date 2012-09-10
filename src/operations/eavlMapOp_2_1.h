// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_MAP_OP_2_1_H
#define EAVL_MAP_OP_2_1_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch_2_1.h"
#include "eavlException.h"

#ifndef DOXYGEN

template <class F,
          class I0, class I1, class O0>
struct cpuMapOp_2_1_function
{
    static void call(int n, int &dummy,
                     I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     O0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        for (int i=0; i<n; i++)
        {
            int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
            int index_i1 = ((i / i1div) % i1mod) * i1mul + i1add;
            int index_o0 = i * o0mul + o0add;
            o0[index_o0] = functor(i0[index_i0], i1[index_i1]);
        }
    }
};

#if defined __CUDACC__

template <class F, class I0, class I1, class O0>
__global__ void
mapKernel_2_1(int n, int dummy,
              I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
              I1 *i1, int i1div, int i1mod, int i1mul, int i1add,
              O0 *o0, int o0mul, int o0add,
              F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < n; index += numThreads)
    {
        int index_i0 = ((index / i0div) % i0mod) * i0mul + i0add;
        int index_i1 = ((index / i1div) % i1mod) * i1mul + i1add;
        int index_o0 = index * o0mul + o0add;
        o0[index_o0] = functor(i0[index_i0], i1[index_i1]);
    }
}

template <class F, class I0, class I1, class O0>
struct gpuMapOp_2_1_function
{
    static void call(int n, int &dummy,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1 *d_i1, int i1div, int i1mod, int i1mul, int i1add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        mapKernel_2_1<<< blocks, threads >>>(n, dummy,
                                            d_i0, i0div, i0mod, i0mul, i0add,
                                            d_i1, i1div, i1mod, i1mul, i1add,
                                            d_o0, o0mul, o0add,
                                            functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlMapOp_2_1
//
// Purpose:
///   A simple operation which takes one input array, applies a functor to
///   it, and places the result in a matching location in the output array.
//
// Programmer:  Jeremy Meredith
// Creation:    September 10, 2012
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlMapOp_2_1 : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex inArray1;
    eavlArrayWithLinearIndex outArray0;
    F           functor;
  public:
    eavlMapOp_2_1(eavlArrayWithLinearIndex in0,
                  eavlArrayWithLinearIndex in1,
                  eavlArrayWithLinearIndex out0,
                  F f)
        : inArray0(in0), inArray1(in1), outArray0(out0), functor(f)
    {
    }
    virtual void GoCPU()
    {
        int n = inArray0.array->GetNumberOfTuples();
        if (inArray1.array->GetNumberOfTuples() != n)
            THROW(eavlException,"Expecting same-length arrays in eavlMapOp_2_1");
        if (outArray0.array->GetNumberOfTuples() != n)
            THROW(eavlException,"Expecting same-length arrays in eavlMapOp_2_1");

        ///\todo: assert output array(s) aren't logical, i.e. check div,mod

        int dummy;
        eavlDispatch_2_1<cpuMapOp_2_1_function>(n, eavlArray::HOST, dummy,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        int n = inArray0.array->GetNumberOfTuples();
        if (inArray1.array->GetNumberOfTuples() != n)
            THROW(eavlException,"Expecting same-length arrays in eavlMapOp_2_1");
        if (outArray0.array->GetNumberOfTuples() != n)
            THROW(eavlException,"Expecting same-length arrays in eavlMapOp_2_1");

        ///\todo: assert output array(s) aren't logical, i.e. check div,mod
        int dummy;
        eavlDispatch_2_1<gpuMapOp_2_1_function>(n, eavlArray::DEVICE, dummy,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

#endif

