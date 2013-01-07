// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_GATHER_OP_1_H
#define EAVL_GATHER_OP_1_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch_io1_int.h"
#include "eavlException.h"

#ifndef DOXYGEN

template <class F,
          class IO0>
struct cpuGatherOp_1_function
{
    static void call(int nout, int &dummy,
                     IO0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *o0, int o0mul, int o0add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        for (int i=0; i<nout; i++)
        {
            int index_idx = i*idxmul + idxadd;
            int in_index = idx[index_idx];

            int index_i0 = ((in_index / i0div) % i0mod) * i0mul + i0add;
            IO0 val = i0[index_i0];
            
            int index_o0 = i*o0mul + o0add;
            o0[index_o0] = val;
        }
    }
};

#if defined __CUDACC__

template <class F, class IO0>
__global__ void
gatherKernel_1(int n, int dummy,
               IO0 *i0, int i0div, int i0mod, int i0mul, int i0add,
               IO0 *o0, int o0mul, int o0add,
               int *idx, int idxmul, int idxadd,
               F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int index_i0 = ((in_index / i0div) % i0mod) * i0mul + i0add;
        IO0 val = i0[index_i0];
            
        int index_o0 = i*o0mul + o0add;
        o0[index_o0] = val;
    }
}

template <class F, class IO0>
struct gpuGatherOp_1_function
{
    static void call(int n, int &dummy,
                     IO0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *d_o0, int o0mul, int o0add,
                     int *d_idx, int idxmul, int idxadd,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        gatherKernel_1<<< blocks, threads >>>(n, dummy,
                                              d_i0, i0div, i0mod, i0mul, i0add,
                                              d_o0, o0mul, o0add,
                                              d_idx, idxmul, idxadd,
                                              functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN


// ****************************************************************************
// Class:  eavlGatherOp_1
//
// Purpose:
///   A simple gather operation on a single input and output array; copies
///   the values specified by the indices array from the source array to
///   the destination array.
//
// Programmer:  Jeremy Meredith
// Creation:    April 13, 2012
//
// Modifications:
// ****************************************************************************
class eavlGatherOp_1 : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    eavlArrayWithLinearIndex indicesArray;
    DummyFunctor functor;
  public:
    eavlGatherOp_1(eavlArrayWithLinearIndex in0,
                   eavlArrayWithLinearIndex out0,
                   eavlArrayWithLinearIndex indices)
        : inArray0(in0), outArray0(out0), indicesArray(indices)
    {
    }
    virtual void GoCPU()
    {
        int n = outArray0.array->GetNumberOfTuples();

        int dummy;
        eavlDispatch_io1_int<cpuGatherOp_1_function>
            (n, eavlArray::HOST, dummy,
             inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
             outArray0.array, outArray0.mul, outArray0.add,
             indicesArray.array, indicesArray.mul, indicesArray.add,
             functor);
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        int n = outArray0.array->GetNumberOfTuples();

        int dummy;
        eavlDispatch_io1_int<gpuGatherOp_1_function>
            (n, eavlArray::DEVICE, dummy,
             inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
             outArray0.array, outArray0.mul, outArray0.add,
             indicesArray.array, indicesArray.mul, indicesArray.add,
             functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};



#endif
