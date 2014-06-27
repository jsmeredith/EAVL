// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_3X3_NODE_STENCIL_OP_1_1_H
#define EAVL_3X3_NODE_STENCIL_OP_1_1_H

#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOperation.h"
#include "eavlOpDispatch_1_1.h"
#include "eavlException.h"
#include <time.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#if defined __CUDACC__

template <class F, class I0, class O0>
__global__ void
nodeStencilKernel_1_1(int npoints,
                           eavlRegularStructure reg,
                           I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                           O0 *o0, int o0mul, int o0add,
                           F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j;

    for (int index = threadID; index < npoints; index += numThreads)
    {
        reg.CalculateLogicalNodeIndices2D(index, i, j);

        float in0[9]; //3x3, which at least used to be 9
        
        //boundaries?
        for(int x = i-1, nextPoint = 0; x <= i+1; x++)
        {
            for(int y = j-1; y <= j+1; y++)
            {
                int localindex = reg.CalculateNodeIndex2D(x, y);
                in0[nextPoint++] = i0[((localindex / i0div) % i0mod) *i0mul + i0add];
            }
        }

        o0[index * o0mul + o0add] = functor(in0);
    }
}

template <class F, class I0, class O0>
struct gpuNodeStencilOp_1_1
{
    static void call(int npoints,
                     eavlRegularStructure reg,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        nodeStencilKernel_1_1<<< blocks, threads >>>(npoints,
                                                           reg,
                                                           d_i0, i0div, i0mod, i0mul, i0add,
                                                           d_o0, o0mul, o0add,
                                                           functor);
        CUDA_CHECK_ERROR();
    }
};


template <class F>
void callNodeStencilKernel_1_1(int npoints,
                          eavlRegularStructure reg,
                          eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                          eavlArray *o0, int o0mul, int o0add,
                          F &functor)
{
    i0->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_1_1<gpuNodeStencilOp_1_1>(npoints,
                                                  eavlArray::DEVICE,
                                                  reg,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  o0, o0mul, o0add,
                                                  functor);
}
#endif

// ****************************************************************************
// Class:  eavl3X3NodeStencilOp_1_1
//
// Purpose:
///   Perform an operation taking as input a 9-cell window around
///   the destination cell on a structured grid.
//
// Programmer:  Rob Sisneros
// Creation:    June 25, 2012
//
// Modifications:
// ****************************************************************************
template <class F>
class eavl3X3NodeStencilOp_1_1 : public eavlOperation
{
  protected:
    eavlField               *field;
    eavlRegularStructure     reg;
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    F                        functor;
  public:
    eavl3X3NodeStencilOp_1_1(eavlField *inField,
                         eavlRegularStructure regStructure,
                         eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex out0,
                         F f)
        : field(inField),
          reg(regStructure),
          inArray0(in0),
          outArray0(out0),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        THROW(eavlException,"unimplemented");
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        callNodeStencilKernel_1_1(field->GetArray()->GetNumberOfTuples(),
                reg,
                inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                outArray0.array, outArray0.mul, outArray0.add,
                functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
}; 

#endif
