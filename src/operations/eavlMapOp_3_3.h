// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_MAP_OP_3_3_H
#define EAVL_MAP_OP_3_3_H

#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOperation.h"
#include "eavlOpDispatch_3_3.h"
#include "eavlTopology.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

#ifndef DOXYGEN
template <class F, class I01, class I2, class O01, class O2>
struct cpu_Map_3_3
{
    static void call(int nitems, int &dummy,
                     I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     F &functor)
    {
        for (int index = 0; index < nitems; index++)
        {
            float out0, out1, out2;

            functor(i0[((index / i0div) % i0mod) * i0mul + i0add],
                    i1[((index / i1div) % i1mod) * i1mul + i1add],
                    i2[((index / i2div) % i2mod) * i2mul + i2add],
                    out0, out1, out2);

            o0[index * o0mul + o0add] = out0;
            o1[index * o1mul + o1add] = out1;
            o2[index * o2mul + o2add] = out2;
        }
    }
};

#if defined __CUDACC__

template <class F, class I01, class I2, class O01, class O2>
__global__ void
mapKernel_3_3(int nitems, int &dummy,
                            I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                            I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                            I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                            O01 *o0, int o0mul, int o0add,
                            O01 *o1, int o1mul, int o1add,
                            O2  *o2, int o2mul, int o2add,
                            F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < nitems; index += numThreads)
    {
        float out0, out1, out2;

        functor(i0[((index / i0div) % i0mod) * i0mul + i0add],
                i1[((index / i1div) % i1mod) * i1mul + i1add],
                i2[((index / i2div) % i2mod) * i2mul + i2add],
                out0, out1, out2);

        o0[index * o0mul + o0add] = out0;
        o1[index * o1mul + o1add] = out1;
        o2[index * o2mul + o2add] = out2;
    }
}

template <class F, class I01, class I2, class O01, class O2>
struct gpuMapOp_3_3
{
    static void call(int nitems,
                     I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        mapKernel_3_3<<< blocks, threads >>>(nitems,
                                                           i0, i0div, i0mod, i0mul, i0add,
                                                           i1, i1div, i1mod, i1mul, i1add,
                                                           i2, i2div, i2mod, i2mul, i2add,
                                                           o0, o0mul, o0add,
                                                           o1, o1mul, o1add,
                                                           o2, o2mul, o2add,
                                                           functor);
        CUDA_CHECK_ERROR();
    }
};


template <class F>
void callMapKernel_3_3(int nitems,
                          eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                          eavlArray *i1, int i1div, int i1mod, int i1mul, int i1add,
                          eavlArray *i2, int i2div, int i2mod, int i2mul, int i2add,
                          eavlArray *o0, int o0mul, int o0add,
                          eavlArray *o1, int o1mul, int o1add,
                          eavlArray *o2, int o2mul, int o2add,
                          F &functor)
{
    ///\todo: assert num for all output arrays is the same?

    ///\todo: just doing this to make timing easier; can probably remove it
    i0->GetCUDAArray();
    i1->GetCUDAArray();
    i2->GetCUDAArray();
    o0->GetCUDAArray();
    o1->GetCUDAArray();
    o2->GetCUDAArray();

    // run the kernel
    eavlDispatch_3_3<gpuMapOp_3_3>(nitems,
                                                  eavlArray::DEVICE,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  i1, i1div, i1mod, i1mul, i1add,
                                                  i2, i2div, i2mod, i2mul, i2add,
                                                  o0, o0mul, o0add,
                                                  o1, o1mul, o1add,
                                                  o2, o2mul, o2add,
                                                  functor);
}


#endif
#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlMapOp_3_3<F>
//
// Purpose:
///   Map from one 3 input arrays to 3 output arrays.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    September 6, 2011
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlMapOp_3_3 : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inArray0, inArray1, inArray2;
    eavlArrayWithLinearIndex outArray0, outArray1, outArray2;
    F                functor;
  public:
    eavlMapOp_3_3(eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex in1,
                         eavlArrayWithLinearIndex in2,
                         eavlArrayWithLinearIndex out0,
                         eavlArrayWithLinearIndex out1,
                         eavlArrayWithLinearIndex out2,
                         F f)
        : inArray0(in0),
          inArray1(in1),
          inArray2(in2),
          outArray0(out0),
          outArray1(out1),
          outArray2(out2),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
            eavlDispatch_3_3<cpu_Map_3_3>(outArray0.array->GetNumberOfTuples(),
                                                            eavlArray::HOST, dummy,
                                                            inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                                            inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                                            inArray2.array, inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                                                            outArray0.array, outArray0.mul, outArray0.add,
                                                            outArray1.array, outArray1.mul, outArray1.add,
                                                            outArray2.array, outArray2.mul, outArray2.add,
                                                            functor);
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        int dummy;
            callMapKernel_3_3(outArray0.array->GetNumberOfTuples(),
                              eavlArray::DEVICE, dummy,
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                   inArray2.array, inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   outArray1.array, outArray1.mul, outArray1.add,
                                   outArray2.array, outArray2.mul, outArray2.add,
                                   functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
}; 

#endif
