// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SIMPLE_REVERSE_INDEX_OP_H
#define EAVL_SIMPLE_REVERSE_INDEX_OP_H


#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlException.h"

/// like reverse-index op, but assume the output counts
/// can only ever be "1", so we treat the output-count array 
/// like a simple boolean flag, and we don't need to generate
/// a reverse subindex array.
static void eavlSimpleReverseIndexOp_CPU(int nInputVals,
                            int *inOF, int inOFdiv, int inOFmod, int inOFmul, int inOFadd,
                            int *inOI, int inOIdiv, int inOImod, int inOImul, int inOIadd,
                            int *outII, int outIImul, int outIIadd)
{
#pragma omp parallel for
    for (int i=0; i<nInputVals; i++)
    {
        int outflag  = inOF[((i/inOFdiv)%inOFmod)*inOFmul+inOFadd];
        int outindex = inOI[((i/inOIdiv)%inOImod)*inOImul+inOIadd];
        if (outflag)
            outII[outindex*outIImul+outIIadd] = i;
    }
}


#if defined __CUDACC__

__global__ static void eavlSimpleReverseIndexOp_kernel(int nInputVals,
                            int *inOF, int inOFdiv, int inOFmod, int inOFmul, int inOFadd,
                            int *inOI, int inOIdiv, int inOImod, int inOImul, int inOIadd,
                            int *outII, int outIImul, int outIIadd)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = threadID; index < nInputVals; index += numThreads)
    {
        int outflag  = inOF[((index/inOFdiv)%inOFmod)*inOFmul+inOFadd];
        int outindex = inOI[((index/inOIdiv)%inOImod)*inOImul+inOIadd];
        if (outflag)
            outII[outindex*outIImul+outIIadd] = index;
    }
}

static void eavlSimpleReverseIndexOp_GPU(int nInputVals,
                            int *d_inOF, int inOFdiv, int inOFmod, int inOFmul, int inOFadd,
                            int *d_inOI, int inOIdiv, int inOImod, int inOImul, int inOIadd,
                            int *d_outII, int outIImul, int outIIadd)
{
    int numBlocks  = 32;
    int numThreads = 256;
    eavlSimpleReverseIndexOp_kernel<<<numBlocks, numThreads>>>
        (nInputVals,
         d_inOF, inOFdiv, inOFmod, inOFmul, inOFadd,
         d_inOI, inOIdiv, inOImod, inOImul, inOIadd,
         d_outII, outIImul, outIIadd);
    CUDA_CHECK_ERROR();
}
#endif


// ****************************************************************************
// Class:  eavlSimpleReverseIndexOp
//
// Purpose:
///   Given an input array of booleans, and an input array of output starting
///   indices (usually created by the caller using an exclusive scan of the
///   first array), generate an output array containing a map back to the
///   input index.
///
///   For example, if inOutputFlag is [0 1 1 0 1 0],
///   and inOutputIndex is thus [0 0 1 2 2 3], then
///   the result in outInputIndex will be [1 2 4] (i.e. the list of
///   indices from the input array which were set to 1).
//
// Programmer:  Jeremy Meredith
// Creation:    March 3, 2012
//
// Modifications:
// ****************************************************************************
class eavlSimpleReverseIndexOp : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inOutputFlag;
    eavlArrayWithLinearIndex inOutputIndex;
    eavlArrayWithLinearIndex outInputIndex;
  public:
    eavlSimpleReverseIndexOp(eavlArrayWithLinearIndex inOutputFlag_,
                             eavlArrayWithLinearIndex inOutputIndex_,
                             eavlArrayWithLinearIndex outInputIndex_)
        : inOutputFlag(inOutputFlag_),
          inOutputIndex(inOutputIndex_),
          outInputIndex(outInputIndex_)
    {
    }

    virtual void GoCPU()
    {
        int n = inOutputFlag.array->GetNumberOfTuples();

        eavlIntArray *inOF = dynamic_cast<eavlIntArray*>(inOutputFlag.array);
        eavlIntArray *inOI = dynamic_cast<eavlIntArray*>(inOutputIndex.array);
        eavlIntArray *outII = dynamic_cast<eavlIntArray*>(outInputIndex.array);
        if (!inOF || !inOI || !outII)
            THROW(eavlException,"eavlSimpleReverseIndexOp expects all integer arrays.");

        eavlSimpleReverseIndexOp_CPU(n,
                               (int*)inOF->GetHostArray(), inOutputFlag.div, inOutputFlag.mod, inOutputFlag.mul, inOutputFlag.add,
                               (int*)inOI->GetHostArray(), inOutputIndex.div, inOutputIndex.mod, inOutputIndex.mul, inOutputIndex.add,
                               (int*)outII->GetHostArray(), outInputIndex.mul, outInputIndex.add);
    }

    virtual void GoGPU()
    {
#if defined __CUDACC__
        int n = inOutputFlag.array->GetNumberOfTuples();

        eavlIntArray *inOF = dynamic_cast<eavlIntArray*>(inOutputFlag.array);
        eavlIntArray *inOI = dynamic_cast<eavlIntArray*>(inOutputIndex.array);
        eavlIntArray *outII = dynamic_cast<eavlIntArray*>(outInputIndex.array);
        if (!inOF || !inOI || !outII)
            THROW(eavlException,"eavlSimpleReverseIndexOp expects all integer arrays.");

        eavlSimpleReverseIndexOp_GPU(n,
                               (int*)inOF->GetCUDAArray(), inOutputFlag.div, inOutputFlag.mod, inOutputFlag.mul, inOutputFlag.add,
                               (int*)inOI->GetCUDAArray(), inOutputIndex.div, inOutputIndex.mod, inOutputIndex.mul, inOutputIndex.add,
                               (int*)outII->GetCUDAArray(), outInputIndex.mul, outInputIndex.add);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

#endif
