// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_REVERSE_INDEX_OP_H
#define EAVL_REVERSE_INDEX_OP_H


#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlException.h"

static void eavlReverseIndexOp_CPU(int nInputVals,
                            int *inOC, int inOCdiv, int inOCmod, int inOCmul, int inOCadd,
                            int *inOI, int inOIdiv, int inOImod, int inOImul, int inOIadd,
                            int *outII, int outIImul, int outIIadd,
                            int *outIS, int outISmul, int outISadd,
                            int maxPerInput)
{
#pragma omp parallel for
    for (int i=0; i<nInputVals; i++)
    {
        int outcount = inOC[((i/inOCdiv)%inOCmod)*inOCmul+inOCadd];
        int outindex = inOI[((i/inOIdiv)%inOImod)*inOImul+inOIadd];
        for (int j=0; j<outcount; j++)
        {
            outII[(outindex+j)*outIImul+outIIadd] = i;
            outIS[(outindex+j)*outISmul+outISadd] = j;
        }
    }
}


#if defined __CUDACC__

__global__ static void eavlReverseIndexOp_kernel(int nInputVals,
                            int *inOC, int inOCdiv, int inOCmod, int inOCmul, int inOCadd,
                            int *inOI, int inOIdiv, int inOImod, int inOImul, int inOIadd,
                            int *outII, int outIImul, int outIIadd,
                            int *outIS, int outISmul, int outISadd,
                            int maxPerInput)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int index = threadID; index < nInputVals; index += numThreads)
    {
        int outcount = inOC[((index/inOCdiv)%inOCmod)*inOCmul+inOCadd];
        int outindex = inOI[((index/inOIdiv)%inOImod)*inOImul+inOIadd];
        for (int j=0; j<outcount; j++)
        {
            outII[(outindex+j)*outIImul+outIIadd] = index;
            outIS[(outindex+j)*outISmul+outISadd] = j;
        }
    }
}

static void eavlReverseIndexOp_GPU(int nInputVals,
                            int *d_inOC, int inOCdiv, int inOCmod, int inOCmul, int inOCadd,
                            int *d_inOI, int inOIdiv, int inOImod, int inOImul, int inOIadd,
                            int *d_outII, int outIImul, int outIIadd,
                            int *d_outIS, int outISmul, int outISadd,
                            int maxPerInput)
{
    int numBlocks  = 32;
    int numThreads = 256;
    eavlReverseIndexOp_kernel<<<numBlocks, numThreads>>>
        (nInputVals,
         d_inOC, inOCdiv, inOCmod, inOCmul, inOCadd,
         d_inOI, inOIdiv, inOImod, inOImul, inOIadd,
         d_outII, outIImul, outIIadd,
         d_outIS, outISmul, outISadd,
        maxPerInput);
    CUDA_CHECK_ERROR();
}

#endif

// ****************************************************************************
// Class:  eavlSimpleReverseIndexOp
//
// Purpose:
///   Given an input array of output counts, and an input array of output
///   starting indices (usually created by the caller using an exclusive scan
///   of the first array), generate an output array containing a map back to
///   the input index, and item within that input index, where the inputs
///   should be written. 
///
///   For example, if inOutputCount is [0 1 2 0 1 0 0 0 3 1]
///   and inOutputIndex is thus        [0 0 1 3 3 4 4 4 4 7]
///   the result in outInputIndex will be  [1 2 2 4 8 8 8 9]
///   and  outInputSubIndex will be        [0 0 1 0 0 1 2 0].
//
// Programmer:  Jeremy Meredith
// Creation:    March 4, 2012
//
// Modifications:
// ****************************************************************************
class eavlReverseIndexOp : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inOutputCounts;
    eavlArrayWithLinearIndex inOutputIndex;
    eavlArrayWithLinearIndex outInputIndex;
    eavlArrayWithLinearIndex outInputSubindex;
    int maxPerInput;
  public:
    eavlReverseIndexOp(eavlArrayWithLinearIndex inOutputCounts_,
                       eavlArrayWithLinearIndex inOutputIndex_,
                       eavlArrayWithLinearIndex outInputIndex_,
                       eavlArrayWithLinearIndex outInputSubindex_,
                       int maxOutValsPerInputCell)
        : inOutputCounts(inOutputCounts_),
          inOutputIndex(inOutputIndex_),
          outInputIndex(outInputIndex_),
          outInputSubindex(outInputSubindex_),
          maxPerInput(maxOutValsPerInputCell)
    {
    }

    virtual void GoCPU()
    {
        int n = inOutputCounts.array->GetNumberOfTuples();

        eavlIntArray *inOC = dynamic_cast<eavlIntArray*>(inOutputCounts.array);
        eavlIntArray *inOI = dynamic_cast<eavlIntArray*>(inOutputIndex.array);
        eavlIntArray *outII = dynamic_cast<eavlIntArray*>(outInputIndex.array);
        eavlIntArray *outIS = dynamic_cast<eavlIntArray*>(outInputSubindex.array);
        if (!inOC || !inOI || !outII || !outIS)
            THROW(eavlException,"eavlReverseIndexOp expects all integer arrays.");

        eavlReverseIndexOp_CPU(n,
                               (int*)inOC->GetHostArray(), inOutputCounts.div, inOutputCounts.mod, inOutputCounts.mul, inOutputCounts.add,
                               (int*)inOI->GetHostArray(), inOutputIndex.div, inOutputIndex.mod, inOutputIndex.mul, inOutputIndex.add,
                               (int*)outII->GetHostArray(), outInputIndex.mul, outInputIndex.add,
                               (int*)outIS->GetHostArray(), outInputSubindex.mul, outInputSubindex.add,
                               maxPerInput);
    }

    virtual void GoGPU()
    {
#if defined __CUDACC__
        int n = inOutputCounts.array->GetNumberOfTuples();

        eavlIntArray *inOC = dynamic_cast<eavlIntArray*>(inOutputCounts.array);
        eavlIntArray *inOI = dynamic_cast<eavlIntArray*>(inOutputIndex.array);
        eavlIntArray *outII = dynamic_cast<eavlIntArray*>(outInputIndex.array);
        eavlIntArray *outIS = dynamic_cast<eavlIntArray*>(outInputSubindex.array);
        if (!inOC || !inOI || !outII || !outIS)
            THROW(eavlException,"eavlReverseIndexOp expects all integer arrays.");

        eavlReverseIndexOp_GPU(n,
                               (int*)inOC->GetCUDAArray(), inOutputCounts.div, inOutputCounts.mod, inOutputCounts.mul, inOutputCounts.add,
                               (int*)inOI->GetCUDAArray(), inOutputIndex.div, inOutputIndex.mod, inOutputIndex.mul, inOutputIndex.add,
                               (int*)outII->GetCUDAArray(), outInputIndex.mul, outInputIndex.add,
                               (int*)outIS->GetCUDAArray(), outInputSubindex.mul, outInputSubindex.add,
                               maxPerInput);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

#endif
