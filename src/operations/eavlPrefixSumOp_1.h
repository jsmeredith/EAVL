// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_PREFIX_SUM_OP_1_H
#define EAVL_PREFIX_SUM_OP_1_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch_io1.h"

#ifndef DOXYGEN

template <class F,
          class IO0>
struct cpuPrefixSumOp_1_function
{
    static void call(int n, bool &inclusive,
                     IO0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        if (inclusive)
        {
            o0[0*o0mul+o0add] = i0[((0/i0div)%i0mod)*i0mul+i0add];
            for (int i=1; i<n; ++i)
                o0[i*o0mul+o0add] = o0[(i-1)*o0mul+o0add] + i0[((i/i0div)%i0mod)*i0mul+i0add];
        }
        else
        {
            o0[0*o0mul+o0add] = 0;
            for (int i=1; i<n; ++i)
                o0[i*o0mul+o0add] = o0[(i-1)*o0mul+o0add] + i0[(((i-1)/i0div)%i0mod)*i0mul+i0add];
        }
    }
};


#if defined __CUDACC__

template<class T>
__device__ void scanSharedMemory(T *temp, int tid, int offset, int vals_per_block)
{
    // build sum in place up the tree
    for (int d = vals_per_block>>1; d > 0; d >>= 1)
    { 
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // clear the last element
    if (tid == 0)
        temp[vals_per_block - 1] = 0;

    // traverse down tree & build scan
    for (int d = 1; d < vals_per_block; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)                     
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; 
        }
    }

    __syncthreads();
}

// ----------------------------------------------------------------------------
template<bool INCLUSIVE, class T>
__global__ void singleBlockSimplePrefixSumKernel(int n,
                                                 T *input,
                                                 T *output)
{
    __shared__ T temp[512]; // enough for 256 threads
    int vals_per_block = 2*blockDim.x;
    int blockstart = 0; // assuming single block
    int tid = threadIdx.x;

    T init = 0;
    while (blockstart < n)
    {
        int offset = 1;
        // load input into shared memory
        int indexA = blockstart + 2*tid;
        int indexB = indexA + 1;

        T inA = 0;
        T inB = 0;
        if (indexA < n)
            inA = input[indexA];
        if (indexB < n)
            inB = input[indexB];

        temp[2*tid]   = inA;
        temp[2*tid+1] = inB;

        // save the last value for the next pass
        __syncthreads();
        T finalvalue = temp[2*blockDim.x - 1];

        // scan local results
        scanSharedMemory(temp, tid, offset, vals_per_block);

        // write results to device memory
        T outA = temp[2*tid];
        T outB = temp[2*tid+1];

        if (INCLUSIVE)
        {
            if (indexA < n)
                output[indexA] = init + outA + inA;
            if (indexB < n)
                output[indexB] = init + outB + inB;
        }
        else // EXCLUSIVE
        {
            if (indexA < n)
                output[indexA] = init + outA;
            if (indexB < n)
                output[indexB] = init + outB;
        }

        __syncthreads();
        init += finalvalue + temp[2*blockDim.x - 1];
        blockstart += vals_per_block;
    }
}  


// ----------------------------------------------------------------------------
template<bool INCLUSIVE, class T>
__global__ void prefixSumBlockwiseKernel_1(int n,
                                  T *i0, int i0div, int i0mod, int i0mul, int i0add,
                                  T *o0, int o0mul, int o0add,
                                  T *blockends)
{
    __shared__ T temp[512]; // enough for 256 threads
    int vals_per_block = 2*blockDim.x;
    int bid = blockIdx.y*gridDim.x + blockIdx.x;
    int blockstart = bid * vals_per_block;
    int tid = threadIdx.x;
    int offset = 1;

    // load input into shared memory
    int indexA = blockstart + 2*tid;
    int indexB = indexA + 1;
    T inA = 0;
    T inB = 0;
    if (indexA < n)
        inA = i0[((indexA/i0div)%i0mod)*i0mul+i0add];
    if (indexB < n)
        inB = i0[((indexB/i0div)%i0mod)*i0mul+i0add];

    temp[2*tid]   = inA;
    temp[2*tid+1] = inB;

    // scan local results
    scanSharedMemory(temp, tid, offset, vals_per_block);

    // write results to device memory
    T outA = temp[2*tid];
    T outB = temp[2*tid+1];

    if (INCLUSIVE)
    {
        if (indexA < n)
            o0[indexA*o0mul+o0add] = outA + inA;
        if (indexB < n)
            o0[indexB*o0mul+o0add] = outB + inB;
    }
    else // EXCLUSIVE
    {
        if (indexA < n)
            o0[indexA*o0mul+o0add] = outA;
        if (indexB < n)
            o0[indexB*o0mul+o0add] = outB;
    }

    if (2*tid+1 == vals_per_block-1)
        blockends[bid] = outB + inB;
}  

template <class T> 
__global__ void inplace_add_by_block(int n,
                                     T *o0, int o0mul, int o0add,
                                     T *blockvals)
{
    int bid = blockIdx.y*gridDim.x + blockIdx.x;
    int blockstart = bid * 2*blockDim.x;
    int tid = threadIdx.x;

    // this indexing matches the scan indexing, but it's likely slower here
    //int indexA = blockstart + 2*tid;
    //int indexB = indexA + 1;

    // instead, let's do this one and stride by blockDim
    int indexA = blockstart + tid;
    int indexB = indexA + blockDim.x;

    if (indexA < n)
        o0[indexA*o0mul+o0add] += blockvals[bid];
    if (indexB < n)
        o0[indexB*o0mul+o0add] += blockvals[bid];
}


template<class T>
struct PrefixSum_Temp_Storage
{
    static int nvals;
    static T *device;
    static T *host;
};

template <class T> int PrefixSum_Temp_Storage<T>::nvals = 0;
template <class T> T *PrefixSum_Temp_Storage<T>::device = NULL;
template <class T> T *PrefixSum_Temp_Storage<T>::host = NULL;



template <class F,
          class IO0>
struct gpuPrefixSumOp_1_function
{
    static void call(int n, bool &inclusive,
                     IO0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 256 threads
        int numThreads = 256;
        int valsPerBlock = numThreads * 2;
        int numBlocksTotal = (n + valsPerBlock-1) / valsPerBlock;

        int numBlocksX = numBlocksTotal;
        int numBlocksY = 1;
        if (numBlocksTotal >= 32768)
        {
            numBlocksY = numBlocksTotal / 32768;
            numBlocksX = (numBlocksTotal + numBlocksY-1) / numBlocksY;
        }
        if (numBlocksX*numBlocksY < numBlocksTotal)
            THROW(eavlException, "Internal error, miscalculated grid in prefix sum");

        if (PrefixSum_Temp_Storage<IO0>::nvals < numBlocksTotal)
        {
            if (PrefixSum_Temp_Storage<IO0>::device)
                cudaFree(PrefixSum_Temp_Storage<IO0>::device);
            CUDA_CHECK_ERROR();
            // allocate at least 4k
            PrefixSum_Temp_Storage<IO0>::nvals = (numBlocksTotal < 4096) ? 4096 : numBlocksTotal;
            cudaMalloc((void**)&PrefixSum_Temp_Storage<IO0>::device,
                       PrefixSum_Temp_Storage<IO0>::nvals * sizeof(IO0));
            PrefixSum_Temp_Storage<IO0>::host = new IO0[PrefixSum_Temp_Storage<IO0>::nvals];
            CUDA_CHECK_ERROR();
        }

        // scan each block.
        // collect per-block sums as you go.
        dim3 threads(numThreads,1,1);
        dim3 blocks(numBlocksX,numBlocksY,1);
        dim3 oneblock(1,1,1);
        CUDA_CHECK_ERROR();
        if (inclusive)
        {
            prefixSumBlockwiseKernel_1<true><<<blocks, threads>>>
                (n,
                 i0, i0div, i0mod, i0mul, i0add,
                 o0, o0mul, o0add,
                 PrefixSum_Temp_Storage<IO0>::device);
        }
        else
        {
            prefixSumBlockwiseKernel_1<false><<<blocks, threads>>>
                (n,
                 i0, i0div, i0mod, i0mul, i0add,
                 o0, o0mul, o0add,
                 PrefixSum_Temp_Storage<IO0>::device);
        }
        CUDA_CHECK_ERROR();

        // this is just debug output
        if (false)
        {
            cudaMemcpy(PrefixSum_Temp_Storage<IO0>::host,
                       PrefixSum_Temp_Storage<IO0>::device,
                       numBlocksTotal * sizeof(IO0),
                       cudaMemcpyDeviceToHost);
            for (int i=0; i<numBlocksTotal; i++)
                cout << "host["<<i<<"] = "<<PrefixSum_Temp_Storage<IO0>::host[i]<<endl;
        }


        // exclusive scan on the per-block sums
        singleBlockSimplePrefixSumKernel<false><<<oneblock, threads>>>
            (numBlocksTotal,
             PrefixSum_Temp_Storage<IO0>::device,
             PrefixSum_Temp_Storage<IO0>::device); // safe to do it in-place
        CUDA_CHECK_ERROR();

        // do the offset for each block to give final results
        inplace_add_by_block<<<blocks, threads>>>
            (n,
             o0, o0mul, o0add,
             PrefixSum_Temp_Storage<IO0>::device);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlPrefixSumOp_1
//
// Purpose:
///   A standard prefix sum operation, either inclusive or exclusive, on
///   a single input array, placing the result in a single output array.
//
// Programmer:  Jeremy Meredith
// Creation:    April 1, 2012
//
// Modifications:
// ****************************************************************************
class eavlPrefixSumOp_1 : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    bool inclusive;
    DummyFunctor functor;
  public:
    eavlPrefixSumOp_1(eavlArrayWithLinearIndex in0,
                      eavlArrayWithLinearIndex out0,
                      bool incl)
        : inArray0(in0), outArray0(out0), inclusive(incl)
    {
    }
    virtual void GoCPU()
    {
        int n = inArray0.array->GetNumberOfTuples();
        if (n == 0)
            return;

        eavlDispatch_io1<cpuPrefixSumOp_1_function>(n, eavlArray::HOST, inclusive,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        int n = inArray0.array->GetNumberOfTuples();
        if (n == 0)
            return;

        eavlDispatch_io1<gpuPrefixSumOp_1_function>(n, eavlArray::DEVICE, inclusive,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

#endif

