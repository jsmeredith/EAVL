// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SegScan_OP_H
#define EAVL_SegScan_OP_H

#include "eavlCUDA.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlOperation.h"
#include "eavlException.h"
#include <time.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN



struct eavlSegScanOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT, class FLAGS>
    static void call(int nitems, int,
                     const IN inputs, OUT outputs,
                     FLAGS flags, F&)
    {
        
    }
};

#if defined __CUDACC__

template<typename T>
class SegAddFunctor
{
  public:
    __device__ SegAddFunctor(){}

    __device__ T static op(T a, T b)
    {
        return a + b;
    }

    __device__ T static getIdentity(){return 0;}
};

template<typename T>
class SegMaxFunctor
{
  public:
    __device__ SegMaxFunctor(){}

    __device__ T static op(T a, T b)
    {
        return max(a, b);
    }

    __device__ T static getIdentity(){return 0;}
};


template<class OP, bool INCLUSIVE, class T>
__device__ T scanWarp(volatile T *data, const unsigned int idx = threadIdx.x)
{
    const unsigned int simdLane = idx & 31;
    if( simdLane >= 1  )  data[idx] = OP::op(data[idx -  1 ], data[idx]);
    if( simdLane >= 2  )  data[idx] = OP::op(data[idx -  2 ], data[idx]);
    if( simdLane >= 4  )  data[idx] = OP::op(data[idx -  4 ], data[idx]);
    if( simdLane >= 8  )  data[idx] = OP::op(data[idx -  8 ], data[idx]);
    if( simdLane >= 16 )  data[idx] = OP::op(data[idx -  16], data[idx]);
    if(INCLUSIVE)
    {
        return data[idx];
    }
    else
    {
        if(simdLane > 0)
        {
            return data[idx - 1];
        }
        else return OP::getIdentity();
    }
}

/*     Warp of 8 threads wide segemented scan example: 
       segFirstIdx is the warp index of the first item in the segment.
       If a segment starts from another warp, it is just 0.
       [1 2 1 1 0 1 0 1] data
       [0 0 0 0 1 0 0 0] flags
       [0 0 0 0 4 0 0 0] new flags (op1)
       [0 0 0 0 4 4 4 4] after max scan (op2)
       [  3 3 2   1 1 1] simdLane >= segFirstIdx + 1
       [    4 5     1 2] simdLane >= segFirstIdx + 2
       [1 3 4 5 0 1 1 2] last section shifts the data as neccesarry
*/
template<class OP, bool INCLUSIVE, class T>
__device__ T segScanWarp(volatile T *data, volatile int *flags, const unsigned int idx = threadIdx.x)
{
    const unsigned int simdLane = idx & 31;
    
    if( flags[idx] == 1 ) 
    {
        flags[idx] = simdLane;                                              /*op1*/
        //printf("Found start of segment at lane %d in thread %d", simdLane, idx);                      
    }
     unsigned int segFirstIdx = scanWarp< SegMaxFunctor<int>, true>(flags);  /*op2*/

    if( simdLane >= segFirstIdx + 1 )  data[idx] = OP::op(data[idx -  1 ], data[idx]);
    if( simdLane >= segFirstIdx + 2 )  data[idx] = OP::op(data[idx -  2 ], data[idx]);
    if( simdLane >= segFirstIdx + 4 )  data[idx] = OP::op(data[idx -  4 ], data[idx]);
    if( simdLane >= segFirstIdx + 8 )  data[idx] = OP::op(data[idx -  8 ], data[idx]);
    if( simdLane >= segFirstIdx + 16)  data[idx] = OP::op(data[idx -  16], data[idx]);
    //if(idx == 34) printf("Thread %d  flag %d %d lane %d\n", idx, segFirstIdx, flags[idx], simdLane);
    //if(idx == 35) printf("Thread %d  flag %d %d lane %d\n", idx, segFirstIdx, flags[idx], simdLane);
    //if(idx == 37) printf("Thread %d  flag %d %d lane %d\n", idx, segFirstIdx, flags[idx], simdLane);
    if(INCLUSIVE)
    {
        return data[idx];
    }
    else
    {
        if(simdLane > 0 && segFirstIdx != simdLane) /*return identity if first in exclusive segment */
        {
            return data[idx - 1];
        }
        else
        {

            return OP::getIdentity();
        } 
    }
}

template<class OP, bool INCLUSIVE, class T>
__device__ T segScanBlock(volatile T *data, volatile int *flags, const unsigned int idx = threadIdx.x)
{
    __shared__ int sm_data[128];
    __shared__ int sm_flags[128];


    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdx = idx >> 5;                              /*drop the first 32 bits */
    unsigned int warpFirstThread = warpIdx << 5;                  /* first warp thread idx, put some zeros back on */
    unsigned int warpLastThread  = warpFirstThread + 31;

    sm_data[idx]  = data[threadID];
    sm_flags[idx] = flags[threadID];

    __syncthreads();

    bool openSegment = (flags[warpFirstThread] == 0);             /* open segment running into warp(carry in if warpIdx !=0), need now since segscan will overwrite flags */
    __syncthreads();

    
    T result = segScanWarp<OP,INCLUSIVE>(sm_data,sm_flags);
    T lastVal = sm_data[warpLastThread];

    bool warpHasFlag = sm_flags[warpLastThread] != 0 || !openSegment; /* were there any segment flags in the warp*/
    bool acc = (openSegment && sm_flags[idx] == 0);                   /*this thread should add the carry in to its current result */   
    
    __syncthreads();

    /* the last thread in the warp stores the final value and a flag, if there was one in the warp, into a value in warp 0*/
    /* warp 0 is a storage location since all values are already correct(no carry in) and stored in result */
    if(idx == warpLastThread) 
    {
        sm_data[warpIdx]  = lastVal;
        sm_flags[warpIdx] = warpHasFlag;
        //printf("warpid: %d   %d\n",warpIdx,data[warpIdx] );
    }

    __syncthreads();
    /*The first warp in each block now seg scans(inclusive) all the warp totals and flags */
    if(warpIdx == 0) segScanWarp<OP,true>(sm_data,sm_flags,idx);
    __syncthreads();

    //if(idx == 35 ) printf("Thread 35 : %d %d", acc, result);
    /* Now add carry over from the previous warp into the individually scanned warps */
    if(warpIdx != 0 && acc)
    {
        result = OP::op(sm_data[warpIdx-1], result);
        //if(idx == warpFirstThread) printf("warpid: %d   %d\n",warpIdx,data[warpIdx-1] );
    }
     __syncthreads();

    sm_data[idx] = result;
    
    __syncthreads();

    data[threadID] = sm_data[idx];

     __syncthreads();
    return result;
}


template<class OP, bool INCLUSIVE, class T>
__device__ T segScanGrid(volatile T *data, volatile int *flags)
{
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;    
    unsigned int blockId = blockIdx.x ;                              
    unsigned int blockFirstThread = blockIdx.x << 7;                  /* first block thread idx, put some zeros back on */
    unsigned int blockLastThread  = blockFirstThread + 127;
    
    bool openSegment = (flags[blockFirstThread] == 0);                /* open segment running into block(carry in if blockIdx !=0)*/ 
    if(threadID == 129) printf("129 Blockid: %d   %d %d  %d\n",blockId, data[threadID], threadID,  flags[threadID] );
    if(threadID == 128) printf("128 Blockid: %d   data %d thread id %d flag %d \n",blockId, data[threadID], threadID,  flags[threadID] );
    __syncthreads();                                                  /* stroe it now since segscan will overwrite flags */

    
    
    T result = segScanBlock<OP,INCLUSIVE>(data,flags);
    T lastVal = data[blockLastThread];

    bool blockHasFlag = flags[blockLastThread] != 0 || !openSegment;      /* were there any segment flags in the block*/
    bool acc = (openSegment && flags[threadID] == 0);                     /*this thread should add the carry in to its current result */   
    
    if(threadID == 128) printf("128 Blockid: %d   data %d thread id %d flag %d \n",blockId, data[threadID], threadID,  flags[threadID] );
    __syncthreads();

    /* the last thread in the block stores the final value and a flag, if there was one in the block, into a value in block 0*/
    /* block 0 is a storage location since all values are already correct(no carry in) and stored in result */
    if(threadID == blockLastThread) /*this is not getting called with the block that ends early*/
    {
        data[blockId]  = lastVal;
        flags[blockId] = blockHasFlag;
        printf("Last Blockid: %d   %d\n",blockId,threadID );
    }

    
    if(threadID == blockFirstThread) printf("Blockid: %d   %d %d %d %d\n",blockId, data[blockId], threadID,acc,  flags[threadID] );
    __syncthreads();
    /*The first block in each block now seg scans(inclusive) all the block totals and flags */
    if(blockId == 0) segScanBlock<OP,true>(data,flags);
    __syncthreads();
    if(blockId != 0 && acc)
    {
        if(threadID == 128) printf("128 accing %d %d %d\n", result,data[blockId-1], blockId );
        //result = OP::op(data[blockId-1], result);
        if(threadID == 128) printf("128 accing %d %d %d\n", result,data[blockId-1], blockId );
        if(threadID == 123) printf("123 Should not happen accing %d\n", result);
    }
    __syncthreads();
    if(threadID == 1) printf(" 1 Blockid: %d   %d %d %d %d\n",blockId, data[blockId], threadID,acc,  flags[threadID] );
    if(threadID == 2) printf(" 2 Blockid: %d   %d %d %d %d\n",blockId, data[blockId], threadID,acc,  flags[threadID] );
    if(threadID == 0)   printf("0 Data : %d %d %d \n", data[0], data[1], data[2]);
    if(threadID == 0)   printf("0 flags : %d %d %d \n", flags[0], flags[1], flags[2]);
    if(threadID == 128) printf("**Data : %d %d %d \n", data[0], data[1], data[2]);
    if(threadID == 128) printf("**flags : %d %d %d \n", flags[0], flags[1], flags[2]);
    if(threadID == 128) printf("128 Blockid: %d   data %d thread id %d flag %d result %d, acc %d\n",blockId, data[threadID], threadID,  flags[threadID], result,acc );
    __syncthreads();

    /* Now add carry over from the previous block into the individually scanned blocks */
     __syncthreads();
    if(threadID == 128) printf("A Data : %d %d %d \n", data[0], data[1], data[2]);
    if(threadID == 128) printf("A flags : %d %d %d \n", flags[0], flags[1], flags[2]);
     
    if(threadID == 128) printf("B Data : %d %d %d \n", data[0], data[1], data[2]);
    if(threadID == 128) printf("B flags : %d %d %d \n", flags[0], flags[1], flags[2]);
     __syncthreads();

    data[threadID] = result;
    
    __syncthreads();

    return result;
}

template <class IN, class OUT, class FLAGS>
__global__ void
eavlSegScanOp_kernel(int nitems,
                     IN inputs, OUT outputs,
                    FLAGS _flags)
{
    int *flags  = get<0>(_flags).array;
    int *input  = get<0>(inputs).array;
    int *output = get<0>(outputs).array;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    //const int numThreads = blockDim.x * gridDim.x;
    
    
    if(threadID > nitems) return;
    /*load arrays into share memory */
    

    
    
    //sm_data[threadIdx.x] = segScanGrid< SegAddFunctor<int>, false >(data,flags);
    output[threadID] =  segScanGrid< SegAddFunctor<int>, false >(input,flags);

    /*copy results into the output*/
    //output[threadID] = sm_data[threadIdx.x];
    __syncthreads();

}




struct eavlSegScanOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT, class FLAGS>
    static void call(int nitems, int,
                     const IN inputs, OUT outputs,
                     FLAGS flags, F&)
    {
        int numThreads = 128;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (64,           1, 1);
        eavlSegScanOp_kernel<<< blocks, threads >>>(nitems,
                                                   inputs, outputs,
                                                   flags);
        CUDA_CHECK_ERROR();
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlSegScanOp
//
//  Purpose: Segmented scan provides a parallel primitve building block.
///          The operation performs the same operation as scan, but adds
///          a flag array to perform the scan on arbitrary segments within 
///          the input array. Flags are integers that are 0 or 1, where
///          a 1 marks the beginning of a new segment.
//                        
//
// Programmer: Matt Larsen 8/3/2014
//
// Modifications:
//    
// ****************************************************************************
template <class I, class O, class FLAGS>
class eavlSegScanOp : public eavlOperation
{
  protected:
    DummyFunctor functor;
    I            inputs;
    O            outputs;
    FLAGS        flags;
    int          nitems;
  public:
    eavlSegScanOp(I i, O o, FLAGS flgs)
        : inputs(i), outputs(o), flags(flgs), nitems(-1)
    {
    }
    eavlSegScanOp(I i, O o, FLAGS flgs, int itemsToProcess)
        : inputs(i), outputs(o), flags(flgs), nitems(itemsToProcess)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
        int n=0;
        if(nitems > 0) n = nitems;
        else n = inputs.first.length();
        eavlOpDispatch<eavlSegScanOp_CPU>(n, dummy, inputs, outputs, flags, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        int dummy;
        int n=0;
        if(nitems > 0) n = nitems;
        else n = inputs.first.length();
        eavlOpDispatch<eavlSegScanOp_GPU>(n, dummy, inputs, outputs, flags, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class FLAGS>
eavlSegScanOp<I,O,FLAGS> *new_eavlSegScanOp(I i, O o, FLAGS flags) 
{
    return new eavlSegScanOp<I,O,FLAGS>(i,o,flags);
}

template <class I, class O, class FLAGS>
eavlSegScanOp<I,O,FLAGS> *new_eavlSegScanOp(I i, O o, FLAGS flags, int itemsToProcess) 
{
    return new eavlSegScanOp<I,O,FLAGS>(i,o,flags, itemsToProcess);
}


#endif
