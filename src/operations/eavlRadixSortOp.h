// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RADIX_SORT_OP_H
#define EAVL_RADIX_SORT_OP_H

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



struct eavlRadixSortOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int,
                     const IN inputs, OUT outputs,
                      F&)
    {
        
    }
};

#if defined __CUDACC__

template<typename T>
class RadixAddFunctor
{
  public:
    __device__ RadixAddFunctor(){}

    __device__ T static op(T a, T b)
    {
        return a + b;
    }

    __device__ T static getIdentity(){return 0;}
};



template<class OP, bool INCLUSIVE, class T>
__device__ T rscanWarp(volatile T *data, const unsigned int idx = threadIdx.x)
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

/*Exclusive scan of block length shared memory */
template< class T>
__device__ void scanBlock(volatile T *flags)
{

    const unsigned int idx = threadIdx.x;
    //int blockId   = blockIdx.y * gridDim.x + blockIdx.x;

    //const int threadID     = blockId * blockDim.x + threadIdx.x;
    unsigned int warpIdx   = idx >> 5;
    unsigned int warpFirstThread = warpIdx << 5;                  
    unsigned int warpLastThread  = warpFirstThread + 31;

    T result = rscanWarp<RadixAddFunctor<T>,true>(flags);
    
    __syncthreads();

    if(idx == warpLastThread) flags[warpIdx] = result;
    
    __syncthreads();
    
    if(warpIdx == 0) rscanWarp<RadixAddFunctor<T>,true>(flags);
    
    __syncthreads();

    if(warpIdx !=0 ) result = flags[warpIdx-1] + result;
    
    __syncthreads();
    
    flags[idx] = result;
    
    __syncthreads();

    if(idx != 0 ) flags[idx] = flags[idx -1];
    else flags[0] = 0;

}

__global__ void interatorKernel(int nitems, volatile int * ids)
{
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadID = blockId * blockDim.x + threadIdx.x;

    if(threadID > nitems) return;
    ids[threadID] = threadID;
}

template<class K, class T>
__global__ void sortBlocksKernel(int nitems, volatile K *keys, volatile T *ids)
{
    const unsigned int idx = threadIdx.x;
    RadixAddFunctor<T> op;
    volatile __shared__ K sm_keys[1024];
    volatile __shared__ K sm_mask[1024];  /*current bit set*/
    volatile __shared__ K sm_nmask[1024]; /*current bit not set*/
    volatile __shared__ K sm_t[1024];
    volatile __shared__ T sm_ids[1024];
    K mask = 1;
    __shared__ int totalFalses; 
    __shared__ int lastVal;
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadID = blockId * blockDim.x + threadIdx.x;
    //unsigned int blockFirstThread = blockId << 8;
    //unsigned int blockLastThread = blockFirstThread + 255; 
    
    if( threadID > nitems) return;
    
    sm_keys[idx] = keys[threadID];
    sm_ids[idx]  = ids[threadID];
    /* Change this to sizeOf(K)*8 or something, or find msb using max scan*/
    for(int i = 0; i<32; i++)
    {   
        sm_mask[idx] = sm_keys[idx] & mask;
        //printf(" mask %d key %d\n", sm_mask[idx], sm_keys[idx]);
        /*flip it*/
        sm_nmask[idx] = (sm_mask[idx] > 0) ? 0 : 1; 
        
        if(idx == 1023) lastVal = sm_nmask[1023];  

        __syncthreads(); 

        /* creates the scatter indexes of the true bits */
        scanBlock(sm_nmask);

        __syncthreads();

        if(idx == 1023) totalFalses = sm_nmask[1023] + lastVal;
        //if(idx == 0) printf("TotalFalses %d nitems %d lastValue %d\n", totalFalses, nitems, lastVal);
        __syncthreads();

        /*scatter indexes of false bits*/
        sm_t[idx] = idx - sm_nmask[idx] + totalFalses;
        //printf("Thread Id %d   f %d t %d current mask %d current Key %d\n", threadID, sm_nmask[idx], sm_t[idx], sm_mask[idx], sm_keys[idx]);
        /*re-use nmask store the combined scatter indexes */
        sm_nmask[idx]  = (sm_mask[idx] > 0) ?  sm_t[idx] : sm_nmask[idx]; 
        /* Hold onto the old values so race conditions don't blow it away */
        T value = sm_ids[idx];
        K key   = sm_keys[idx];
        int sIndex = sm_nmask[idx]; 
        //printf("Thread Id %d with scatter index of %d key %d value %d  \n", threadID, sIndex, key, value);    
        __syncthreads();

        /* scatter */
        
        sm_ids[sIndex]  = value;
        sm_keys[sIndex] = key;
        mask = mask << 1;
        __syncthreads();
        if(idx==0)
        {   //printf("-------------------\n");
            //for (int j=0; j<nitems; j++) printf("%d ", sm_keys[j]);
            //printf("\n\n\n New Mask %d\n", mask);
        }
        __syncthreads();
    }
    //printf("Copying data back\n");
    /*copy back to global array*/
    keys[threadID] = sm_keys[idx];
    ids[threadID]  = sm_ids[idx];
    //printf("After Copying data back\n");
    __syncthreads();

}






struct eavlRadixSortOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int,
                     IN inputs, OUT outputs, F&)
    {
                 
        int *_inputs  = get<0>(inputs).array;
        int *_outputs = get<0>(outputs).array;


        int numBlocks = nitems / 1024;
        if(nitems%256 > 0) numBlocks++;

        int numBlocksX = numBlocks;
        int numBlocksY = 1;
        if (numBlocks >= 32768)
        {
            numBlocksY = numBlocks / 32768;
            numBlocksX = (numBlocks + numBlocksY-1) / numBlocksY;
        }
        //cudaMalloc((void **)&tempFlags,nitems*sizeof(int));
        //CUDA_CHECK_ERROR();
        //cudaMemcpy(tempFlags, &(_flags[0]),
        //           nitems*sizeof(int), cudaMemcpyDeviceToDevice);
        //CUDA_CHECK_ERROR();
        //cudaMemcpy(_outputs, &(_inputs[0]),
        //           nitems*sizeof(int), cudaMemcpyDeviceToDevice);
        //CUDA_CHECK_ERROR();
        

         
        int numThreads = 1024;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (numBlocksX,numBlocksY, 1);

        interatorKernel<<< blocks, threads >>>(nitems, _outputs);
        CUDA_CHECK_ERROR();
        sortBlocksKernel<<< blocks, threads >>>(nitems, _inputs, _outputs);
        CUDA_CHECK_ERROR();
        

    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlRadixSortOp
//
//  Purpose: 
///          
///          
///          
///          
//                        
//
// Programmer: Matt Larsen 8/19/2014
//
// Modifications:
//    
// ****************************************************************************
template <class I, class O>
class eavlRadixSortOp : public eavlOperation
{
  protected:
    DummyFunctor functor;
    I            inputs;
    O            outputs;
    int          nitems;
  public:
    eavlRadixSortOp(I i, O o)
        : inputs(i), outputs(o), nitems(-1)
    {
    }
    eavlRadixSortOp(I i, O o, int itemsToProcess)
        : inputs(i), outputs(o), nitems(itemsToProcess)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
        int n=0;
        if(nitems > 0) n = nitems;
        else n = inputs.first.length();
        eavlOpDispatch<eavlRadixSortOp_CPU>(n, dummy, inputs, outputs, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        int dummy;
        int n=0;
        if(nitems > 0) n = nitems;
        else n = inputs.first.length();
        eavlOpDispatch<eavlRadixSortOp_GPU>(n, dummy, inputs, outputs, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O>
eavlRadixSortOp<I,O> *new_eavlRadixSortOp(I i, O o) 
{
    return new eavlRadixSortOp<I,O>(i,o);
}

template <class I, class O>
eavlRadixSortOp<I,O> *new_eavlRadixSortOp(I i, O o, int itemsToProcess) 
{
    return new eavlRadixSortOp<I,O>(i,o, itemsToProcess);
}


#endif