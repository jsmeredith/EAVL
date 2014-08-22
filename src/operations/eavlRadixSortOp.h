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

template<typename T>
class RadixMaxFunctor
{
  public:
    __device__ RadixMaxFunctor(){}

    __device__ T static op(T a, T b)
    {
        return max(a, b);
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
template<class OP, bool INCLUSIVE, class T>
__device__ void scanBlock(volatile T *flags)
{

    const unsigned int idx = threadIdx.x;
    unsigned int warpIdx   = idx >> 5;
    unsigned int warpFirstThread = warpIdx << 5;                  
    unsigned int warpLastThread  = warpFirstThread + 31;

    T result = rscanWarp<OP,true>(flags);
    
    __syncthreads();

    if(idx == warpLastThread) flags[warpIdx] = result;
    
    __syncthreads();
    
    if(warpIdx == 0) rscanWarp<OP,true>(flags);
    
    __syncthreads();

    if(warpIdx !=0 ) result = OP::op(flags[warpIdx-1], result);
    
    __syncthreads();
    
    flags[idx] = result;
    
    __syncthreads();

    if(!INCLUSIVE)
    {
        if(idx != 0 ) result = flags[idx -1];
        __syncthreads();

        if(idx != 0 ) flags[idx] = result;
        else flags[0] = OP::getIdentity();
        __syncthreads();
    }
    
    
}


__global__ void interatorKernel(int nitems, volatile int * ids)
{
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadID = blockId * blockDim.x + threadIdx.x;

    if(threadID > nitems) return;
    ids[threadID] = threadID;
}

/*
    Do a binary search on sorted shared memory to find the scatter
    position of the key in a merge sort.

    search 1 will insert identical elements before those of the second array.
    search 2 will insert identical elements after. This ensures that we do
    not get scatter address collisions.
*/
template<class T>
__device__ int binarySearch(int first, int last, volatile T * array, volatile T key)
{
    int index = -99;
    bool found = false;
    bool end = false;


    int mx = last;
    int mn = first;

    if(last - first < 0 )
    {
        printf(" 1 nothing left\n");
        index = 0;
        end = true;
        return index;
    }

    if(key > array[last]) 
    {
        index = -1;
        end = true;
        printf("greater than last array key %d  last val %d id %d \n", key, array[last], threadIdx.x);
    }


    if(key <= array[first]) 
    {
        index = 0;
        end = true;
        printf("Less than first array %d id %d\n", key, threadIdx.x);
    }

    while(!end  && ! found)
    {
        int gap = mx - mn;
        int mid = mn + gap/2;
        end = gap < 1;
        if(array[mid-1] < key && key <=array[mid])
        {
            found = true;

            index = mid;
            //if(key == array[mid]) index = mid;
        }

        if(key <= array[mid]) mx = mid -1;
        else mn = mid +1;
    }

    return index;

}
template<class T>
__device__ int binarySearchPrint(int first, int last,volatile T * array, T key)
{
    int index = -99;
    bool found = false;
    bool end = false;


    int mx = last;
    int mn = first;

    if(key > array[last]) 
    {
        index = -1;
        end = true;
        printf("greater than last array %d id %d\n", key, threadIdx.x);
    }

    if(key <= array[first]) 
    {
        index = 0;
        end = true;
        printf("Less than first array %d id %d\n", key, threadIdx.x);
    }


        
        while(!end  && ! found)
    {
        int gap = mx - mn;
        int mid = mn + gap/2;
        end = gap < 1;
        printf("Key %d Mid %d Gap %d MidVal %d Mid1val %d\n", key, mid, gap, array[mid], array[mid-1]);
        if(array[mid-1] < key && key <=array[mid])
        {
            found = true;

            index = mid;
            //if(key == array[mid]) index = mid;
        }

        if(key <= array[mid]) mx = mid -1;
        else mn = mid +1;
    }

    return index;

}

template<class T>
__device__ int binarySearch2(int first, int last,volatile T * array, volatile T key)
{
    int index = -1;
    bool found = false;
    bool end = false;


    int mx = last;
    int mn = first;

    if(last - first < 0 )
    {
        printf("nothing left\n");
        index = 0;
        end = true;
        return index;
    }

    if(key > array[last]) 
    {
        index = -1;
        end = true;
    }

    if(key == array[last])
    {
        index = last + 1;
        end = true;
    }

    if(key < array[first]) 
    {
        index = 0;
        end = true;
    }

    while(!end  && ! found)
    {
        int gap = mx - mn;
        int mid = mn + gap/2;
        end = gap < 1;
        if(array[mid] <= key && key < array[mid + 1])
        {
            found = true;
            index = mid +1;
        }

        if(key < array[mid]) mx = mid -1;
        else mn = mid +1;
    }

    return index;

}

template<class K, class T>
__global__ void mergeKernel(int nItems,int first1, int last1, int first2, int last2, volatile K *keys, volatile K *outKeys, volatile T* ids, K maxKey)
{
    const int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int idx = threadIdx.x;
    const int threadID = blockId * blockDim.x + threadIdx.x;
    printf("Nitems %d \n", nItems);
    volatile __shared__ K    sm_keys1[64];
    volatile __shared__ K    sm_keys2[64];
    volatile __shared__ K    sm_out[128];  /*write out to a larger buffer so we don't scatter to global mem*/
    volatile __shared__ int  sm_sa[64];    /*need to scan all the scatter addresses to find the max*/


    __shared__ int keys1Start;  /* should always point to the beginning of the current working chunk*/
    __shared__ int writePtr;
    __shared__ int keys2Start;
    __shared__ int maxScatter;

    __shared__ int partialShift1;
    __shared__ int partialShift2;
    __shared__ int shiftAmount;
    __shared__ int done;
    //int counter=0;
    //__shared__ int gStart = first1;

    if(idx == 0) 
    {
        printf("Input------------------------------\n");
        for(int i=0 ; i<256 ; i++)
        {
            printf("%d ", keys[i]);
        }
        printf("\n");
    }

    if(idx == 0)
    {
        keys1Start = first1;
        keys2Start = first2;
        writePtr = first1;
        done = 0;
        printf("Start 1 %d Start 2 %d\n",keys1Start, keys2Start);
        
    }
    __syncthreads();
    //partialShift1 = false;

    /* load starting chunks into share mem*/
    sm_keys1[idx] = keys[keys1Start+idx];
    sm_keys2[idx] = keys[keys2Start+idx];
    
    __syncthreads();
    while(!done)
    {
        int numItems1 = last1 - keys1Start; /* 0 = one left */
        int numItems2 = last2 - keys2Start;
        if(numItems1 > 63) numItems1 = 63;
        if(numItems2 > 63) numItems2 = 63;
        numItems1 = 63;
        numItems2 = 63;
        if(idx == 0 ) printf("Items left 1 %d 2 %d\n", numItems1, numItems2);
        /* -1 indicates that the keys are greater than the last value of the other key array */
        int sa1 = binarySearch(0,numItems2,sm_keys2, sm_keys1[idx]);
        if(sa1 != -1) sa1 += idx;
        if(numItems1 < 0) sa1 = -1;
        //if(numItems2 < 63 && sa1 == -1) sa1 = numItems2 + 1 + idx; /* Nothing left on the other list of keys, so it is safe to append it*/
        int sa2 = binarySearch2(0,numItems1,sm_keys1, sm_keys2[idx]);
        if(numItems2 < 0) sa2 = -1;
        if(sa2 != -1) sa2 += idx;
        //if(numItems1 < 63 && sa2 == -1) sa2 = numItems1 + 1 + idx;
         
        __syncthreads();
        printf("Thread %d key1 %d  sa1 %d  key2 %d sa2 %d\n", idx, sm_keys1[idx], sa1, sm_keys2[idx], sa2);
        
        if(sa1 != -1)
        {
            sm_out[sa1] = sm_keys1[idx];
        } 

        if(sa2 != -1) sm_out[sa2] = sm_keys2[idx];
        /* end condition test */
        if(idx == 0)
        {
            partialShift1 = 0;
            partialShift2 = 0;
        }

        __syncthreads();

        /*Find the maximum scatter address*/    
        sm_sa[idx] = sa1;

        __syncthreads();
        
        scanBlock<RadixMaxFunctor<int>, true> (sm_sa);
        
        __syncthreads();
        
        if(idx == 0) maxScatter = sm_sa[63];
        
        sm_sa[idx] = sa2;
        
        __syncthreads();
        
        scanBlock<RadixMaxFunctor<int>, true> (sm_sa);
        
        __syncthreads();
        
        if(idx == 0) 
        {
            if(maxScatter < sm_sa[63]) maxScatter = sm_sa[63];
        }
        
        __syncthreads();

        /* write the output back to global mem */
        if(idx == 0) printf("Max SCATTER %d writePtr %d\n", maxScatter, writePtr);
        if(idx <= maxScatter)
        {
            if(writePtr + idx < nItems)
            { 
                outKeys[writePtr + idx] = sm_out[idx];
                printf("writing to gmem at %d with value %d ITEMS %d\n",writePtr + idx, sm_out[idx], nItems); 
            } 
            if(idx == 0)
            {
                printf("Out:\n");
                for(int i=0; i<128;i++) printf("%d ", sm_out[i]);
                printf("\n");
            }

            
        } 
        if(idx + 64 <= maxScatter && writePtr + idx + 64 < nItems)
        { 
            outKeys[writePtr + idx + 64] = sm_out[idx + 64];
            printf("Writing to address %d num Items %d \n",writePtr + idx + 64, nItems );
        }
        
        __syncthreads();
        /*advance the write pointer*/
        if(idx == 0) writePtr += maxScatter + 1;
        
        /*calc the shift if there is one*/
        if( sa1 == -1 ) partialShift1 = 1;
        if( sa2 == -1 ) partialShift2 = 1;

        
        __syncthreads();

        if(partialShift1 == 1 && partialShift2 == 1)
        {
            printf("THIS SHOULD NEVER HAPPEN\n");
        }


        /* TODO: make this a 2d array and just index into the partial*/
        if(partialShift1)
        {   if(idx == 0) printf("Entering shift 1 \n");
            
            

            sm_sa[idx] = sa1 == -1 ? 1 : 0;
            scanBlock<RadixAddFunctor<int>,true>(sm_sa);
            /* all the -1 will be grouped at the end so find the first 1*/
            if(sm_sa[idx] == 1 ) 
            {
                shiftAmount = idx;
                printf("sm1 Shift amount %d\n", shiftAmount);
            }
            K shiftVal = sm_keys1[idx];
            __syncthreads();
            if(idx == 0)
            {
                keys2Start += 64;
                keys1Start += shiftAmount;
            } 
            /* perform the shift*/
            if(idx >= shiftAmount) sm_keys1[idx - shiftAmount] = shiftVal;
            
            __syncthreads();
            
            if(idx == 0)
            {
                printf("Key start 1 %d keyStart2 %d last2 %d\n", keys1Start, keys2Start, last2);
            }

            /*Pull in new chunk for the other array*/
            int newIdx = (keys2Start + idx <= last2) ? keys2Start + idx : maxKey;
            
            if(newIdx != maxKey) sm_keys2[idx] = keys[newIdx];
            else sm_keys2[idx] = newIdx;
            printf("Thread %d with new index %d with value %d\n", idx, newIdx, sm_keys2[idx]);
            
            __syncthreads();
            /*pull in new values to fill remaining empty slots */
            //newIdx = 
            int readStart = keys1Start - shiftAmount + 64 + idx;
            if (idx < shiftAmount ) 
            {
                //printf("Before Keys 1 Thread %d Getting new index for shift %d value %d into sm at %d\n",idx, readStart , sm_keys1[64-shiftAmount+ idx],64-shiftAmount+ idx );
                
                if(readStart > last1) sm_keys1[64 - shiftAmount + idx] = maxKey;
                else sm_keys1[64 - shiftAmount + idx] = keys[readStart];
                printf("Keys 1 Thread %d Getting new index for shift %d value %d into sm at %d\n",idx, readStart , sm_keys1[64-shiftAmount+ idx],64-shiftAmount+ idx );
            }
            //if(idx == 0) sm_keys1[2]=99;
            
            __syncthreads();
        }
        
        if(partialShift2)
        {   if(idx == 0) printf("Entering shift 2 \n");
            
            

            sm_sa[idx] = sa2 == -1 ? 1 : 0;
            scanBlock<RadixAddFunctor<int>,true>(sm_sa);
            /* all the -1 will be grouped at the end so find the first 1*/
            if(sm_sa[idx] == 1 ) 
            {
                shiftAmount = idx;
                printf("sm2 Shift amount %d\n", shiftAmount);
            }
            K shiftVal = sm_keys2[idx];
            __syncthreads();
            if(idx == 0)
            {
                keys1Start += 64;
                keys2Start += shiftAmount;
            } 
            /* perform the shift*/
            if(idx >= shiftAmount) sm_keys2[idx - shiftAmount] = shiftVal;
            
            __syncthreads();
            
            if(idx == 0)
            {
                printf("Key start 2 %d keyStart1 %d last1 %d\n", keys2Start, keys1Start, last1);
            }

            /*Pull in new chunk for the other array*/
            int newIdx = (keys1Start + idx <= last1) ? keys1Start + idx : maxKey;
            
            if(newIdx != maxKey) sm_keys1[idx] = keys[newIdx];
            else sm_keys1[idx] = newIdx;
            printf("Thread %d with new index %d with value %d\n", idx, newIdx, sm_keys1[idx]);
            
            __syncthreads();
            /*pull in new values to fill remaining empty slots */
            //newIdx = 
            int readStart = keys2Start - shiftAmount + 64 + idx;
            if (idx < shiftAmount ) 
            {
                //printf("Before Keys 1 Thread %d Getting new index for shift %d value %d into sm at %d\n",idx, readStart , sm_keys1[64-shiftAmount+ idx],64-shiftAmount+ idx );
                if(readStart > last2) sm_keys2[64 - shiftAmount + idx] = maxKey;
                else sm_keys2[64 - shiftAmount + idx] = keys[readStart];
                printf("Keys 2 Thread %d Getting new index for shift %d value %d into sm at %d\n",idx, readStart , sm_keys2[64-shiftAmount+ idx],64-shiftAmount+ idx );
            }
            //if(idx == 0) sm_keys1[2]=99;
            
            __syncthreads();
        }
        
        if(!partialShift1 && !partialShift2)
        {
            if(idx == 0)
            {
                keys2Start += 64;
                keys1Start += 64;
            } 

            if(idx == 0) printf("Entering shift both \n");
            if(idx == 0)
            {
                printf("Key start 1 %d keyStart2 %d last2 %d\n", keys1Start, keys2Start, last2);
            }

            int newIdx = (keys1Start + idx <= last1) ? keys1Start + idx : maxKey;
            if(newIdx != maxKey) sm_keys1[idx] = keys[newIdx];
            else sm_keys1[idx] = newIdx;

            newIdx = (keys2Start + idx <= last2) ? keys2Start + idx : maxKey;
            if(newIdx != maxKey) sm_keys2[idx] = keys[newIdx];
            else sm_keys2[idx] = newIdx;

            __syncthreads();
        }


        if(idx == 0)
        {
            printf("Max scatter address %d\n", maxScatter);
            printf("New keys 1\n");
            for(int i = 0; i < 64;  i++)
            {
                printf("%d ", sm_keys1[i]);
            }
            printf("\n");
            printf("New keys 2\n");
            for(int i = 0; i < 64;  i++)
            {
                printf("%d ", sm_keys2[i]);
            }
            printf("\n");
            printf("KEYS \n");
            for(int i = 0; i < 256;  i++)
            {
                printf("%d ", outKeys[i]);
            }
            printf("\n");
        }


       
        if(idx == 0) 
        {
            //if(counter == 2 ) done = 1;
            //counter++;  
            if(keys1Start > last1 && keys2Start > last2) done = 1;
        }
        __syncthreads();
    }

}


template<class K, class T>
__global__ void sortBlocksKernel(int nitems, volatile K *keys, volatile T *ids)
{

    const unsigned int idx = threadIdx.x;
    //if(idx == 0) printf("*******************************************STARTING*******\n");
    //RadixAddFunctor<T> op;
    volatile __shared__ K sm_keys[128];
    volatile __shared__ K sm_mask[128];  /*current bit set*/
    volatile __shared__ K sm_nmask[128]; /*current bit not set*/
    volatile __shared__ K sm_t[128];
    volatile __shared__ T sm_ids[128];
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
    {   //if(idx==0) printf("---------------------------------------- round %d\n",i);
        sm_mask[idx] = sm_keys[idx] & mask;
        //printf(" mask %d key %d\n", sm_mask[idx], sm_keys[idx]);
        /*flip it*/
        sm_nmask[idx] = (sm_mask[idx] > 0) ? 0 : 1; 
        
        if(idx == 127) lastVal = sm_nmask[127];  

        __syncthreads(); 

        /* creates the scatter indexes of the true bits */
        scanBlock<RadixAddFunctor<T>,false>(sm_nmask);

        __syncthreads();

        if(idx == 127) totalFalses = sm_nmask[127] + lastVal;
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
        //if(idx==0)
        //{   printf("-------------------\n");
        //    for (int j=0; j<64; j++) printf("%d ", sm_keys[j]);
        //    printf("\n\n\n New Mask %d\n", mask);
        //}
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
                 
        int *_keys  = get<0>(inputs).array;
        int *_values = get<0>(outputs).array;
        int *tempKeys;

        int numBlocks = nitems / 64;
        if(nitems%256 > 0) numBlocks++;

        int numBlocksX = numBlocks;
        int numBlocksY = 1;
        if (numBlocks >= 32768)
        {
            numBlocksY = numBlocks / 32768;
            numBlocksX = (numBlocks + numBlocksY-1) / numBlocksY;
        }
        cudaMalloc((void **)&tempKeys,nitems*sizeof(int));
        CUDA_CHECK_ERROR();
        //cudaMemcpy(tempFlags, &(_flags[0]),
        //           nitems*sizeof(int), cudaMemcpyDeviceToDevice);
        //CUDA_CHECK_ERROR();
        //cudaMemcpy(_values, &(_keys[0]),
        //           nitems*sizeof(int), cudaMemcpyDeviceToDevice);
        //CUDA_CHECK_ERROR();
        


        dim3 one(1,1,1);
        dim3 t(64,1,1);
        int numThreads = 128;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (numBlocksX,numBlocksY, 1);

        interatorKernel<<< blocks, threads >>>(nitems, _values);
        CUDA_CHECK_ERROR();
        sortBlocksKernel<<< blocks, threads >>>(nitems, _keys, _values);
        CUDA_CHECK_ERROR();
        
        mergeKernel<<< one, t>>>(nitems,0,127,128,255, _keys, tempKeys, _values,std::numeric_limits<int>::max()); 
        CUDA_CHECK_ERROR();
        cudaMemcpy( &(_keys[0]),tempKeys,
                   nitems*sizeof(int), cudaMemcpyDeviceToDevice);
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