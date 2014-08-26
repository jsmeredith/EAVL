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

__global__ void printKernel(int nitems, volatile int * ids)
{

    printf("temp keys \n");
    for( int i=0 ; i< nitems ; i++)
    {
        printf("%d ", ids[i]);
    }
    printf("\n");
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
        //printf(" 1 nothing left\n");
        index = 0;
        end = true;
        return index;
    }

    if(key > array[last]) 
    {
        index = -1;
        end = true;
        //printf("greater than last array key %d  last val %d id %d \n", key, array[last], threadIdx.x);
    }


    if(key <= array[first]) 
    {
        index = 0;
        end = true;
        //printf("Less than first array %d id %d\n", key, threadIdx.x);
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
       
    }

    if(key <= array[first]) 
    {
        index = 0;
        end = true;
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
        }

        if(key <= array[mid]) mx = mid -1;
        else mn = mid +1;
    }

    return index;

}

template<class T>
__device__ int binarySearch2(int first, int last,volatile T * array, volatile T key)
{
    int index = -99;
    bool found = false;
    bool end = false;


    int mx = last;
    int mn = first;

    if(last - first < 0 )
    {
        //printf("nothing left\n");
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

#define BLOCK_WIDTH 512
#define BLOCK_MAX 511
template<class K, class T>
__device__ void merge(int nItems, int first1, int last1, int first2, int last2, volatile K *keys, volatile K *outKeys, volatile T* values, volatile T* outValues, K maxKey)
{
    const unsigned int idx = threadIdx.x;

    volatile __shared__ K    sm_keys1[BLOCK_WIDTH];
    volatile __shared__ K    sm_keys2[BLOCK_WIDTH];
    volatile __shared__ K    sm_vals1[BLOCK_WIDTH];
    volatile __shared__ K    sm_vals2[BLOCK_WIDTH];
    volatile __shared__ K    sm_out[BLOCK_WIDTH*2];  /*write out to a larger buffer so we don't scatter to global mem*/
    volatile __shared__ K    sm_outVals[BLOCK_WIDTH*2];
    volatile __shared__ int  sm_sa[BLOCK_WIDTH];    /*need to scan all the scatter addresses to find the max*/


    __shared__ int keys1Start;  /* should always point to the beginning of the current working chunk*/
    __shared__ int writePtr;
    __shared__ int keys2Start;
    __shared__ int maxScatter;

    __shared__ int partialShift1;
    __shared__ int partialShift2;
    __shared__ int shiftAmount;
    __shared__ int done;

    if(idx == 0)
    {
        keys1Start = first1;
        keys2Start = first2;
        writePtr = first1;
        done = 0;
        
        
    }
    __syncthreads();
    

    /* load starting chunks into share mem*/

    if(idx < last1 - first1 + 1) 
    {
        sm_keys1[idx] = keys[keys1Start+idx];
        sm_vals1[idx] = values[keys1Start+idx];
    }
    else 
    {
        sm_keys1[idx] = maxKey;
        sm_vals1[idx] = -1;
    }

    
    if(idx < last2 - first2 + 1) 
    {
        sm_keys2[idx] = keys[keys2Start+idx];
        sm_vals2[idx] = values[keys2Start+idx];
    }
    else 
    {
        sm_keys2[idx] = maxKey;
        sm_vals2[idx] = -1;
    }
    __syncthreads();
    while(!done)
    {
        int numItems1 = last1 - keys1Start + 1; /* last inlcusive */
        int numItems2 = last2 - keys2Start + 1;
        if(numItems1 < 0) numItems1 = 0;
        if(numItems2 < 0) numItems2 = 0;
        /* -1 indicates that the keys are greater than the last value of the other key array */
        int sa1 = binarySearch(0, BLOCK_MAX, sm_keys2, sm_keys1[idx]);
        if(sa1 != -1) sa1 += idx;
        if(numItems1 < 0) sa1 = -1;
        
        int sa2 = binarySearch2(0, BLOCK_MAX, sm_keys1, sm_keys2[idx]);
        if(numItems2 < 0) sa2 = -1;
        if(sa2 != -1) sa2 += idx;
        
        if(sa1 != -1)
        {
            sm_out[sa1] = sm_keys1[idx];
            sm_outVals[sa1] = sm_vals1[idx];
        } 

        if(sa2 != -1)
        {
            sm_out[sa2] = sm_keys2[idx];
            sm_outVals[sa2] = sm_vals2[idx];
        }
        
        if( sm_keys1[idx] ==  maxKey) sa1 = -1;
        if( sm_keys2[idx] ==  maxKey) sa2 = -1;

        
        

        if(idx == 0)
        {
            partialShift1 = 0;
            partialShift2 = 0;
            shiftAmount = 0;
        }

        __syncthreads();

        /*Find the maximum scatter address*/    
        sm_sa[idx] = sa1;

        __syncthreads();
        
        scanBlock<RadixMaxFunctor<int>, true> (sm_sa);
        
        __syncthreads();
        
        if(idx == 0) maxScatter = sm_sa[BLOCK_MAX];
        
        __syncthreads();

        sm_sa[idx] = sa2;
        
        __syncthreads();
        
        scanBlock<RadixMaxFunctor<int>, true> (sm_sa);
        
        __syncthreads();
        
        if(idx == 0) 
        {
            if(maxScatter < sm_sa[BLOCK_MAX]) maxScatter = sm_sa[BLOCK_MAX];
        }
        
        __syncthreads();

        /* write the output back to global mem */
        if(idx <= maxScatter)
        {   
            if(writePtr + idx <= last2)
            { 
                outKeys[writePtr + idx] = sm_out[idx];
                outValues[writePtr + idx] = sm_outVals[idx];
            } 
        } 

        if(idx + BLOCK_WIDTH <= maxScatter && writePtr + idx + BLOCK_WIDTH <= last2)
        { 
            outKeys[writePtr + idx + BLOCK_WIDTH] = sm_out[idx + BLOCK_WIDTH];
            outValues[writePtr + idx + BLOCK_WIDTH] = sm_outVals[idx + BLOCK_WIDTH];   
        }
        
        if(idx == 0) 
        {
            if(keys1Start > last1 && keys2Start > last2) done = 1;
            
        }
        __syncthreads();
        /*advance the write pointer*/
        if(idx == 0 ) {writePtr += maxScatter + 1; }
        
        /*calc the shift if there is one*/
        if( sa1 == -1 && sm_keys1[idx] != maxKey) partialShift1 = 1;
        if( sa2 == -1 && sm_keys2[idx] != maxKey) partialShift2 = 1;
        
        __syncthreads();

        /*
        if(partialShift1 == 1 && partialShift2 == 1)
        {
            printf("THIS SHOULD NEVER HAPPEN\n");
        }
        */

        /* TODO: make this a 2d array and just index into the partial*/
        if(partialShift1)
        {   
            sm_sa[idx] = sa1 == -1 ? 1 : 0;
            scanBlock<RadixAddFunctor<int>,true>(sm_sa);
            /* all the -1 will be grouped at the end so find the first 1*/
            if(sm_sa[idx] == 1 ) 
            {
                shiftAmount = idx;
            }
            K shiftKey = sm_keys1[idx];
            T shiftValue = sm_vals1[idx];
            __syncthreads();
            if(idx == 0)
            {
                keys2Start += BLOCK_WIDTH;
                keys1Start += shiftAmount;
            } 
            /* perform the shift*/
            if(idx >= shiftAmount)
            {
                sm_keys1[idx - shiftAmount] = shiftKey;
                sm_vals1[idx - shiftAmount] = shiftValue;
            } 
            
            __syncthreads();
            

            /*Pull in new chunk for the other array*/
            int newIdx = (keys2Start + idx <= last2) ? keys2Start + idx : maxKey;
            
            if(newIdx != maxKey) 
            {
                sm_keys2[idx] = keys[newIdx];
                sm_vals2[idx] = values[newIdx];
             }
            else
            { 
                sm_keys2[idx] = newIdx;
                sm_vals2[idx] = -1;
            }
            
            __syncthreads();

            /*pull in new values to fill remaining empty slots */
            //newIdx = 
            int readIndex = keys1Start - shiftAmount + BLOCK_WIDTH + idx;
            int writeIndex = BLOCK_WIDTH - shiftAmount + idx;
            if (idx < shiftAmount ) 
            {
                
                if(readIndex > last1)
                { 
                    sm_keys1[writeIndex] = maxKey;
                    sm_vals1[writeIndex] = -1;

                }
                else
                {
                    sm_keys1[writeIndex] = keys[readIndex];
                    sm_vals1[writeIndex] = values[readIndex];
                }
            }
            
             __syncthreads();
             
        }
        
        if(partialShift2)
        {   
            sm_sa[idx] = sa2 == -1 ? 1 : 0;
            scanBlock<RadixAddFunctor<int>,true>(sm_sa);
            /* all the -1 will be grouped at the end so find the first 1*/
            if(sm_sa[idx] == 1 ) 
            {
                shiftAmount = idx;
            }
            K shiftKey = sm_keys2[idx];
            T shiftValue = sm_vals2[idx];
            __syncthreads();
            if(idx == 0)
            {
                keys1Start += BLOCK_WIDTH ; 
                keys2Start += shiftAmount;; 
            } 
            /* perform the shift*/
            if(idx >= shiftAmount)
            {
                sm_keys2[idx - shiftAmount] = shiftKey;
                sm_vals2[idx - shiftAmount] = shiftValue;
            } 
            
            __syncthreads();


            /*Pull in new chunk for the other array*/
            int newIdx = (keys1Start + idx <= last1) ? keys1Start + idx : maxKey;
            
            if(newIdx != maxKey) 
            {
                sm_keys1[idx] = keys[newIdx];
                sm_vals1[idx] = values[newIdx];
             }
            else
            { 
                sm_keys1[idx] = newIdx;
                sm_vals1[idx] = -1;
            }
            
            
            __syncthreads();
            /*pull in new values to fill remaining empty slots */
            int readStart = keys2Start - shiftAmount + BLOCK_WIDTH + idx;
            int writeStart = BLOCK_WIDTH - shiftAmount + idx;
            if (idx < shiftAmount ) 
            {
                
                if(readStart > last2)
                { 
                    sm_keys2[writeStart] = maxKey;
                    sm_vals2[writeStart] = -1;
                }
                else
                {
                    sm_keys2[writeStart] = keys[readStart];
                    sm_vals2[writeStart] = values[readStart];
                }
                
            }
            
            
            __syncthreads();
        }
        
        if(!partialShift1 && !partialShift2)
        {   
            if(idx == 0)
            {
                keys2Start += BLOCK_WIDTH;
                keys1Start += BLOCK_WIDTH;
            } 
            __syncthreads();
            int newIdx = (keys1Start + idx <= last1) ? keys1Start + idx : maxKey;
            if(newIdx != maxKey)
            { 
                sm_keys1[idx] = keys[newIdx];
                sm_vals1[idx] = values[newIdx];
            }
            else
            { 
                sm_keys1[idx] = newIdx;
                sm_vals1[idx] = -1;
            }

            newIdx = (keys2Start + idx <= last2) ? keys2Start + idx : maxKey;

            if(newIdx != maxKey)
            { 
                sm_keys2[idx] = keys[newIdx];
                sm_vals2[idx] = values[newIdx];
            }
            else
            { 
                sm_keys2[idx] = newIdx;
                sm_vals2[idx] = -1;
            }
           

            __syncthreads();
        }
        
        if(idx == 0) 
        {          
            if(keys1Start > last1 && keys2Start > last2) done = 1;         
        }
        __syncthreads();
    }

}

template<class K, class T>
__global__ void mergeKernel(int nItems, int chunkSize, int numChunks, int extra, K *keys,  K *outKeys,  T* values,  T* outValues, K maxKey)
{
    const int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int idx = threadIdx.x;
    const int threadID = blockId * blockDim.x + threadIdx.x;

    int chunk1 = blockId * 2;
    int chunk2 = blockId * 2 + 1;

    int first1 = chunkSize * chunk1;
    int last1  = chunkSize * chunk1 + chunkSize - 1;

    int first2 = chunkSize * chunk2;
    int last2  = chunkSize * chunk2 + chunkSize - 1;
    
    if(chunk2 == numChunks - 1 && extra !=0 ) last2 = chunkSize * chunk2 + extra - 1;
    else if(chunk1 == numChunks - 1 && extra != 0) last1 = chunkSize * chunk1 + extra - 1;
    
    if(chunk1 == numChunks - 1 && chunk2 >= numChunks)
    {
        for (int i = first1; i <= last1; i += blockDim.x)
        {
            int index = i + idx;
            if(index <= last1)
            {
                
                outKeys[index] = keys[index];
                outValues[index] = values[index];
            }
            
        }
    }

    /* if odd number and last, do nothing */
    if(chunk1 != numChunks - 1 && chunk2 < numChunks)
    {
        merge(nItems, first1, last1, first2, last2, keys, outKeys, values, outValues, maxKey);
    }
    

    

}


template<class K, class T>
__global__ void sortBlocksKernel(int nitems, volatile K *keys, volatile T *ids)
{

    const unsigned int idx = threadIdx.x;
    volatile __shared__ K sm_keys[1024];
    volatile __shared__ K sm_mask[1024];  /*current bit set*/
    volatile __shared__ K sm_nmask[1024]; /*current bit not set*/
    volatile __shared__ K sm_t[1024];
    volatile __shared__ T sm_ids[1024];
    K mask = 1;
    __shared__ int totalFalses; 
    __shared__ int lastVal;
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    int lastBlock = nitems / blockDim.x;
    int extra = nitems %  blockDim.x;
    if(extra != 0) lastBlock++;
    lastBlock--; /* get the last block id*/
    int lastIndex = 1023;
    if(blockId == lastBlock && extra != 0 ) lastIndex = extra - 1;
   

    
    const int threadID = blockId * blockDim.x + threadIdx.x;
    
    if( threadID >= nitems) return;
    
    sm_keys[idx] = keys[threadID];
    sm_ids[idx]  = ids[threadID];
    /* Change this to sizeOf(K)*8 or something, or find msb using max scan*/
    for(int i = 0; i<32; i++)
    {   
        sm_mask[idx] = sm_keys[idx] & mask;
        
        /*flip it*/
        sm_nmask[idx] = (sm_mask[idx] > 0) ? 0 : 1; 
        
        if(idx == lastIndex) lastVal = sm_nmask[lastIndex];  

        __syncthreads(); 

        /* creates the scatter indexes of the true bits */
        scanBlock<RadixAddFunctor<T>,false>(sm_nmask);

        __syncthreads();

        if(idx == lastIndex) totalFalses = sm_nmask[lastIndex] + lastVal;
       
        __syncthreads();

        /*scatter indexes of false bits*/
        sm_t[idx] = idx - sm_nmask[idx] + totalFalses;
        /*re-use nmask store the combined scatter indexes */
        sm_nmask[idx]  = (sm_mask[idx] > 0) ?  sm_t[idx] : sm_nmask[idx]; 
        /* Hold onto the old values so race conditions don't blow it away */
        T value = sm_ids[idx];
        K key   = sm_keys[idx];
        int sIndex = sm_nmask[idx]; 

        __syncthreads();

        /* scatter */
        
        sm_ids[sIndex]  = value;
        sm_keys[sIndex] = key;
        mask = mask << 1;
        __syncthreads();
    }
  
    keys[threadID] = sm_keys[idx];
    ids[threadID]  = sm_ids[idx];

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
        int *tempVals;

        int numBlocks = nitems / BLOCK_WIDTH;
        if(nitems % BLOCK_WIDTH > 0) numBlocks++;

        int numBlocksX = numBlocks;
        int numBlocksY = 1;
        if (numBlocks >= 32768)
        {
            numBlocksY = numBlocks / 32768;
            numBlocksX = (numBlocks + numBlocksY-1) / numBlocksY;
        }

        

        cout<<"Nitems NUM BLOCKS "<<numBlocksX<<endl;
        cudaMalloc((void **)&tempKeys,nitems*sizeof(int));
        CUDA_CHECK_ERROR();
        cudaMalloc((void **)&tempVals,nitems*sizeof(int));
        CUDA_CHECK_ERROR();
        //cudaMemcpy(tempFlags, &(_flags[0]),
        //           nitems*sizeof(int), cudaMemcpyDeviceToDevice);
        //CUDA_CHECK_ERROR();
        //cudaMemcpy(_values, &(_keys[0]),
        //           nitems*sizeof(int), cudaMemcpyDeviceToDevice);
        //CUDA_CHECK_ERROR();
        


        dim3 one(1,1,1);
        dim3 t(BLOCK_WIDTH,1,1);
        int numThreads = 1024;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (numBlocksX,numBlocksY, 1);

        interatorKernel<<< blocks, threads >>>(nitems, _values);
        CUDA_CHECK_ERROR();
        sortBlocksKernel<<< blocks, threads >>>(nitems, _keys, _values);
        CUDA_CHECK_ERROR();
        //printKernel<<< one, one>>>(nitems, _keys);
        int sortSize = 1024;
        int chunkSize;
        int numChunks;
        int extra;
       // int *badInput = new int[nitems];

        chunkSize = sortSize;
        numChunks = nitems / chunkSize;
        extra = nitems % chunkSize;
        if(extra != 0) numChunks++;
        int counter = 0;

        while( numChunks != 1)
        {
        
            if(counter % 2 == 0) mergeKernel<<< blocks, t>>>(nitems, chunkSize, numChunks, extra, _keys, tempKeys, _values, tempVals,std::numeric_limits<int>::max());
            else mergeKernel<<< blocks, t>>>(nitems, chunkSize, numChunks, extra, tempKeys, _keys, tempVals, _values,std::numeric_limits<int>::max());
            CUDA_CHECK_ERROR();
            chunkSize = chunkSize * 2;
            numChunks = nitems / chunkSize;
            extra = nitems % chunkSize;
            if(extra != 0) numChunks++;
            counter++;
            

        }
        
        if(counter % 2 == 1)
        {
            cout<<"Copying "<<counter%2<<endl;
            CUDA_CHECK_ERROR();
            cudaMemcpy( &(get<0>(inputs).array[0]),tempKeys,
                       nitems*sizeof(int), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ERROR();
            cudaMemcpy( &(_values[0]),tempVals,
                       nitems*sizeof(int), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ERROR();
        }   
        
        cudaFree(tempKeys);
        cudaFree(tempVals);
       

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