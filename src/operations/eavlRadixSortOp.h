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

    if(key > array[last]) 
    {
        index = last + 1;
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
__device__ int binarySearch2(int first, int last,volatile T * array, volatile T key)
{
    int index = -99;
    bool found = false;
    bool end = false;


    int mx = last;
    int mn = first;


    if(key >= array[last])
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

#define R_BLOCK_WIDTH 512
#define R_BLOCK_MAX 511
template<class K, class T>
__device__ void merge(int nItems, int first1, int last1, int first2, int last2, volatile K *keys, volatile T* values, K maxKey)
{
    const unsigned int idx = threadIdx.x;
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    volatile __shared__ K    sm_keys1[BLOCK_WIDTH];
    volatile __shared__ K    sm_keys2[BLOCK_WIDTH];
    volatile __shared__ K    sm_vals1[BLOCK_WIDTH];
    volatile __shared__ K    sm_vals2[BLOCK_WIDTH];
    volatile __shared__ K    sm_out[BLOCK_WIDTH*2];      /*write out to a larger buffer so we don't scatter to global mem*/
    volatile __shared__ K    sm_outVals[BLOCK_WIDTH*2];

    int numItems1 = last1 - first1; 
    int numItems2 = last2 - first2;

    
    //if(idx == 0 ) printf("blockID %d first1 %d last1 %d first2 %d last2 %d \n", blockId, first1, last1, first2, last2);
    //__syncthreads();
    

    /* load chunks into share mem*/
    //printf("blockID %d idx %d less than %d\n",blockId, idx, last1 - first1 + 1);
    //printf("blockID %d idx %d less than %d\n",blockId, idx, last2 - first2 + 1);
    if(idx <= numItems1) 
    {
        sm_keys1[idx] = keys[first1 + idx];
        sm_vals1[idx] = values[first1 + idx];
    }
    else 
    {
        sm_keys1[idx] = maxKey;
        sm_vals1[idx] = -1;
    }

    
    if(idx <= numItems2 ) 
    {
        sm_keys2[idx] = keys[first2 + idx];
        sm_vals2[idx] = values[first2 + idx];
    }
    else 
    {
        sm_keys2[idx] = maxKey;
        sm_vals2[idx] = -1;
    }
    __syncthreads();

   
    /* -1 indicates that the keys are greater than the last value of the other key array */
    int sa1 = binarySearch(0, BLOCK_MAX, sm_keys2, sm_keys1[idx]);
    if(sa1 != -1) sa1 += idx;

    
    int sa2 = binarySearch2(0, BLOCK_MAX, sm_keys1, sm_keys2[idx]);
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
    //printf("Block id %d idx %d key1 %d sa1 %d key2 %d sa2 %d \n", blockId, idx, sm_keys1[idx], sa1, sm_keys2[idx], sa2);
    

    __syncthreads();
    keys[first1 + idx] = sm_out[idx];
    values[first1 + idx] = sm_outVals[idx];
    if(idx <= numItems2 )
    {
        keys[first2 + idx]   = sm_out[idx + BLOCK_WIDTH];
        values[first2 + idx] = values[idx + BLOCK_WIDTH];

    } 
   

}

template<class K, class T>
__global__ void mergeKernel(int nitems, int nBlocks, int P, int R, int D, K *keys, T* values, K maxKey)
{
    const int blockId   = blockIdx.y * gridDim.x + blockIdx.x;

    //for (int K=1; K <= N-D; ++K)
    //{
    //                if (((K-1) & P) == R)
    //                    cout << "[" <<K-1<< "," << K+D-1 << "],";
    //}

    int J = blockId + 1;
    if(J <= nBlocks - D)
    {
        if( ((J - 1) & P ) == R )
        {
            int first1 = blockId * BLOCK_WIDTH;
            int last1  = first1 + BLOCK_WIDTH - 1;

            int first2 = (J + D - 1) * BLOCK_WIDTH;
            int last2  = first2 + BLOCK_WIDTH - 1;
            if((J+D-1) == nBlocks-1) 
            {
                last2 = first2 + nitems % BLOCK_WIDTH - 1;
            } 
           // if( threadIdx.x == 0 ) printf("A %d == B %d \n",(J+D-1), nBlocks-1);
            //if( threadIdx.x == 0 ) printf("Launching [%d,%d] P %d R %d D %d extra %d \n", blockId-1, J+D-1, P, R, D, nitems % nBlocks);
            merge(nitems, first1, last1, first2, last2, keys, values, maxKey);
        }
    }
    
}


template<class K, class T>
__global__ void sortBlocksKernel(int nitems, volatile K *keys, volatile T *ids)
{

    const unsigned int idx = threadIdx.x;
    volatile __shared__ K sm_keys[R_BLOCK_WIDTH];
    volatile __shared__ K sm_mask[R_BLOCK_WIDTH];  /*current bit set*/
    volatile __shared__ K sm_nmask[R_BLOCK_WIDTH]; /*current bit not set*/
    volatile __shared__ K sm_t[R_BLOCK_WIDTH];
    volatile __shared__ T sm_ids[R_BLOCK_WIDTH];
    K mask = 1;
    __shared__ int totalFalses; 
    __shared__ int lastVal;
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    int lastBlock = nitems / blockDim.x;
    int extra = nitems %  blockDim.x;
    if(extra != 0) lastBlock++;
    lastBlock--; /* get the last block id*/
    int lastIndex = R_BLOCK_MAX;
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

        //__syncthreads();

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

        int numBlocks = nitems / BLOCK_WIDTH;
        if(nitems % BLOCK_WIDTH > 0) numBlocks++;

        
        int numBlocksX = numBlocks;
        int numBlocksY = 1;
        if (numBlocks >= 32768)
        {
            numBlocksY = numBlocks / 32768;
            numBlocksX = (numBlocks + numBlocksY-1) / numBlocksY;
        }

        int rnumBlocks = nitems / R_BLOCK_WIDTH;
        if(nitems % R_BLOCK_WIDTH > 0) rnumBlocks++;

        int rnumBlocksX = rnumBlocks;
        int rnumBlocksY = 1;
        if (rnumBlocks >= 32768)
        {
            rnumBlocksY = rnumBlocks / 32768;
            rnumBlocksX = (rnumBlocks + rnumBlocksY-1) / rnumBlocksY;
        }

        cout<<"Nitems NUM BLOCKS "<<numBlocksX<<endl;
        
        dim3 one(1,1,1);
        dim3 t(BLOCK_WIDTH,1,1);
        int numThreads = R_BLOCK_WIDTH;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (numBlocksX,numBlocksY, 1);

        dim3 rthreads(numThreads,   1, 1);
        dim3 rblocks (rnumBlocksX,rnumBlocksY, 1);


        interatorKernel<<< blocks, threads >>>(nitems, _values);
        CUDA_CHECK_ERROR();
        sortBlocksKernel<<< rblocks, threads >>>(nitems, _keys, _values);
        CUDA_CHECK_ERROR();
        //printKernel<<< one, one>>>(nitems, _keys);
        //cudaDeviceSynchronize();
        int sortSize = R_BLOCK_WIDTH;
        int chunkSize;
        int numChunks;
        int extra;
       // int *badInput = new int[nitems];

        chunkSize = sortSize;
        numChunks = nitems / chunkSize;
        extra = nitems % chunkSize;
        if(extra != 0) numChunks++;
        int counter = 0;

        int N = numBlocks;
        int T = ceil(log(double(N))/log(2.));
        //cout << "T="<<T<<endl;
        
        for (int pp = T-1; pp >= 0; --pp)
        {
            int P = 1 << pp;

            int R = 0;
            int D = P;
            for (int qq = T-1; qq >= pp ; --qq)
            {
                int Q = 1 << qq;
                //cout << "pp="<<pp<<"  qq="<<qq<<"   P="<<P<<" Q="<<Q<<": ";

                mergeKernel<<< blocks, t>>>(nitems, numBlocks, P, R, D, _keys, _values ,std::numeric_limits<int>::max());
                CUDA_CHECK_ERROR();
                //cout << endl;
                D = Q - P;
                R = P;
            }
        }

        
        //CUDA_CHECK_ERROR();
        //printKernel<<< one, one>>>(nitems, _keys);
        //cudaDeviceSynchronize();
            
        
       
       

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