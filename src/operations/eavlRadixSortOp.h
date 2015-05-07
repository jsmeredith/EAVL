// Copyright 2010-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RADIX_SORT_OP_H
#define EAVL_RADIX_SORT_OP_H

#include "eavlCUDA.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlOperation.h"
#include "eavlException.h"
#include "eavlPrefixSumOp_1.h"
#include <time.h>
#include <limits>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

#define WARP_SIZE 32
#define SORT_BLOCK_SIZE 128
#define SCAN_BLOCK_SIZE 256
typedef unsigned int uint;



/* Merge sorted lists A and B into list A. Av and Bv are then values  A must have dim >= m+n */
template<class T>
void merge(T A[], T B[], T Av[], T Bv[], int m, int n) 
{
    int i=0, j=0, k=0;
    int size = m+n;
    T *C  = (T *)malloc(size*sizeof(T));
    T *Cv = (T *)malloc(size*sizeof(T));
    while (i < m && j < n) 
    {
        if (A[i] <= B[j])
        {
          C[k]  = A[i]; 
          Cv[k] = Av[i++];  
        } 
        else
        {
          C[k]  = B[j];
          Cv[k] = Bv[j++];  
        } 
        k++;
    }
    if (i < m) for (int p = i; p < m; p++,k++)
    {
      C[k]  = A[p];
      Cv[k] = Av[p];  
    } 
    else for (int p = j; p < n; p++,k++)
    {
      C[k]  = B[p];
      Cv[k] = Bv[p]; 
    } 
    for( i=0; i<size; i++ )
    {
      A[i]  = C[i];
      Av[i] = Cv[i];  
    } 
    free(C);
    free(Cv);
}

static void insertion_sort(uint *keys, uint *values, int offset, int end) {
    int x, y;
    uint temp, tempv;
    for (x=offset; x<end; ++x) 
    {
        for (y=x; y>offset && keys[y-1]>keys[y]; y--) 
        {
            temp = keys[y];
            tempv = values[y];
            keys[y] = keys[y-1];
            values[y] = values[y-1];
            keys[y-1] = temp;
            values[y-1] = tempv;
        }
    }
}

static void radix_sort(uint *keys, uint *values, int offset, int end, int shift) {
    int x, y;
    uint value, valuev, temp, tempv;
    int last[256] = { 0 }, pointer[256];

    for (x=offset; x<end; ++x) 
    {
        ++last[(keys[x] >> shift) & 0xFF];
    }

    last[0] += offset;
    pointer[0] = offset;
    for (x=1; x<256; ++x) 
    {
        pointer[x] = last[x-1];
        last[x] += last[x-1];
    }

    for (x=0; x<256; ++x) 
    {
        while (pointer[x] != last[x]) 
        {
            value = keys[pointer[x]];
            valuev = values[pointer[x]];
            y = (value >> shift) & 0xFF;
            while (x != y) 
            {
                temp = keys[pointer[y]];
                tempv = values[pointer[y]];
                keys[pointer[y]] = value;
                values[pointer[y]++] = valuev;
                value = temp;
                valuev = tempv;
                y = (value >> shift) & 0xFF;
            }
            keys[pointer[x]] = value;
            values[pointer[x]++] = valuev;
        }
    }

    if (shift > 0) 
    {
        shift -= 8;
        for (x=0; x<256; ++x) 
        {
            temp = x > 0 ? pointer[x] - pointer[x-1] : pointer[0] - offset;
            if (temp > 64) 
            {
                radix_sort(keys, values, pointer[x] - temp, pointer[x], shift);
            } 
            else if (temp > 1) 
            {
                insertion_sort(keys, values, pointer[x] - temp, pointer[x]);
            }
        }
    }
}

/* Merges N sorted sub-sections of keys a into final, fully sorted keys a */
static void keysmerge(uint *keys, uint *values, int size, int *index, int N)
{
    int i;
    while (N > 1) 
    {
        for( i = 0; i < N; i++ ) index[i]=i*size/N; 
        index[N] = size;

#pragma omp parallel for private(i) 
        for( i=0; i<N; i+=2 ) 
        {
            merge(keys + index[i], keys + index[i+1], values + index[i], values + index[i+1],
                  index[i+1] - index[i], index[i+2] - index[i+1]);
        }
        N /= 2;
    }
}



struct eavlRadixSortOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int useValues,
                     const IN inputs, OUT outputs,
                      F&)
    {
        uint *keys = (uint*)inputs.first.array;
        uint *values = (uint*)outputs.first.array;
        if(!useValues)
        {   
            #pragma omp parallel for
            for(int i = 0; i < nitems; i++)
            {
                values[i] = i;
            }
        }
#ifdef HAVE_OPENMP
        int threads = omp_get_max_threads();
        threads = pow(2, floor(log(threads)/log(2))); // needs to be power of 2
        int *index = (int *)malloc((threads+1)*sizeof(int));
       
        for(int i = 0; i < threads; i++) index[i] = i*nitems/threads; 
        index[threads] = nitems;
        #pragma omp parallel for
        for(int i = 0; i < threads; i++) radix_sort(keys,values,index[i], index[i+1],24);
        /* Merge sorted keys pieces */
        if( threads > 1 ) keysmerge(keys,values,nitems,index,threads);
#else   
        radix_sort(keys,values,0, nitems,24);
#endif
    }
};

#if defined __CUDACC__

// Alternative macro to catch CUDA errors
#define CUDA_SAFE_CALL( call) do {                                            \
   cudaError err = call;                                                      \
   if (cudaSuccess != err) {                                                  \
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
           __FILE__, __LINE__, cudaGetErrorString( err) );                    \
       exit(1);                                                               \
   }                                                                          \
} while (0)


template<class T, int maxlevel>
__device__ T scanwarp(T val, volatile T* sData)
{
    // The following is the same as 2 * WARP_SIZE * warpId + threadInWarp = 
    // 64*(threadIdx.x >> 5) + (threadIdx.x & (WARP_SIZE - 1))
    int idx = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE - 1));
    sData[idx] = 0;
    idx += WARP_SIZE;
    T t = sData[idx] = val;

    if (0 <= maxlevel) { sData[idx] = t = t + sData[idx - 1]; }
    if (1 <= maxlevel) { sData[idx] = t = t + sData[idx - 2]; }
    if (2 <= maxlevel) { sData[idx] = t = t + sData[idx - 4]; }
    if (3 <= maxlevel) { sData[idx] = t = t + sData[idx - 8]; }
    if (4 <= maxlevel) { sData[idx] = t = t + sData[idx -16]; }

    return sData[idx] - val;  // convert inclusive -> exclusive
}
template<class T>
__device__ uint4 scan4(T idata, uint* ptr)
{    
    //extern  __shared__  uint ptr[];
    
    uint idx = threadIdx.x;

    uint4 val4 = idata;
    uint sum[3];
    sum[0] = val4.x;
    sum[1] = val4.y + sum[0];
    sum[2] = val4.z + sum[1];
    
    uint val = val4.w + sum[2];
    
    val = scanwarp<uint, 4>(val, ptr);
    __syncthreads();

    if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
    {
        ptr[idx >> 5] = val + val4.w + sum[2];
    }
    __syncthreads();

    if (idx < WARP_SIZE)
    {
        ptr[idx] = scanwarp<uint, 2>(ptr[idx], ptr);
    }
    __syncthreads();

    val += ptr[idx >> 5];

    val4.x = val;
    val4.y = val + sum[0];
    val4.z = val + sum[1];
    val4.w = val + sum[2];      
        
    return val4;
}

template <int ctasize>
__device__ uint4 rank4(uint4 preds,uint* s_data)
{
    uint4 address = scan4<uint4>(preds,s_data);  

    __shared__ uint numtrue;
    if (threadIdx.x == ctasize-1)
    {
        numtrue = address.w + preds.w;
    }
    __syncthreads();

    uint4 rank;
    uint idx = threadIdx.x << 2;
    rank.x = (preds.x) ? address.x : numtrue + idx   - address.x;
    rank.y = (preds.y) ? address.y : numtrue + idx + 1 - address.y;
    rank.z = (preds.z) ? address.z : numtrue + idx + 2 - address.z;
    rank.w = (preds.w) ? address.w : numtrue + idx + 3 - address.w;     
                
    return rank;
}
//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of 4*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------
template<bool loop>
__global__ void radixSortBlocks(const uint nbits, const uint startbit,
                                uint4* keysOut, uint4* valuesOut,
                                uint4* keysIn,  uint4* valuesIn,
                                const size_t totalBlocks)
{
    __shared__ uint sMem[512];
    uint blockId = blockIdx.x;

    uint4 key, value;
    while(!loop || blockId < totalBlocks)
    {
        // Get Indexing information
        uint i = threadIdx.x + (blockId * blockDim.x);
        uint tid = threadIdx.x;
        uint localSize = blockDim.x;

        // Load keys and vals from global memory
        key = keysIn[i];
        value = valuesIn[i];

        // For each of the 4 bits
        for(uint shift = startbit; shift < (startbit + nbits); ++shift)
        {
            // Check if the LSB is 0
            uint4 lsb;
            lsb.x = !((key.x >> shift) & 0x1);
            lsb.y = !((key.y >> shift) & 0x1);
            lsb.z = !((key.z >> shift) & 0x1);
            lsb.w = !((key.w >> shift) & 0x1);

            // Do an exclusive scan of how many elems have 0's in the LSB
            // When this is finished, address.n will contain the number of
            // elems with 0 in the LSB which precede elem n
            uint4 rank = rank4<128>(lsb, sMem);

            
            // Scatter keys into local mem
            sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
            sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
            sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
            sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
            __syncthreads();

            // Read keys out of local mem into registers, in prep for
            // write out to global mem
            key.x = sMem[tid];
            key.y = sMem[tid +     localSize];
            key.z = sMem[tid + 2 * localSize];
            key.w = sMem[tid + 3 * localSize];
            __syncthreads();

            // Scatter values into local mem
            sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
            sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
            sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
            sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
            __syncthreads();

            // Read keys out of local mem into registers, in prep for
            // write out to global mem
            value.x = sMem[tid];
            value.y = sMem[tid +     localSize];
            value.z = sMem[tid + 2 * localSize];
            value.w = sMem[tid + 3 * localSize];
            __syncthreads();
        }
        keysOut[i]   = key;
        valuesOut[i] = value;

        if(loop)
        {
            blockId += gridDim.x;
        }
        else break;
    }
}

//----------------------------------------------------------------------------
// Given an keys with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters keys, and the starting offsets to the
// blockOffsets keys.
//
//----------------------------------------------------------------------------
template<bool loop>
__global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    extern __shared__ uint sRadix1[];

    uint blockId = blockIdx.x;

    while(!loop || blockId < totalBlocks)
    {

        uint localId = threadIdx.x;
        uint groupSize = blockDim.x;

        uint2 radix2;
        radix2 = keys[threadIdx.x + (blockId * blockDim.x)];
    
        sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
        sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

        // Finds the position where the sRadix1 entries differ and stores start
        // index for each radix.
        if(localId < 16)
        {
            sStartPointers[localId] = 0;
        }
        __syncthreads();

        if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
        {
            sStartPointers[sRadix1[localId]] = localId;
        }
        if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
        {
            sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
        }
        __syncthreads();

        if(localId < 16)
        {
            blockOffsets[blockId*16 + localId] = sStartPointers[localId];
        }
        __syncthreads();

        // Compute the sizes of each block.

        if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
        {
            sStartPointers[sRadix1[localId - 1]] =
                localId - sStartPointers[sRadix1[localId - 1]];
        }
        if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
        {
            sStartPointers[sRadix1[localId + groupSize - 1]] =
                localId + groupSize - sStartPointers[sRadix1[localId +
                                                             groupSize - 1]];
        }

        if(localId == groupSize - 1)
        {
            sStartPointers[sRadix1[2 * groupSize - 1]] =
                2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
        }
        __syncthreads();

        if(localId < 16)
        {
            counters[localId * totalBlocks + blockId] = sStartPointers[localId];
        }

        if(loop)
        {
            blockId += gridDim.x;
        }
        else break;
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the keys globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------
template<bool loop>
__global__ void reorderData(uint  startbit,
                            uint  *outKeys,
                            uint  *outValues,
                            uint2 *keys,
                            uint2 *values,
                            uint  *blockOffsets,
                            uint  *offsets,
                            uint  *sizes,
                            uint  totalBlocks)
{
    uint GROUP_SIZE = blockDim.x;
    __shared__ uint2 sKeys2[256];
    __shared__ uint2 sValues2[256];
    __shared__ uint  sOffsets[16];
    __shared__ uint  sBlockOffsets[16];
    uint* sKeys1   = (uint*) sKeys2;
    uint* sValues1 = (uint*) sValues2;

    uint blockId = blockIdx.x;

    while(!loop || blockId < totalBlocks)
    {
        uint i = blockId * blockDim.x + threadIdx.x;

        sKeys2[threadIdx.x]   = keys[i];
        sValues2[threadIdx.x] = values[i];

        if(threadIdx.x < 16)
        {
            sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks + blockId];
            sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
        }
        __syncthreads();

        uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
        uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];
        
        outKeys[globalOffset]   = sKeys1[threadIdx.x];
        outValues[globalOffset] = sValues1[threadIdx.x];
        
        radix = (sKeys1[threadIdx.x + GROUP_SIZE] >> startbit) & 0xF;
        globalOffset = sOffsets[radix] + threadIdx.x + GROUP_SIZE -
                       sBlockOffsets[radix];

        outKeys[globalOffset]   = sKeys1[threadIdx.x + GROUP_SIZE];
        outValues[globalOffset] = sValues1[threadIdx.x + GROUP_SIZE];

        if(loop)
        {
            blockId += gridDim.x;
            __syncthreads();
        }
        else break;
    }
}


template <class T>
__device__ void storeSharedChunkToMem4(T   *d_out,
                                       T   threadScan[2][4],
                                       T   *s_in,
                                       int numElements, 
                                       int oDataOffset,
                                       int ai, 
                                       int bi, 
                                       int aiDev, 
                                       int biDev,
                                       bool fullBlock)
{

    // Convert to 4-vector
    uint4 tempData;
    uint4* outData = (uint4*)d_out;

    // write results to global memory


    T temp;
    temp = s_in[ai]; 


    tempData.x = temp;
    tempData.y = temp + threadScan[0][0];
    tempData.z = temp + threadScan[0][1];
    tempData.w = temp + threadScan[0][2];
   

    int i = aiDev * 4;
    if (fullBlock || i + 3 < numElements)
    {                       
        outData[aiDev] = tempData; 
    }
    else 
    {       
        // we can't use vec4 because the original keys isn't a multiple of 
        // 4 elements
        if ( i    < numElements) { d_out[i]   = tempData.x;
        if ((i+1) < numElements) { d_out[i+1] = tempData.y;
        if ((i+2) < numElements) { d_out[i+2] = tempData.z; } } }
    }

    temp       = s_in[bi]; 


    tempData.x = temp;
    tempData.y = temp + threadScan[1][0];
    tempData.z = temp + threadScan[1][1];
    tempData.w = temp + threadScan[1][2];
   

    i = biDev * 4;
    if (fullBlock || i + 3 < numElements)
    {
        outData[biDev] = tempData;
    }
    else 
    {
        // we can't use vec4 because the original keys isn't a multiple of 
        // 4 elements
        if ( i    < numElements) { d_out[i]   = tempData.x;
        if ((i+1) < numElements) { d_out[i+1] = tempData.y;
        if ((i+2) < numElements) { d_out[i+2] = tempData.z; } } }
    }

}

template<class T>
void radixSortStep(T nbits, uint startbit, uint4* keys, uint4* values,
        uint4* tempKeys, uint4* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets,
        uint numElements)
{
    // Threads handle either 4 or two elements each
    const size_t radixGlobalWorkSize   = numElements / 4;
    const size_t findGlobalWorkSize    = numElements / 2;
    const size_t reorderGlobalWorkSize = numElements / 2;

    // Radix kernel uses block size of 128, others use 256 (same as scan)
    const size_t radixBlocks   = radixGlobalWorkSize   / SORT_BLOCK_SIZE;
    const size_t findBlocks    = findGlobalWorkSize    / SCAN_BLOCK_SIZE;
    const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;
    //cout<<"Num Blocks "<<radixBlocks<<" "<<findBlocks<<endl;
    bool loop = radixBlocks > 65535;
    
    if(loop)
    {
        radixSortBlocks<true>
        <<<65535, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        (nbits, startbit, tempKeys, tempValues, keys, values, radixBlocks);
    }
    else
    {
        radixSortBlocks<false>
        <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        (nbits, startbit, tempKeys, tempValues, keys, values, radixBlocks);
    }

    loop = findBlocks > 65535;

    if(loop)
    {
        findRadixOffsets<true>
        <<<65535, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
        ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
         findBlocks);
    }
    else
    {
        findRadixOffsets<false>
        <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
        ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
         findBlocks);
    }
    // using the EAVL scan function.
    DummyFunctor dummy;
    gpuPrefixSumOp_1_function<DummyFunctor,uint> scanner;
    bool inclusive = false;
    scanner.call( (int)16*reorderBlocks, inclusive,
                   counters, 1, 1e9, 1, 0,
                   countersSum, 1 , 0, dummy);
    
    if(loop)
    {
        reorderData<true>
        <<<65535, SCAN_BLOCK_SIZE>>>
        (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
        (uint2*)tempValues, blockOffsets, countersSum, counters,
        reorderBlocks);
    }
    else
    {
        reorderData<false>
        <<<reorderBlocks, SCAN_BLOCK_SIZE>>>
        (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
        (uint2*)tempValues, blockOffsets, countersSum, counters,
        reorderBlocks);
    }
    
}


template<class T>
__global__ void interatorKernel(T nitems, volatile uint * ids)
{
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadID = blockId * blockDim.x + threadIdx.x;

    if(threadID > nitems) return;
    ids[threadID] = threadID;
}

struct eavlRadixSortOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int useValues,
                     IN inputs, OUT outputs, F&)
    {
                 
        uint *_keys;  
        uint *_values; 

        //needs to be multiple of 1024;
        int extra = nitems % 1024;
        int newSize = nitems;
        uint bytes = nitems*sizeof(uint);
        if(extra != 0)
        {
            // if the size is not a multiple of 1024, get the padding amount
            newSize += 1024 - extra;
            bytes = newSize * sizeof(uint);
            // create new keys
            cudaMalloc((void**)&_keys, bytes);
            CUDA_CHECK_ERROR();
            cudaMalloc((void**)&_values, bytes);
            CUDA_CHECK_ERROR();
            // copy the values over
            cudaMemcpy(_keys, get<0>(inputs).array, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ERROR();
            if(useValues)
            {   
                cudaMemcpy(_values, get<0>(outputs).array, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
                CUDA_CHECK_ERROR();
            }
            
            // pad the keys with max values.
            uint * temp = &_keys[nitems];
            uint maxVal = std::numeric_limits<uint>::max();
            cudaMemset(temp, maxVal, (1024 - extra)*sizeof(uint) );
            CUDA_CHECK_ERROR();
        }
        else
        {
            cudaMalloc((void**)&_keys, bytes);
            CUDA_CHECK_ERROR();
            cudaMalloc((void**)&_values, bytes);
            CUDA_CHECK_ERROR();
            cudaMemcpy(_keys, get<0>(inputs).array, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ERROR();
            if(useValues)
            {
                cudaMemcpy(_values, get<0>(outputs).array, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
                CUDA_CHECK_ERROR();
            }
        }
        
        // Find grid and block dimensions for iterator kernel
        int numBlocks = nitems / 256;
        if(nitems % 256 > 0) numBlocks++;
        int numBlocksX = numBlocks;
        int numBlocksY = 1;
        if (numBlocks >= 32768)
        {
            numBlocksY = numBlocks / 32768;
            numBlocksX = (numBlocks + numBlocksY-1) / numBlocksY;
        }

        dim3 threads(256,   1, 1);
        dim3 blocks (numBlocksX,numBlocksY, 1);
        // Generate keys with values 0 ..numElements to find scatter positions
        if(!useValues)
        {
            interatorKernel<int><<< blocks, threads >>>(newSize, _values);
            CUDA_CHECK_ERROR();    
        }
        

        // Allocate device mem for sorting kernels
        uint *dTempKeys, *dTempVals;

        CUDA_SAFE_CALL(cudaMalloc((void**)&dTempKeys, bytes));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dTempVals, bytes));

        // Each thread in the sort kernel handles 4 elements
        size_t numSortGroups = newSize / (4 * SORT_BLOCK_SIZE);  //Num Blocks
       
        uint* dCounters, *dCounterSums, *dBlockOffsets;
        CUDA_SAFE_CALL(cudaMalloc((void**)&dCounters, WARP_SIZE
                * numSortGroups * sizeof(uint)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dCounterSums, WARP_SIZE
                * numSortGroups * sizeof(uint)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dBlockOffsets, WARP_SIZE
                * numSortGroups * sizeof(uint)));


        for (int i = 0; i < 32; i += 4)
        {
            radixSortStep<uint>(4, i, (uint4*)_keys, (uint4*)_values,
                    (uint4*)dTempKeys, (uint4*)dTempVals, dCounters,
                    dCounterSums, dBlockOffsets, newSize);
        }
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        // Copy values back
        cudaMemcpy(get<0>(inputs).array, _keys, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
        CUDA_CHECK_ERROR();
        cudaMemcpy(get<0>(outputs).array, _values, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
        CUDA_CHECK_ERROR();


        CUDA_SAFE_CALL(cudaFree(_keys));
        CUDA_SAFE_CALL(cudaFree(_values));
        CUDA_SAFE_CALL(cudaFree(dTempKeys));
        CUDA_SAFE_CALL(cudaFree(dTempVals));
        CUDA_SAFE_CALL(cudaFree(dCounters));
        CUDA_SAFE_CALL(cudaFree(dCounterSums));
        CUDA_SAFE_CALL(cudaFree(dBlockOffsets));
    
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlRadixSortOp
//
//  Purpose: to sort arrays of unsigned int keys-value pairs. A boolean flag  
//           indicates whether to use the values provide(false) or generate 
//           indexes(true) that provide the scatter postions for more complex
//           structures. See testsort.cu for example usage and benckmark versus
//           std::sort.           
//
//   Example : keys  [2 0 3 1]          
//             index [0 1 2 3] (indexes generated internally)
//             ---------------
//        out index  [1 3 0 2]
//                   [0 1 2 3]        
//
// Programmer: Matt Larsen 8/19/2014 (Cuda Kernels adapted from cudpp. CPU version 
//             adapted from Erik Gorset. See COPYRIGHT.txt )
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
    int          usevals;
  public:
    eavlRadixSortOp(I i, O o, bool genIndexes)
        : inputs(i), outputs(o), nitems(-1)
    {
        usevals = !genIndexes;
    }
    eavlRadixSortOp(I i, O o, bool genIndexes, int itemsToProcess)
        : inputs(i), outputs(o), nitems(itemsToProcess)
    {
        usevals = !genIndexes;
    }
    virtual void GoCPU()
    {
        
        int n = 0;
        if(nitems > 0) n = nitems;
        else n = inputs.first.length();
        eavlOpDispatch<eavlRadixSortOp_CPU>(n, usevals, inputs, outputs, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        
        int n=0;
        if(nitems > 0) n = nitems;
        else n = inputs.first.length();
        eavlOpDispatch<eavlRadixSortOp_GPU>(n, usevals, inputs, outputs, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O>
eavlRadixSortOp<I,O> *new_eavlRadixSortOp(I i, O o, bool genIndexes) 
{
    return new eavlRadixSortOp<I,O>(i,o, genIndexes);
}

template <class I, class O>
eavlRadixSortOp<I,O> *new_eavlRadixSortOp(I i, O o, bool genIndexes, int itemsToProcess) 
{
    return new eavlRadixSortOp<I,O>(i,o, genIndexes, itemsToProcess);
}


#endif
