// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RADIX_SORT_OP_H
#define EAVL_RADIX_SORT_OP_H

#include "eavlCUDA.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlOperation.h"
#include "eavlException.h"
#include <time.h>
#include <limits>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

#define BLOCK_WIDTH 1024
#define BLOCK_MAX BLOCK_WIDTH - 1

#define R_BLOCK_WIDTH 1024
#define R_BLOCK_MAX R_BLOCK_WIDTH - 1

#define WARP_SIZE 32
#define SORT_BLOCK_SIZE 128
#define SCAN_BLOCK_SIZE 256
typedef unsigned int uint;


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

// Alternative macro to catch CUDA errors
#define CUDA_SAFE_CALL( call) do {                                            \
   cudaError err = call;                                                      \
   if (cudaSuccess != err) {                                                  \
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
           __FILE__, __LINE__, cudaGetErrorString( err) );                    \
       exit(1);                                               \
   }                                                                          \
} while (0)

// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

__device__ uint scanLSB(const uint val, uint* s_data)
{
    // Shared mem is 256 uints long, set first half to 0's
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 128 in this case

    // Unrolled scan in local memory

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx] - val;  // convert inclusive -> exclusive
}

__device__ uint4 scan4(uint4 idata, uint* ptr)
{
    uint4 val4 = idata;
    uint4 sum;

    // Scan the 4 elements in idata within this thread
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;

    return val4;
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

__global__ void radixSortBlocks(const uint nbits, const uint startbit,
                              uint4* keysOut, uint4* valuesOut,
                              uint4* keysIn,  uint4* valuesIn)
{
    __shared__ uint sMem[512];

    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    uint4 key, value;
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
        uint4 address = scan4(lsb, sMem);

        __shared__ uint numtrue;

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
            numtrue = address.w + lsb.w;
        }
        __syncthreads();

        // Determine rank -- position in the block
        // If you are a 0 --> your position is the scan of 0's
        // If you are a 1 --> your position is calculated as below
        uint4 rank;
        const int idx = tid*4;
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
        rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;

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
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters array, and the starting offsets to the
// blockOffsets array.
//
//----------------------------------------------------------------------------
__global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    extern __shared__ uint sRadix1[];

    uint groupId = blockIdx.x;
    uint localId = threadIdx.x;
    uint groupSize = blockDim.x;

    uint2 radix2;
    radix2 = keys[threadIdx.x + (blockIdx.x * blockDim.x)];

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
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
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
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------
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

    uint i = blockId * blockDim.x + threadIdx.x;

    sKeys2[threadIdx.x]   = keys[i];
    sValues2[threadIdx.x] = values[i];

    if(threadIdx.x < 16)
    {
        sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks +
                                             blockId];
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

}

__device__ uint scanLocalMem(const uint val, uint* s_data)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = threadIdx.x;
    s_data[idx] = 0.0f;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 256

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 128]; __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx-1];
}

__global__ void
scan(uint *g_odata, uint* g_idata, uint* g_blockSums, const int n,
     const bool fullBlock, const bool storeSum)
{
    __shared__ uint s_data[512];

    // Load data into shared mem
    uint4 tempData;
    uint4 threadScanT;
    uint res;
    uint4* inData  = (uint4*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
    const int i = gid * 4;

    // If possible, read from global mem in a uint4 chunk
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elems read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        threadScanT.z = tempData.z + threadScanT.y;
        threadScanT.w = tempData.w + threadScanT.z;
        res = threadScanT.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;
        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;
        res = threadScanT.w;
    }

    res = scanLocalMem(res, s_data);
    __syncthreads();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
        g_blockSums[blockIdx.x] = res + threadScanT.w;
    }

    // write results to global memory
    uint4* outData = (uint4*) g_odata;

    tempData.x = res;
    tempData.y = res + threadScanT.x;
    tempData.z = res + threadScanT.y;
    tempData.w = res + threadScanT.z;

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }
    }
}

__global__ void
vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n)
{
    __shared__ uint uni[1];

    if (threadIdx.x == 0)
    {
        uni[0] = d_uniforms[blockIdx.x];
    }

    unsigned int address = threadIdx.x + (blockIdx.x *
            blockDim.x * 4);

    __syncthreads();

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += blockDim.x;
    }
}

__global__ void printKernel(int nitems, volatile uint * ids)
{

    printf("temp keys \n");
    for( int i=0 ; i< nitems ; i++)
    {
        printf("%d ", ids[i]);
    }
    printf("\n");
}


// ****************************************************************************
// Function: radixSortStep
//
// Purpose:
//   This function performs a radix sort, using bits startbit to
//   (startbit + nbits).  It is designed to sort by 4 bits at a time.
//   It also reorders the data in the values array based on the sort.
//
// Arguments:
//      nbits: the number of key bits to use
//      startbit: the bit to start on, 0 = lsb
//      keys: the input array of keys
//      values: the input array of values
//      tempKeys: temporary storage, same size as keys
//      tempValues: temporary storage, same size as values
//      counters: storage for the index counters, used in sort
//      countersSum: storage for the sum of the counters
//      blockOffsets: storage used in sort
//      scanBlockSums: input to Scan, see below
//      numElements: the number of elements to sort
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void scanArrayRecursive(uint* outArray, uint* inArray, int numElements,
        int level, uint** blockSums)
{
    // Kernels handle 8 elems per thread
    unsigned int numBlocks = max(1,
            (unsigned int)ceil((float)numElements/(4.f*SCAN_BLOCK_SIZE)));
    unsigned int sharedEltsPerBlock = SCAN_BLOCK_SIZE * 2;
    unsigned int sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

    bool fullBlock = (numElements == numBlocks * 4 * SCAN_BLOCK_SIZE);

    dim3 grid(numBlocks, 1, 1);
    dim3 threads(SCAN_BLOCK_SIZE, 1, 1);

    // execute the scan
    if (numBlocks > 1)
    {
        scan<<<grid, threads, sharedMemSize>>>
           (outArray, inArray, blockSums[level], numElements, fullBlock, true);
    } else
    {
        scan<<<grid, threads, sharedMemSize>>>
           (outArray, inArray, blockSums[level], numElements, fullBlock, false);
    }
    if (numBlocks > 1)
    {
        scanArrayRecursive(blockSums[level], blockSums[level],
                numBlocks, level + 1, blockSums);
        vectorAddUniform4<<< grid, threads >>>
                (outArray, blockSums[level], numElements);
    }
}

void radixSortStep(uint nbits, uint startbit, uint4* keys, uint4* values,
        uint4* tempKeys, uint4* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets, uint** scanBlockSums,
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

    radixSortBlocks
        <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        (nbits, startbit, tempKeys, tempValues, keys, values);

    findRadixOffsets
        <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
        ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
         findBlocks);

    scanArrayRecursive(countersSum, counters, 16*reorderBlocks, 0,
            scanBlockSums);

    reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
        (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
        (uint2*)tempValues, blockOffsets, countersSum, counters,
        reorderBlocks);
}



__global__ void interatorKernel(int nitems, volatile uint * ids)
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
    static void call(int nitems, int,
                     IN inputs, OUT outputs, F&)
    {
                 
        uint *_keys;  //= get<0>(inputs).array;
        uint *_values; //= get<0>(outputs).array;

        //needs to be multiple of 1024;
        int extra = nitems % 1024;
        int newSize = nitems;
        uint bytes = nitems*sizeof(uint);
        dim3 one(1,1,1);

        if(extra != 0)
        {
            // if the size is not a multiple of 1024, get the padding amount
            newSize += 1024 - extra;
            bytes = newSize * sizeof(uint);
            // create new arrays
            cudaMalloc((void**)&_keys, bytes);
            CUDA_CHECK_ERROR();
            cudaMalloc((void**)&_values, bytes);
            CUDA_CHECK_ERROR();
            // copy the values over
            cout<<"Size in bytes "<<bytes<<endl;
            cudaMemcpy(_keys, get<0>(inputs).array, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ERROR();
            // pad the array with max values.
            uint * temp = &_keys[nitems];
            uint maxVal = std::numeric_limits<uint>::max();
            cudaMemset(temp, maxVal, (1024 - extra)*sizeof(uint) );
            CUDA_CHECK_ERROR();

            //printKernel<<< one, one>>>(newSize, _keys);

        }
        else
        {
            cudaMalloc((void**)&_keys, bytes);
            CUDA_CHECK_ERROR();
            cudaMalloc((void**)&_values, bytes);
            CUDA_CHECK_ERROR();

            cudaMemcpy(_keys, get<0>(inputs).array, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ERROR();
        }
        
        // Fill values with the original index position 
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
        
        interatorKernel<<< blocks, threads >>>(newSize, _values);
        CUDA_CHECK_ERROR();

        // Allocate space for block sums in the scan kernel.
        uint numLevelsAllocated = 0;
        uint maxNumScanElements = newSize;
        uint numScanElts = maxNumScanElements;
        uint level = 0;

        do
        {
            uint numBlocks = max(1, (int) ceil((float) numScanElts / (4
                    * SCAN_BLOCK_SIZE)));
            if (numBlocks > 1)
            {
                level++;
            }
            numScanElts = numBlocks;
        }
        while (numScanElts > 1);

        uint** scanBlockSums = (uint**) malloc((level + 1) * sizeof(uint*));
        assert(scanBlockSums != NULL);
        numLevelsAllocated = level + 1;
        numScanElts = maxNumScanElements;
        level = 0;

        do
        {
            uint numBlocks = max(1, (int) ceil((float) numScanElts / (4
                    * SCAN_BLOCK_SIZE)));
            if (numBlocks > 1)
            {
                // Malloc device mem for block sums
                CUDA_SAFE_CALL(cudaMalloc((void**)&(scanBlockSums[level]),
                        numBlocks*sizeof(uint)));
                level++;
            }
            numScanElts = numBlocks;
        }
        while (numScanElts > 1);

        CUDA_SAFE_CALL(cudaMalloc((void**)&(scanBlockSums[level]),
                sizeof(uint)));

        // Allocate device mem for sorting kernels
        uint *dTempKeys, *dTempVals;

        //CUDA_SAFE_CALL(cudaMalloc((void**)&dKeys, bytes));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&dVals, bytes));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dTempKeys, bytes));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dTempVals, bytes));

        // Each thread in the sort kernel handles 4 elements
        size_t numSortGroups = newSize / (4 * SORT_BLOCK_SIZE);

        uint* dCounters, *dCounterSums, *dBlockOffsets;
        CUDA_SAFE_CALL(cudaMalloc((void**)&dCounters, WARP_SIZE
                * numSortGroups * sizeof(uint)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dCounterSums, WARP_SIZE
                * numSortGroups * sizeof(uint)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dBlockOffsets, WARP_SIZE
                * numSortGroups * sizeof(uint)));


        for (int i = 0; i < 32; i += 4)
        {
            radixSortStep(4, i, (uint4*)_keys, (uint4*)_values,
                    (uint4*)dTempKeys, (uint4*)dTempVals, dCounters,
                    dCounterSums, dBlockOffsets, scanBlockSums, newSize);
        }
        //printKernel<<< one, one>>>(newSize, _keys);
        // Copy values back
        cudaMemcpy(get<0>(inputs).array, _keys, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
        CUDA_CHECK_ERROR();
        cudaMemcpy(get<0>(outputs).array, _keys, nitems*sizeof(uint), cudaMemcpyDeviceToDevice);
        CUDA_CHECK_ERROR();

        

        // Clean up
        for (int i = 0; i < numLevelsAllocated; i++)
        {
            CUDA_SAFE_CALL(cudaFree(scanBlockSums[i]));
        }

        CUDA_SAFE_CALL(cudaFree(_keys));
        CUDA_SAFE_CALL(cudaFree(_values));
        CUDA_SAFE_CALL(cudaFree(dTempKeys));
        CUDA_SAFE_CALL(cudaFree(dTempVals));
        CUDA_SAFE_CALL(cudaFree(dCounters));
        CUDA_SAFE_CALL(cudaFree(dCounterSums));
        CUDA_SAFE_CALL(cudaFree(dBlockOffsets));
    

/*
        int numBlocks = nitems / BLOCK_WIDTH;
        cout<<"Numb "<<numBlocks<<endl;
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

        cout<<"NUM BLOCKS "<<numBlocksX<<endl;
        
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
        if(rnumBlocks == 1) return;
        //printKernel<<< one, one>>>(nitems, _keys);
        //cudaDeviceSynchronize();

        int N = numBlocks;
        int T = ceil(log(double(N-1))/log(2.));
        //cout << "T="<<T<<endl;
        int c = 0;
        for (int pp = T-1; pp >= 0; --pp)
        {
            int P = 1 << pp;

            int R = 0;
            int D = P;
            for (int qq = T-1; qq >= pp ; --qq)
            {   cout<<"C "<<c<<endl;
                c++;
                int Q = 1 << qq;
                mergeKernel<<< blocks, t>>>(nitems, numBlocks, P, R, D, _keys, _values ,std::numeric_limits<int>::max());
                CUDA_CHECK_ERROR();
                D = Q - P;
                R = P;
            }
        }*/
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