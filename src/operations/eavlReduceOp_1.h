// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_REDUCE_OP_1_H
#define EAVL_REDUCE_OP_1_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch_io1.h"
#include "eavlTimer.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

#ifdef HAVE_OPENMP
template <class F,
          class IO0>
struct cpuReduceOp_1_function
{
    static void call(int n, int &dummy,
                     IO0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        if (n == 0)
        {
            *o0 = functor.identity();
            return;
        }

        IO0 *tmp = NULL;
#pragma omp parallel default(none) shared(cerr,tmp,n,i0,i0div,i0mod,i0mul,i0add,o0,o0mul,o0add,functor)
        {
            int nthreads = std::min(omp_get_num_threads(), n);
            int threadid = omp_get_thread_num();
#pragma omp single
            {
                tmp = new IO0[nthreads];
                for (int i=0; i<nthreads; i++)
                {
                    int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
                    tmp[i] = i0[index_i0];
                }
            }
#pragma omp barrier

            // we might be able to change this to use a omp for directive,
            // but if so, just do nthreads to n, not strided
            for (int i=nthreads+threadid; i<n; i+=nthreads)
            {
                int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
                tmp[threadid] = functor(i0[index_i0], tmp[threadid]);
            }
#pragma omp barrier

#pragma omp single
            {
                *o0 = tmp[0];
                for (int i=1; i<nthreads; i++)
                {
                    *o0 = functor(tmp[i],*o0);
                }
            }
        }
        delete[] tmp;
    }
};
#else
template <class F,
          class IO0>
struct cpuReduceOp_1_function
{
    static void call(int n, int &dummy,
                     IO0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        if (n == 0)
        {
            *o0 = functor.identity();
            return;
        }

        *o0 = *i0;
        for (int i=1; i<n; i++)
        {
            int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
            *o0 = functor(i0[index_i0], *o0);
        }
    }
};
#endif

#if defined __CUDACC__
// Reduction Kernel
template <class F, class T, int blockSize>
__global__ void
reduceKernel_1(int n,
               const T * __restrict__ i0, int i0div, int i0mod, int i0mul, int i0add,
               T * __restrict__ o0, int o0mul, int o0add,
               F functor,
               T identity)
{
    const unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*(blockDim.x*2)) + tid;
    const unsigned int gridSize = blockDim.x*2*gridDim.x;
    
    volatile __shared__ T sdata[256];

    sdata[tid] = identity;

    // Reduce multiple elements per thread
    while (i < n)
    {
        sdata[tid] = functor(sdata[tid], i0[((i/i0div)%i0mod)*i0mul+i0add]);
        if (i+blockSize < n)
            sdata[tid] = functor(sdata[tid], i0[(((i+blockSize)/i0div)%i0mod)*i0mul+i0add]);
        i += gridSize;
    }
    __syncthreads();

    // Reduce the contents of shared memory
    // NB: This is an unrolled loop, and assumes warp-syncrhonous
    // execution.
    if (blockSize >= 512) 
    { 
        if (tid < 256) 
        { 
            sdata[tid] = functor(sdata[tid], sdata[tid + 256]);
        } 
        __syncthreads();
    }
    if (blockSize >= 256) 
    { 
        if (tid < 128) 
        { 
            sdata[tid] = functor(sdata[tid], sdata[tid + 128]);
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) 
    { 
        if (tid < 64)
        {
            sdata[tid] = functor(sdata[tid], sdata[tid + 64]);
        }
        __syncthreads(); 
    }
    if (tid < warpSize) 
    {
        // NB2: This section would also need __sync calls if warp
        // synchronous execution were not assumed
        if (blockSize >= 64) sdata[tid] = functor(sdata[tid], sdata[tid + 32]);
        if (blockSize >= 32) sdata[tid] = functor(sdata[tid], sdata[tid + 16]);
        if (blockSize >= 16) sdata[tid] = functor(sdata[tid], sdata[tid + 8]);
        if (blockSize >= 8)  sdata[tid] = functor(sdata[tid], sdata[tid + 4]);
        if (blockSize >= 4)  sdata[tid] = functor(sdata[tid], sdata[tid + 2]);
        if (blockSize >= 2)  sdata[tid] = functor(sdata[tid], sdata[tid + 1]);
    }
    
    // Write result for this block to global memory
    if (tid == 0)
    {
        o0[blockIdx.x*o0mul+o0add] = sdata[0];
    }
}

template<class T>
struct Reduce_Temp_Storage
{
    static int nvals;
    static T *device;
    static T *host;
};

template <class T> int Reduce_Temp_Storage<T>::nvals = 0;
template <class T> T *Reduce_Temp_Storage<T>::device = NULL;
template <class T> T *Reduce_Temp_Storage<T>::host = NULL;


///\todo: big question: do we WANT the reduction to put the result
/// in an output array?  or just return it to the host?
template <class F, class IO0>
struct gpuReduceOp_1_function
{
    static void call(int n, int &dummy,
                     IO0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     IO0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        int numBlocks = 64;

        if (Reduce_Temp_Storage<IO0>::nvals < numBlocks)
        {
            if (Reduce_Temp_Storage<IO0>::device)
                cudaFree(Reduce_Temp_Storage<IO0>::device);
            // allocate at least 4k
            Reduce_Temp_Storage<IO0>::nvals = (numBlocks < 4096) ? 4096 : numBlocks;
            cudaMalloc((void**)&Reduce_Temp_Storage<IO0>::device,
                       Reduce_Temp_Storage<IO0>::nvals * sizeof(IO0));
            Reduce_Temp_Storage<IO0>::host = new IO0[Reduce_Temp_Storage<IO0>::nvals];
            CUDA_CHECK_ERROR();
        }

        IO0 identity = functor.identity();
        // fixing at 256 threads
        reduceKernel_1<F,IO0,256><<<numBlocks, 256>>>(n,
                                              d_i0, i0div, i0mod, i0mul, i0add,
                                              Reduce_Temp_Storage<IO0>::device, 1, 0,
                                              functor,
                                              identity);
        CUDA_CHECK_ERROR();

        //    cudaMemcpy(Reduce_Temp_Storage<IO0>::host,
        //               Reduce_Temp_Storage<IO0>::device,
        //               numBlocks * sizeof(IO0),
        //               cudaMemcpyDeviceToHost);
        //    for (int i=0; i<numBlocks; i++)
        //        cerr << "host["<<i<<"] = "<<Reduce_Temp_Storage<IO0>::host[i]<<endl;

        reduceKernel_1<F,IO0,256><<<1, 256>>>(numBlocks,
                                              Reduce_Temp_Storage<IO0>::device, 1, 1e9, 1, 0,
                                              d_o0, o0mul, o0add,
                                              functor,
                                              identity);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlReduceOp_1
//
// Purpose:
///   A simple reduce operation; performs the given 2-input functor on an
///   input array (assuming it is associative and commutative, like addition),
///   and places the result in the first index in the output array.
//
// Programmer:  Jeremy Meredith
// Creation:    April 13, 2012
//
// Modifications: Matt Larse 10/21/2014 - added ability to process subset of
//                                        input
// ****************************************************************************
template <class F>
class eavlReduceOp_1 : public eavlOperation
{
  protected:
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    F           functor;
    int         nitems;
  public:
    eavlReduceOp_1(eavlArrayWithLinearIndex in0,
                   eavlArrayWithLinearIndex out0,
                   F f)
        : inArray0(in0), outArray0(out0), functor(f)
    {
        nitems = -1;
    }
    eavlReduceOp_1(eavlArrayWithLinearIndex in0,
                   eavlArrayWithLinearIndex out0,
                   F f, int itemsToProcess)
        : inArray0(in0), outArray0(out0), functor(f)
    {
        nitems = itemsToProcess;
    }
    virtual void GoCPU()
    {
        if(nitems < 1) nitems = inArray0.array->GetNumberOfTuples();

        int dummy;
        eavlDispatch_io1<cpuReduceOp_1_function>(nitems, eavlArray::HOST, dummy,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        if(nitems < 1) nitems = inArray0.array->GetNumberOfTuples();

        int dummy;
        eavlDispatch_io1<gpuReduceOp_1_function>(nitems, eavlArray::DEVICE, dummy,
                     inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                     outArray0.array, outArray0.mul, outArray0.add,
                     functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

#endif

