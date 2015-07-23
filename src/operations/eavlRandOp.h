// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RAND_MAP_OP_H
#define EAVL_RAND_MAP_OP_H

#include "eavlCUDA.h"
#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlException.h"
#ifdef HAVE_CUDA
#include "curand_kernel.h"
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifndef DOXYGEN

struct eavlRandOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, int, const IN inputs, OUT outputs, F &functor)
    {
#pragma omp parallel for
        for (int index = 0; index < nitems; ++index)
        {
            typename collecttype<IN>::const_type in(collect(index, inputs));
            typename collecttype<OUT>::type out(collect(index, outputs));
            unsigned int RANDOM = rand();
            //out = functor(in, .5f);
            //int tid = omp_get_thread_num();
            //cerr<<index<< " "<<endl;
            // or more simply:
            collect(index, outputs) = functor(collect(index, inputs), RANDOM);
        }
     }
};


#if defined __CUDACC__




template <class F, class IN, class OUT>
__global__ void
randKernel(int nitems, const IN inputs, OUT outputs, F functor, unsigned int *rands)
{
    // int ind = threadIdx.x;
    //curandState localState = globalState[ind];
    
    //globalState[ind] = localState; 
    
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;


    for (int index = threadID; index < nitems; index += numThreads)
    { 
        //if (threadID<nitems)  
        unsigned int RANDOM = rands[index];
        //if(threadID < 100) printf("%f \n", RANDOM);
        collect(index, outputs) = functor(collect(index, inputs), RANDOM);
    }
} 

struct eavlRandOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }

    template <class F, class IN, class OUT>
    static void call(int nitems, int, const IN inputs, OUT outputs, F &functor)
    {
        int numThreads = 128;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (240,           1, 1);
        //set up random gen
        
        curandGenerator_t gen;
        unsigned int *devData;
        curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
        cudaMalloc((void **)&devData, nitems*sizeof(unsigned int));
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        curandGenerate(gen, devData, nitems);

        cudaFuncSetCacheConfig(randKernel<F, IN,OUT>, cudaFuncCachePreferL1);
        randKernel<<< blocks, threads >>>(nitems, inputs, outputs, functor, devData);
        CUDA_CHECK_ERROR();

        curandDestroyGenerator(gen);
        cudaFree(devData);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlRandOp
//
// Purpose:
///   A simple operation which takes one set of input arrays, applies a functor
///   to them, and places the results matching locations in the output arrays.
//
// Programmer:  Jeremy Meredith, Matt Larsen
// Creation:    July 25, 2013
//
// Modifications: Modified MapOp to input random seed
// ****************************************************************************
template <class I, class O, class F>
class eavlRandOp : public eavlOperation
{
  protected:
    I  inputs;
    O  outputs;
    F  functor;
    int nitems;
  public:
    eavlRandOp(I i, O o, F f) : inputs(i), outputs(o), functor(f), nitems(-1)
    {
    }
    eavlRandOp(I i, O o, F f, int nItems) : inputs(i), outputs(o), functor(f), nitems(nItems)
    {
    }
    virtual void GoCPU()
    {
        int dummy;
        int n;
        if( nitems > 0 ) n = nitems;
        else n = outputs.first.length();
        eavlOpDispatch<eavlRandOp_CPU>(n, dummy, inputs, outputs, functor);
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        int dummy;
        int n;
        if( nitems > 0 ) n = nitems;
        else n = outputs.first.length();
        eavlOpDispatch<eavlRandOp_GPU>(n, dummy, inputs, outputs, functor);
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class I, class O, class F>
eavlRandOp<I,O,F> *new_eavlRandOp(I i, O o, F f) 
{
    return new eavlRandOp<I,O,F>(i,o,f);
}
template <class I, class O, class F>
eavlRandOp<I,O,F> *new_eavlRandOp(I i, O o, F f, int itemsToProcess) 
{
    return new eavlRandOp<I,O,F>(i,o,f, itemsToProcess);
}


#endif