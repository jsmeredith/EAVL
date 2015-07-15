#ifndef EAVL_COUNTING_ITERATOR_H
#define EAVL_COUNTING_ITERATOR_H

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#if defined __CUDA_ACC__

__global__ void countKernel(int nitems, int * iter)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < nitems; index += numThreads)
    { 
        iter[index] = index;
    }
} 

#endif

class eavlCountingIterator
{

	public:
		eavlCountingIterator()
		{
		}

		~eavlCountingIterator()
		{
		}

		static void generateIterator(eavlIntArray *iter)
		{
			int nitems = iter->GetNumberOfTuples();
			if(nitems > 0)
			{
#ifdef __CUDA_ACC__
				int * device = (int *) iter->GetCUDAArray();
				int numThreads = 128;
		        dim3 threads(numThreads,   1, 1);
		        dim3 blocks (240,           1, 1);
		        cudaFuncSetCacheConfig(countKernel, cudaFuncCachePreferL1);
		        countKernel<<< blocks, threads >>>(nitems, device);
		        CUDA_CHECK_ERROR();
#else 
		        int * host = (int *) iter->GetHostArray();
		        #pragma omp parallel for
		        for(int index = 0; index < nitems; index++)
		        {
		        	host[index] = index;
		        }
#endif
	    	}
		}
};

#endif