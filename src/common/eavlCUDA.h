// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CUDA_H
#define EAVL_CUDA_H

#define CUDA_CHECK_ERROR()                              \
{                                                       \
    cudaError_t err = cudaGetLastError();               \
    if (err)                                            \
        THROW(eavlException,cudaGetErrorString(err));   \
}

void eavlInitializeGPU(int devid=-1);

#endif
