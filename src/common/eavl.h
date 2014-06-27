// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_H
#define EAVL_H

#include "eavlConfig.h"
#include "eavlPlatform.h"

#ifdef HAVE_CUDA

#define EAVL_FUNCTOR    __host__ __device__ inline
#define EAVL_HOSTDEVICE __host__ __device__ inline
#define EAVL_HOSTONLY   __host__
#define EAVL_DEVICEONLY __device__ inline
#else
#define EAVL_FUNCTOR    inline
#define EAVL_HOSTDEVICE inline
#define EAVL_HOSTONLY
#define EAVL_DEVICEONLY inline
#endif

typedef unsigned char byte;

struct nulltype { };
#ifdef __CUDACC__
EAVL_HOSTDEVICE const nulltype cnull() { return nulltype(); }
#else
inline const nulltype cnull() { return nulltype(); }
#endif

#define MAX_LOCAL_TOPOLOGY_IDS 12


#endif
