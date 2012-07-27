// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_H
#define EAVL_H

#include "eavlConfig.h"

#ifdef HAVE_CUDA

#define EAVL_FUNCTOR    __host__ __device__
#define EAVL_HOSTDEVICE __host__ __device__
#define EAVL_HOSTONLY   __host__
#define EAVL_DEVICEONLY __device__
#else
#define EAVL_FUNCTOR    inline
#define EAVL_HOSTDEVICE inline
#define EAVL_HOSTONLY
#define EAVL_DEVICEONLY inline
#endif

typedef unsigned char byte;


#endif
