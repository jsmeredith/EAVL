// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_NEW_ISO_TABLES_H
#define EAVL_NEW_ISO_TABLES_H

#include "eavlArray.h"
#include "eavl.h"

template <class T>
class eavlConstArray
{
  public:
    T *host;
    T *device;
  public:
    eavlConstArray(T *from, int N)
    {
        host = from;
        // Instead of "host=from", we could copy, but is there a point? :
        //host = new T[N];
        //for (int i=0; i<N; i++)
        //    host[i] = from[i];

#ifdef HAVE_CUDA
        int nbytes = N * sizeof(T);
        cudaMalloc((void**)&device, nbytes);
        cudaMemcpy(device, &(host[0]),
                   nbytes, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
#endif
    }
#ifdef __CUDA_ARCH__
    EAVL_DEVICEONLY const T &operator[](int index) const
    {
        return device[index];
    }
#else
    EAVL_HOSTONLY const T &operator[](int index) const
    {
        return host[index];
    }
#endif
};


extern eavlConstArray<int>  *eavlTetIsoTriStart;
extern eavlConstArray<byte> *eavlTetIsoTriCount;
extern eavlConstArray<byte> *eavlTetIsoTriGeom;

extern eavlConstArray<int>  *eavlVoxIsoTriStart;
extern eavlConstArray<byte> *eavlVoxIsoTriCount;
extern eavlConstArray<byte> *eavlVoxIsoTriGeom;

extern eavlConstArray<int>  *eavlHexIsoTriStart;
extern eavlConstArray<byte> *eavlHexIsoTriCount;
extern eavlConstArray<byte> *eavlHexIsoTriGeom;

extern eavlConstArray<int>  *eavlWdgIsoTriStart;
extern eavlConstArray<byte> *eavlWdgIsoTriCount;
extern eavlConstArray<byte> *eavlWdgIsoTriGeom;

extern eavlConstArray<int>  *eavlPyrIsoTriStart;
extern eavlConstArray<byte> *eavlPyrIsoTriCount;
extern eavlConstArray<byte> *eavlPyrIsoTriGeom;

extern eavlConstArray<int>  *eavlPyrIsoTriStart;
extern eavlConstArray<byte> *eavlPyrIsoTriCount;
extern eavlConstArray<byte> *eavlPyrIsoTriGeom;

extern eavlConstArray<int>  *eavlTriIsoLineStart;
extern eavlConstArray<byte> *eavlTriIsoLineCount;
extern eavlConstArray<byte> *eavlTriIsoLineGeom;

extern eavlConstArray<int>  *eavlQuadIsoLineStart;
extern eavlConstArray<byte> *eavlQuadIsoLineCount;
extern eavlConstArray<byte> *eavlQuadIsoLineGeom;

extern eavlConstArray<int>  *eavlPixelIsoLineStart;
extern eavlConstArray<byte> *eavlPixelIsoLineCount;
extern eavlConstArray<byte> *eavlPixelIsoLineGeom;

// Don't need beam iso points; it's so trivial it's actually
// easier (and obviously faster to execute) if we hardcode
// it intead of using a lookup table.
//extern eavlConstArray<int>  *eavlBeamIsoPointStart;
//extern eavlConstArray<byte> *eavlBeamIsoPointCount;
//extern eavlConstArray<byte> *eavlBeamIsoPointGeom;


void eavlInitializeIsoTables();

#endif
