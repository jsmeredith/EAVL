// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_FLAT_ARRAY_H
#define EAVL_FLAT_ARRAY_H

#include "STL.h"
#include "eavl.h"
#include "eavlException.h"
#include "eavlCUDA.h"
#include "eavlSerialize.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <cfloat>


// ****************************************************************************
// Class:  eavlFlatArray<T>
//
// Purpose:
///   We'd like an array we can pass to a GPU kernel and have it work on
///   both host and device identically.  This is an attempt at that.
///   It relies on the fact that when a CUDA kernel is about to be
///   launched, this gets a copy made on the host, which here triggers
///   a CUDA memory copy from host to device.
///   \todo: There's a lot that could be improved here.  Maybe being
///   totally automatic isn't quite the right thing to do, as it's
///   now somewhat fragile (e.g. must copy by reference within the host
///   and only pass by value in kernel launches; the latter triggers
///   an error if you pass by ref, but the former doesn't).
//
// Programmer:  Jeremy Meredith
// Creation:    July 26, 2012
//
// Modifications:
// ****************************************************************************
template <class T>
class eavlFlatArray
{
  public:
    enum MemoryState
    {
        LAST_MODIFIED_HOST,
        LAST_MODIFIED_DEV,
        IDENTICAL_BOTH
    };
    MemoryState  state;
#ifdef HAVE_CUDA
    T           *device; ///< \todo: device memory is currently allocated of size capacity, not length; is that right?
#endif

    int          length;
    int          capacity;
    T           *host;
    bool         copied; ///< if this flag is set

  public:
    eavlFlatArray(const eavlFlatArray &a)
    {
        state =  a.state;
#ifdef HAVE_CUDA
        device = a.device;
#endif
        length = a.length;
        capacity=a.capacity;
        host   = NULL;
        copied = true;
    }
    void operator=(const eavlFlatArray &a)
    {
        if (host)
            delete[] host;
#ifdef HAVE_CUDA
        if (device)
            cudaFree(device);
#endif

        state =  a.state;
#ifdef HAVE_CUDA
        device = a.device;
#endif
        length = a.length;
        capacity=a.capacity;
        host   = NULL;
        copied = true;
    }
    eavlFlatArray(int len = 0)
    {
        if (len > 0)
        {
            host = new T[len];
        }
        else
        {
            host = NULL;
        }
        capacity = len;
        length = len;
#ifdef HAVE_CUDA
        device = NULL;
#endif
        state = LAST_MODIFIED_HOST;
        copied = false;
    }
    ~eavlFlatArray()
    {
        if (!copied)
        {
            if (host)
                delete[] host;
            host = NULL;
#ifdef HAVE_CUDA
            if (device)
                cudaFree(device);
            device = NULL;
#endif
        }
    }
    
    virtual const char *GetBasicType() const;
    virtual string className() const {return string("eavlFlatArray<")+GetBasicType() +">";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className() << state << length;
	s.write((const char *)host, sizeof(T)*length);
	s << copied;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	string nm;
	s >> nm;
	s >> state >> length;
	if (host) delete [] host;
	host = new T[length];
	s.read((char *)host, sizeof(T)*length);
	s >> copied;
	return s;
    }
    EAVL_HOSTONLY void clear()
    {
#ifdef HAVE_CUDA
        NeedOnHost();
#endif
        length = 0;
    }

    EAVL_HOSTONLY void reserve(long long newcap)
    {
#ifdef HAVE_CUDA
        NeedOnHost();
#endif
        if (capacity == newcap)
            return;

        // copy old to new
        T *newhost = new T[newcap];
        for (int i=0; i<length; ++i)
            newhost[i] = host[i];

        // make new old
        if (host)
            delete[] host;
        host = newhost;
        capacity = newcap;

#ifdef HAVE_CUDA
        if (device)
        {
            cudaFree(device);
            device = NULL;
        }
#endif
    }
    EAVL_HOSTONLY void resize(long long newlength)
    {
#ifdef HAVE_CUDA
        NeedOnHost();
#endif
        if (capacity < newlength)
            reserve(newlength);
        length = newlength;
    }
    EAVL_HOSTONLY void push_back(const T &newval)
    {
#ifdef HAVE_CUDA
        NeedOnHost();
#endif
        if (capacity <= length)
        {
            if (capacity == 0)
                reserve(8); // minimum size
            else 
                reserve(capacity * 2); // 2x exponential growth
        }
        host[length++] = newval;
    }

    /// \todo: separate NeedOnDevice to a read-only and read-write versions to
    ///        minimize bus traffic; lets us use that IDENTICAL_BOTH state.
#ifdef HAVE_CUDA
    EAVL_HOSTONLY void NeedOnDevice()
    {
        ///\todo: probably check length and realloc if it differs from last?
        if (copied)
            THROW(eavlException,"eavlFlatArray was copied by value");
        if (state == LAST_MODIFIED_HOST)
        {
            int nbytes = length * sizeof(T);
            if (!device)
            {
                cudaMalloc((void**)&device, nbytes);
                CUDA_CHECK_ERROR();
            }
            cudaMemcpy(device, &(host[0]),
                       nbytes, cudaMemcpyHostToDevice);
            CUDA_CHECK_ERROR();
        }
        state = LAST_MODIFIED_DEV;
    }
    EAVL_HOSTONLY void NeedOnHost()
    {
        if (copied)
            THROW(eavlException,"eavlFlatArray was copied by value");
        if (state == LAST_MODIFIED_DEV)
        {
            int nbytes = length * sizeof(T);
            // assert(device != NULL)
            cudaMemcpy(&(host[0]), device,
                       nbytes, cudaMemcpyDeviceToHost);
            CUDA_CHECK_ERROR();
        }
        state = LAST_MODIFIED_HOST;
    }
#endif

    EAVL_HOSTDEVICE long long size() const
    {
        return length;
    }

    ///\todo: it's dumb that even with hostonly and deviceonly
    /// we have to ifdef out one of the two.  Check if newer CUDA versions
    /// fix this.
#ifdef __CUDA_ARCH__
    EAVL_DEVICEONLY const T &operator[](int index) const
    {
        //printf("const-accessor on device, copied=%d index=%d\n",int(copied),index);
        return device[index];
    }
#else
    EAVL_HOSTONLY const inline T &operator[](int index) const
    {
        // disabled for performance temporarily;
        //if (copied)
        //    THROW(eavlException,"eavlFlatArray was copied by value");

        return host[index];
    }
#endif

#ifdef __CUDA_ARCH__
    EAVL_DEVICEONLY T &operator[](int index)
    {
        //printf("non-const-accessor on device, copied=%d index=%d\n",int(copied),index);
        return device[index];
    }
#else
    EAVL_HOSTONLY inline T &operator[](int index)
    {
        ///\todo: do we call NeedOnHost here?
        ///       I'd say let's force clients to call it manually,
        ///       save the performance penalty, and 

        // disabled for performance temporarily;
        //if (copied)
        //    THROW(eavlException,"eavlFlatArray was copied by value");

        return host[index];
    }
#endif

    static eavlFlatArray<T> * CreateObjFromName(const string &nm);

};

#endif
