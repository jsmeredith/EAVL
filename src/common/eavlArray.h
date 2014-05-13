// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ARRAY_H
#define EAVL_ARRAY_H

#include "STL.h"
#include "eavl.h"
#include "eavlException.h"
#include "eavlCUDA.h"
#include "eavlSerialize.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>

//#define DEBUG_ARRAY_TRANSFERS

// ****************************************************************************
// Class:  eavlArray
//
// Purpose:
///   Abstract base class for a variable-length, multiple-component array
///   of primitive types.  Subclasses handle specific types.
///   \todo: Maybe we DON'T need a name in here?  Upside is it's associated
///   most directly with the data.  Downside is that most things that use
///   arrays may want their own name anyway, e.g. a map<> in eavlField
///   or something.....
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 14, 2011
//
// Modifications:
// ****************************************************************************
class eavlArray
{
  protected:
    string        name;
    int           ncomponents;
  public:
    eavlArray(const string &n,     ///< name
              int nc = 1)          ///< number of components
        : name(n)
    {
        SetNumberOfComponents(nc);
    }
    virtual ~eavlArray() { }
    virtual string className() const {return "eavlArray";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << name << ncomponents;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	s >> name >> ncomponents;
	return s;
    }

    const string &GetName() const
    {
        return name;
    }
    void SetName(const string &n)
    {
        name = n;
    }
    virtual eavlArray *Create(const string &n, int nc = 1, int nt = 0) = 0;
    virtual const char *GetBasicType() const = 0;
    virtual void   SetNumberOfTuples(int) = 0;
    virtual int    GetNumberOfTuples() const = 0;
    virtual double GetComponentAsDouble(
                                      int i,  ///< tuple index
                                      int c   ///< component index
                                      ) = 0;
    virtual void   SetComponentFromDouble(int i, int c, double v) = 0;

    enum Location { HOST, DEVICE };
#ifdef HAVE_CUDA
    virtual void *GetCUDAArray() = 0;
#else
    virtual void *GetCUDAArray() {THROW(eavlException,"CUDA not available");}
#endif
    virtual void *GetHostArray() = 0;
    ///\todo: Refresh is a little odd; we're using it for CUDA-based
    /// in situ where we need some way of forcing it to assume the 
    /// device data has been updated and force new data back to the host.
    virtual void MarkAsDirty(Location) = 0;
    void *GetRawPointer(Location loc)
    {
        if (loc == HOST)
            return GetHostArray();
        else
            return GetCUDAArray();
    }

    virtual long long GetMemoryUsage()
    {
        return sizeof(string) + name.size()*sizeof(char) + sizeof(int);
    }
    int GetNumberOfComponents() const {return ncomponents;}
    double GetTupleMin(int index)
    {
        double mymin = +DBL_MAX;
        for (int j=0; j<ncomponents; j++)
        {
            double v = GetComponentAsDouble(index,j);
            if (v < mymin)
                mymin = v;
        }
        return mymin;
    }
    double GetTupleMax(int index)
    {
        double mymax = -DBL_MAX;
        for (int j=0; j<ncomponents; j++)
        {
            double v = GetComponentAsDouble(index,j);
            if (v > mymax)
                mymax = v;
        }
        return mymax;
    }
    double GetTupleMagnitude(int index)
    {
        double mymag = 0;
        for (int j=0; j<ncomponents; j++)
        {
            double v = GetComponentAsDouble(index,j);
            mymag += v*v;
        }
        return sqrt(mymag);
    }

    double GetComponentWiseMin()
    {
        int nt = GetNumberOfTuples();
        double mymin = +DBL_MAX;
        for (int i=0; i<nt; i++)
        {
            double v = GetTupleMin(i);
            if (v < mymin)
                mymin = v;
        }
        return mymin;
    }
    double GetComponentWiseMax()
    {
        int nt = GetNumberOfTuples();
        double mymax = -DBL_MAX;
        for (int i=0; i<nt; i++)
        {
            double v = GetTupleMax(i);
            if (v > mymax)
                mymax = v;
        }
        return mymax;
    }

    double GetMagnitudeMin()
    {
        int nt = GetNumberOfTuples();
        double mymin = +DBL_MAX;
        for (int i=0; i<nt; i++)
        {
            double v = GetTupleMagnitude(i);
            if (v < mymin)
                mymin = v;
        }
        return mymin;
    }
    double GetMagnitudeMax()
    {
        int nt = GetNumberOfTuples();
        double mymax = 0;
        for (int i=0; i<nt; i++)
        {
            double v = GetTupleMagnitude(i);
            if (v > mymax)
                mymax = v;
        }
        return mymax;
    }

    void PrintSummary(ostream &out)
    {
        int n = GetNumberOfTuples() * GetNumberOfComponents();
        out << GetBasicType() <<" "
            << GetName()
            <<"["<< GetNumberOfTuples() <<"]"
            <<"["<< GetNumberOfComponents()<<"] = ";
        if (n == 0)
            out << "(empty)";
        const int NV=11;
        for (int i=0; i<n; i++)
        {
            if (n <= NV)
                out << GetComponentAsDouble(i/ncomponents,i%ncomponents) << "  ";
            else
            {
                out << GetComponentAsDouble(i/ncomponents,i%ncomponents);
                if ((i>=0 && i<int(NV/2)-1) ||
                    (i>=n-int(NV/2) && i<=n-2))
                {
                    out << "  ";
                }
                else if (i == int(NV/2)-1)
                {
                    out << " ... ";
                    i = n-int(NV/2+1);
                }
            }
        }
        out << endl;
    }
    static eavlArray * CreateObjFromName(const string &nm);
  private:
    void SetNumberOfComponents(int nc)
    {
        ncomponents = nc;
    }
};

// ****************************************************************************
// Class:  eavlConcreteArray
//
// Purpose:
///   Concrete subclass of eavlArray.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 14, 2011
//
// Modifications:
//   Jeremy Meredith, Mon Feb 25 16:12:23 EST 2013
//   Change to templated type instead of code duplication.
//
//   Jeremy Meredith, Tue Feb 26 14:23:48 EST 2013
//   Allow externally-provided host arrays, e.g. for tightly-coupled in situ.
//
//   Jeremy Meredith, Mon Jul 29 16:39:37 EDT 2013
//   Allow externally-provided device arrays, for tightly-coupled in situ for
//   CUDA-based codes.  Changed method signature to specify the location.
//
// ****************************************************************************
template<class T>
class eavlConcreteArray : public eavlArray
{
  public:
    typedef T type;
  protected:
    vector<T> host_values_self;
    T *host_values_external;
    int provided_ntuples;
    bool host_provided; ///< we don't own the host array, it was given to us, and we cannot write to it
#ifdef HAVE_CUDA
    bool device_provided; ///< we don't own the dev array, it was given to us, and we cannot write to it
    bool host_dirty;
    bool device_dirty;
    T *device_values;
    void NeedToUseOnHost()
    {
        if (host_provided)
        {
            // nothing to do
            return;
        }
        if (device_dirty)
        {
            CUDA_CHECK_ERROR();
            if (device_provided && host_values_self.size() == 0)
            {
                // first time we need the device-provided array on the
                // host, we need to allocate the host array spacee.
                host_values_self.resize(ncomponents * provided_ntuples);
            }
#ifdef DEBUG_ARRAY_TRANSFERS
            cerr << "Transferring "<<name<<" array to host\n";
#endif
            int nbytes = host_values_self.size() * sizeof(T);
            cudaMemcpy(&(host_values_self[0]), device_values,
                       nbytes, cudaMemcpyDeviceToHost);
            CUDA_CHECK_ERROR();
        }
        device_dirty = false;
        host_dirty = true;
    }
    void NeedToUseOnDevice()
    {
        if (device_provided)
        {
            // nothing to do
            return;
        }
        int nbytes = host_values_self.size() * sizeof(T);
        if (device_values == NULL)
        {
            CUDA_CHECK_ERROR();
#ifdef DEBUG_ARRAY_TRANSFERS
            cerr << "Allocating "<<nbytes<<" device bytes ("<<host_values_self.size()<<" vals) for array "<<name<<endl;
#endif
            cudaMalloc((void**)&device_values, nbytes);
            //cudaThreadSynchronize();
            CUDA_CHECK_ERROR();
            // Set to zero for debugging purposes:
            //cudaMemset((void*)device_values, 0, nbytes);
            //cudaThreadSynchronize();
            //CUDA_CHECK_ERROR();
        }
        if (host_dirty)
        {
            CUDA_CHECK_ERROR();
#ifdef DEBUG_ARRAY_TRANSFERS
            cerr << "Transferring "<<name<<" array to device\n";
#endif
            cudaMemcpy(device_values, &(host_values_self[0]),
                       nbytes, cudaMemcpyHostToDevice);
            CUDA_CHECK_ERROR();
        }
        host_dirty = false;
        device_dirty = true;
    }
    void MarkAsDirty(eavlArray::Location loc)
    {
        if (loc == eavlArray::DEVICE)
        {
            device_dirty = true;
            //NeedToUseOnHost();
        }
        else  // loc == eavlArray::HOST
        {
            host_dirty = true;
            //NeedToUseOnDevice();
        }
    }

#else
    void NeedToUseOnHost() const {}
    void NeedToUseOnDevice() const {}
    void MarkAsDirty(eavlArray::Location) {}
#endif
  public:
    eavlConcreteArray(const string &n, int nc = 1, int nt = 0) : eavlArray(n,nc)
    {
        host_values_external = NULL;
        provided_ntuples = -1;
        host_provided = false;
#ifdef HAVE_CUDA
        device_provided = false;
        // the _dirty values are initialized to false because
        // the user might start writing on either host or
        // device memory; either one is a valid option.
        host_dirty = false;
        device_dirty = false;
        device_values = NULL;
#endif
        if (nt > 0)
            host_values_self.resize(ncomponents * nt);
    }
    eavlConcreteArray(eavlArray::Location loc, T *extarray,
                      const string &n, int nc, int nt) : eavlArray(n,nc)
    {
        provided_ntuples = nt;

        if (loc == eavlArray::HOST)
        {
            host_values_external = extarray;
            host_provided = true;
#ifdef HAVE_CUDA
            device_provided = false;
            // assume host array is filled with valid data; set its _dirty flag to true
            host_dirty = true;
            device_dirty = false;
            device_values = NULL;
#endif
        }
        else // loc == eavlArray::DEVICE
        {
#ifdef HAVE_CUDA
            device_values = extarray;
            host_provided = false;
            device_provided = true;
            // assume device array is filled with valid data; set its _dirty flag to true
            host_dirty = false;
            device_dirty = true;
            host_values_external = NULL;
#else
            THROW(eavlException, "Cannot provide device values without CUDA support.");
#endif
        }
    }
    virtual ~eavlConcreteArray()
    {
#ifdef HAVE_CUDA
        if (device_values)
            cudaFree(device_values);
#endif
    }
    virtual eavlArray *Create(const string &n, int nc = 1, int nt = 0)
    {
        return new eavlConcreteArray<T>(n, nc, nt);
    }
    
    virtual string className() const {return string("eavlConcreteArray<")+GetBasicType()+">";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlArray::serialize(s);
	s << provided_ntuples << host_provided;
	s << host_values_self;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlArray::deserialize(s);
	s >> provided_ntuples >> host_provided;
	s >> host_values_self;
	return s;
    }
    virtual const char *GetBasicType() const;
    virtual void *GetHostArray() ///\todo: we might like to make this return const
    {
        NeedToUseOnHost();
        if (host_provided)
            return host_values_external;
        else
            return &(host_values_self[0]);
    }
    virtual void SetNumberOfTuples(int n)
    {
        if (host_provided)
            THROW(eavlException, "Cannot resize externally-provided array");
        NeedToUseOnHost();
        host_values_self.resize(ncomponents * n);
    }
    virtual int GetNumberOfTuples() const
    {
        //NeedToUseOnHost();
        if (ncomponents == 0)
            return 0;
        if (host_provided
#ifdef HAVE_CUDA
            || device_provided
#endif
            )
            return provided_ntuples;
        else
            return host_values_self.size() / ncomponents;
    }
    void SetTuple(int index, T *v)
    {
        if (host_provided)
            THROW(eavlException, "Cannot write to externally-provided array");
        NeedToUseOnHost();
        for (int c=0; c<ncomponents; c++)
            host_values_self[index*ncomponents+c] = v[c];
    }
    const T *GetTuple(int index) // can't make this method const
    {
        NeedToUseOnHost();
        if (host_provided)
            return &(host_values_external[index*ncomponents]);
        else
            return &(host_values_self[index*ncomponents]);
    }
    T *GetTupleWritable(int index)
    {
        if (host_provided)
            THROW(eavlException, "Cannot write to externally-provided array");
        NeedToUseOnHost();
        return &(host_values_self[index*ncomponents]);
    }
    T GetValue(int index)
    {
        // assert ncomponents==1?
        NeedToUseOnHost();
        if (host_provided)
            return host_values_external[index*ncomponents+0];
        else
            return host_values_self[index*ncomponents+0];
    }
    void SetValue(int index, T v)
    {
        if (host_provided)
            THROW(eavlException, "Cannot write to externally-provided array");
        // assert ncomponents==1?
        NeedToUseOnHost();
        host_values_self[index*ncomponents+0] = v;
    }
    void AddValue(T v)
    {
        if (host_provided)
            THROW(eavlException, "Cannot write to externally-provided array");
        // assert ncomponents==1?
        NeedToUseOnHost();
        host_values_self.push_back(v);
    }
    virtual double GetComponentAsDouble(int i, int c)
    {
        NeedToUseOnHost();
        return GetTuple(i)[c];
    }
    virtual void SetComponentFromDouble(int i, int c, double v)
    {
        NeedToUseOnHost();
        GetTupleWritable(i)[c] = v;
    }
#ifdef HAVE_CUDA
    ///\todo: can we make this return const? we kind of want that
    /// if we were handed the device memory pointer.
    virtual void *GetCUDAArray()
    {
        NeedToUseOnDevice();
        return (void*)(device_values);
    }
#endif
    virtual long long GetMemoryUsage()
    {
        ///\todo: ignores device memory; is that right??
        long long mem = 0;

        mem += sizeof(vector<T>);
        mem += host_values_self.size() * sizeof(T);

#ifdef HAVE_CUDA
        mem += sizeof(bool);
        mem += sizeof(bool);
        mem += sizeof(T*);
#endif
        return mem + eavlArray::GetMemoryUsage();
    }
};


typedef eavlConcreteArray<int> eavlIntArray;
typedef eavlConcreteArray<byte> eavlByteArray;
typedef eavlConcreteArray<float> eavlFloatArray;

#endif
