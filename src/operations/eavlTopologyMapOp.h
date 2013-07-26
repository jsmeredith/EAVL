// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TOPOLOGY_MAP_OP_H
#define EAVL_TOPOLOGY_MAP_OP_H

#include "eavlCUDA.h"
#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOpDispatch.h"
#include "eavlOperation.h"
#include "eavlTopology.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

#ifndef DOXYGEN

///\todo: the source-in + dest-in  topologymap should just be a different op from
/// the source-in only (i.e. no dest-in) topologymap
template <class CONN>
struct eavlTopologyMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN0, class IN1, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, const IN1 d_inputs, OUT outputs, F &functor)
    {
        int ids[MAX_LOCAL_TOPOLOGY_IDS];
        for (int index = 0; index < nitems; ++index)
        {
            int nids;
            int shapeType = conn.GetElementComponents(index, nids, ids);

            typename collecttype<IN1>::const_type in_d(collect(index, d_inputs));
            typename collecttype<OUT>::type out(collect(index, outputs));

            out = functor(shapeType, nids, ids, s_inputs, in_d);
        }
    }
};

template <class CONN>
struct eavlTopologyMapOp_CPU_noDIN
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs, F &functor)
    {
        int ids[MAX_LOCAL_TOPOLOGY_IDS];
        for (int index = 0; index < nitems; ++index)
        {
            int nids;
            int shapeType = conn.GetElementComponents(index, nids, ids);

            typename collecttype<OUT>::type out(collect(index, outputs));

            out = functor(shapeType, nids, ids, s_inputs);
        }
    }
};

#if defined __CUDACC__

template <class CONN>
struct eavlTopologyMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN0, class IN1, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, const IN1 d_inputs, OUT outputs, F &functor)
    {
        cerr << "IMPLEMENT ME!\n";
        ///\todo: implement!
    }
};


template <class CONN, class F, class IN, class OUT>
__global__ void
topologyMapKernel_noDIN(int nitems, CONN conn,
                        const IN s_inputs, OUT outputs, F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int ids[MAX_LOCAL_TOPOLOGY_IDS];
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int nids;
        int shapeType = conn.GetElementComponents(index, nids, ids);

        typename collecttype<OUT>::type out(collect(index, outputs));

        out = functor(shapeType, nids, ids, s_inputs);
    }
}

template <class CONN>
struct eavlTopologyMapOp_GPU_noDIN
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs, F &functor)
    {
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        topologyMapKernel_noDIN<<< blocks, threads >>>(nitems, conn,
                                                       s_inputs, outputs, functor);
        CUDA_CHECK_ERROR();
    }
};

#endif

#endif

// ****************************************************************************
// Class:  eavlTopologyMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with
///   input arrays on the either the source or destination topology
///   and with outputs on the destination topology.
//
// Programmer:  Jeremy Meredith
// Creation:    July 26, 2013
//
// Modifications:
// ****************************************************************************
template <class IS, class ID, class O, class F>
class eavlTopologyMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    ID           d_inputs;
    O            outputs;
    F            functor;
  public:
    eavlTopologyMapOp(eavlCellSet *c, eavlTopology t,
                      IS is, ID id, O o, F f)
        : cells(c), topology(t), s_inputs(is), d_inputs(id), outputs(o), functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.array->GetNumberOfTuples();
        if (elExp)
        {
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);
            eavlOpDispatch<eavlTopologyMapOp_CPU<eavlExplicitConnectivity> >(n, conn, s_inputs, d_inputs, outputs, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlTopologyMapOp_CPU<eavlRegularConnectivity> >(n, conn, s_inputs, d_inputs, outputs, functor);
        }
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.array->GetNumberOfTuples();
        if (elExp)
        {
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);

            conn.shapetype.NeedOnDevice();
            conn.connectivity.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();

            eavlOpDispatch<eavlTopologyMapOp_GPU<eavlExplicitConnectivity> >(n, conn, s_inputs, d_inputs, outputs, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlTopologyMapOp_GPU<eavlRegularConnectivity> >(n, conn, s_inputs, d_inputs, outputs, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// specialization without destination-topology inputs
template <class IS, class O, class F>
class eavlTopologyMapOp<IS,nulltype,O,F> : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    O            outputs;
    F            functor;
  public:
    eavlTopologyMapOp(eavlCellSet *c, eavlTopology t,
                      IS is, O o, F f)
        : cells(c), topology(t), s_inputs(is), outputs(o), functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.array->GetNumberOfTuples();
        if (elExp)
        {
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);
            eavlOpDispatch<eavlTopologyMapOp_CPU_noDIN<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlTopologyMapOp_CPU_noDIN<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, functor);
        }
    }
    virtual void GoGPU()
    {
#ifdef HAVE_CUDA
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        int n = outputs.first.array->GetNumberOfTuples();
        if (elExp)
        {
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);

            conn.shapetype.NeedOnDevice();
            conn.connectivity.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();

            eavlOpDispatch<eavlTopologyMapOp_GPU_noDIN<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlTopologyMapOp_GPU_noDIN<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class IS, class ID, class O, class F>
eavlTopologyMapOp<IS,ID,O,F> *new_eavlTopologyMapOp(eavlCellSet *c, eavlTopology t,
                                                    IS is, ID id, O o, F f) 
{
    return new eavlTopologyMapOp<IS,ID,O,F>(c,t,is,id,o,f);
}


template <class IS, class O, class F>
eavlTopologyMapOp<IS,nulltype,O,F> *new_eavlTopologyMapOp(eavlCellSet *c, eavlTopology t,
                                                    IS is, O o, F f) 
{
    return new eavlTopologyMapOp<IS,nulltype,O,F>(c,t,is,o,f);
}


#endif
