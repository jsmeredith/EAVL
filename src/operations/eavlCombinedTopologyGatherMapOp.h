// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COMBINED_TOPOLOGY_MAP_OP_H
#define EAVL_COMBINED_TOPOLOGY_MAP_OP_H

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

template <class CONN>
struct eavlCombinedTopologyGatherMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN0, class IN1, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, const IN1 d_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int *fromindices = get<0>(indices).array;

        int ids[MAX_LOCAL_TOPOLOGY_IDS]; // these are effectively our src indices
        for (int destindex = 0; destindex < nitems; ++destindex)
        {
            int fromindex = fromindices[destindex];

            int nids;
            int shapeType = conn.GetElementComponents(fromindex, nids, ids);

            typename collecttype<IN1>::const_type in_d(collect(destindex, d_inputs));
            typename collecttype<OUT>::type out(collect(destindex, outputs));

            out = functor(shapeType, nids, ids, s_inputs, in_d);
        }
    }
};

#if defined __CUDACC__

template <class CONN>
struct eavlCombinedTopologyGatherMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN0, class IN1, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, const IN1 d_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        cerr << "IMPLEMENT ME!\n";
        ///\todo: implement!
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlCombinedTopologyGatherMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with
///   input arrays on both the source and destination topology
///   and with outputs on the destination topology.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************
template <class IS, class ID, class O, class INDEX, class F>
class eavlCombinedTopologyGatherMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    ID           d_inputs;
    O            outputs;
    INDEX        indices;
    F            functor;
  public:
    eavlCombinedTopologyGatherMapOp(eavlCellSet *c, eavlTopology t,
                                    IS is, ID id, O o, INDEX ind, F f)
        : cells(c), topology(t), s_inputs(is), d_inputs(id), outputs(o), indices(ind), functor(f)
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
            eavlOpDispatch<eavlCombinedTopologyGatherMapOp_CPU<eavlExplicitConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlCombinedTopologyGatherMapOp_CPU<eavlRegularConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);
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

            eavlOpDispatch<eavlCombinedTopologyGatherMapOp_GPU<eavlExplicitConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlCombinedTopologyGatherMapOp_GPU<eavlRegularConnectivity> >(n, conn, s_inputs, d_inputs, outputs, indices, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class IS, class ID, class O, class INDEX, class F>
eavlCombinedTopologyGatherMapOp<IS,ID,O,INDEX,F> *new_eavlCombinedTopologyGatherMapOp(eavlCellSet *c, eavlTopology t,
                                                                                      IS is, ID id, O o, INDEX indices, F f) 
{
    return new eavlCombinedTopologyGatherMapOp<IS,ID,O,INDEX,F>(c,t,is,id,o,indices,f);
}


#endif
