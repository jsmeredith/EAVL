// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SIMPLE_TOPOLOGY_SPARSE_MAP_OP_H
#define EAVL_SIMPLE_TOPOLOGY_SPARSE_MAP_OP_H

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
struct eavlSimpleTopologySparseMapOp_CPU
{
    static inline eavlArray::Location location() { return eavlArray::HOST; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN s_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        int *sparseindices = get<0>(indices).array;

        int ids[MAX_LOCAL_TOPOLOGY_IDS]; // these are effectively our src indices
        for (int denseindex = 0; denseindex < nitems; ++denseindex)
        {
            int sparseindex = sparseindices[denseindex];

            int nids;
            int shapeType = conn.GetElementComponents(sparseindex, nids, ids);

            typename collecttype<OUT>::type out(collect(sparseindex, outputs));

            out = functor(shapeType, nids, ids, s_inputs);
        }
    }
};

#if defined __CUDACC__

template <class CONN>
struct eavlSimpleTopologySparseMapOp_GPU
{
    static inline eavlArray::Location location() { return eavlArray::DEVICE; }
    template <class F, class IN, class OUT, class INDEX>
    static void call(int nitems, CONN &conn,
                     const IN0 s_inputs, OUT outputs,
                     INDEX indices, F &functor)
    {
        cerr << "IMPLEMENT ME!\n";
        ///\todo: implement!
    }
};


#endif

#endif

// ****************************************************************************
// Class:  eavlSimpleTopologySparseMapOp
//
// Purpose:
///   Map from one topological element in a mesh to another, with
///   input arrays on the source topology (at sparsely indexed locations)
///   and with outputs on the destination topology.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************
template <class IS, class O, class INDEX, class F>
class eavlSimpleTopologySparseMapOp : public eavlOperation
{
  protected:
    eavlCellSet *cells;
    eavlTopology topology;
    IS           s_inputs;
    O            outputs;
    INDEX        indices;
    F            functor;
  public:
    eavlSimpleTopologySparseMapOp(eavlCellSet *c, eavlTopology t,
                            IS is, O o, INDEX ind, F f)
        : cells(c), topology(t), s_inputs(is), outputs(o), indices(ind), functor(f)
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
            eavlOpDispatch<eavlSimpleTopologySparseMapOp_CPU<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, indices, functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlSimpleTopologySparseMapOp_CPU<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, indices, functor);
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

            eavlOpDispatch<eavlSimpleTopologySparseMapOp_GPU<eavlExplicitConnectivity> >(n, conn, s_inputs, outputs, indices, functor);

            conn.shapetype.NeedOnHost();
            conn.connectivity.NeedOnHost();
            conn.mapCellToIndex.NeedOnHost();
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlOpDispatch<eavlSimpleTopologySparseMapOp_GPU<eavlRegularConnectivity> >(n, conn, s_inputs, outputs, indices, functor);
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
};

// helper function for type deduction
template <class IS, class O, class INDEX, class F>
eavlSimpleTopologySparseMapOp<IS,O,INDEX,F> *new_eavlSimpleTopologySparseMapOp(eavlCellSet *c, eavlTopology t,
                                                                   IS is, O o, INDEX indices, F f) 
{
    return new eavlSimpleTopologySparseMapOp<IS,O,INDEX,F>(c,t,is,o,indices,f);
}


#endif
