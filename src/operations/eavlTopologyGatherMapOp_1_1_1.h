// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TOPOLOGY_GATHER_MAP_OP_1_1_1_H
#define EAVL_TOPOLOGY_GATHER_MAP_OP_1_1_1_H

#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOperation.h"
#include "eavlOpDispatch_2_1_int.h"
#include "eavlTopology.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

///\todo: note that the eavlTopologyMapOp_0_p_q versions are just
/// like cellMapOp, so we can get rid of that one.  We just don't
/// need to do anything but get the cell shape (i.e. not the connectivity)
/// if we're not looking up anything in a source-topology array.

///\todo: HERE and for ALL kernels, use const and __restrict__ on the
/// input arrays.  Can also use __restrict__ on output arrays BUT ONLY IF
/// we only have a single output array.  It's probably not safe in the
/// event that our output arrays are strided.  (It may not be a performance
/// optimization anyway.)

#ifndef DOXYGEN
template <class F, class I0, class I1, class O0>
struct cpuTopologyGatherMapOp_1_1_1_regular
{
    static void call(int n,
                     eavlRegularConnectivity &reg,
                     I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     O0 *o0, int o0mul, int o0add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        int nodeIds[12];
        for (int i = 0; i < n; i++)
        {
            int index_idx = i*idxmul + idxadd;
            int in_index = idx[index_idx];

            int npts;
            int shapeType = reg.GetElementComponents(in_index, npts, nodeIds);

            ///\todo: really, we're EITHER doing div/mod or mul/add for each dim.
            ///       we should be able to optimize this quite a bit more....
            ///       No, that's not necessarily true; we could, in theory, have
            ///       a 3-vector on each value of a logical dimension, right?
            ///       Not that there aren't many ways to optimize this still....
            float in0[12];
            for (int nodeindex=0; nodeindex<npts; nodeindex++)
            {
                int node = nodeIds[nodeindex];
                in0[nodeindex] = i0[((node / i0div) % i0mod) * i0mul + i0add];
            }

            float in1 = i1[((i / i1div) % i1mod) * i1mul + i1add];
            o0[i * o0mul + o0add] = functor(shapeType, npts,
                                            in0, in1);
        }
    }
};

template <class F, class I0, class I1, class O0>
struct cpuTopologyGatherMapOp_1_1_1_explicit
{
    static void call(int n,
                     eavlExplicitConnectivity &conn,
                     I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     O0 *o0, int o0mul, int o0add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        int nodeIds[12];
        int npts;
        for (int i = 0; i < n; i++)
        {
            int index_idx = i*idxmul + idxadd;
            int in_index = idx[index_idx];

            int shapeType = conn.GetElementComponents(in_index, npts, nodeIds);

            /// \todo: we're converting explicitly to float here,
            /// forcing floats in the functor operator()
            float in0[12];
            for (int nodeindex=0; nodeindex<npts; nodeindex++)
            {
                int node = nodeIds[nodeindex];
                in0[nodeindex] = i0[((node / i0div) % i0mod) * i0mul + i0add];
            }

            float in1 = i1[((i / i1div) % i1mod) * i1mul + i1add];
            o0[i * o0mul + o0add] = functor(shapeType, npts,
                                            in0, in1);
        }
    }
};

#if defined __CUDACC__

template <class F, class I0, class I1, class O0>
__global__ void
topologyGatherMapKernelRegular_1_1_1(int n,
                           eavlRegularConnectivity reg,
                           I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                           I1 *i1, int i1div, int i1mod, int i1mul, int i1add,
                           O0 *o0, int o0mul, int o0add,
                           int *idx, int idxmul, int idxadd,
                           F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIds[12];
    for (int i = threadID; i < n; i += numThreads)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int npts;
        int shapeType = reg.GetElementComponents(in_index, npts, nodeIds);

        ///\todo: really, we're EITHER doing div/mod or mul/add for each dim.
        ///       we should be able to optimize this quite a bit more....
        ///       No, that's not necessarily true; we could, in theory, have
        ///       a 3-vector on each value of a logical dimension, right?
        ///       Not that there aren't many ways to optimize this still....
        float in0[12];
        for (int nodeindex=0; nodeindex<npts; nodeindex++)
        {
            int node = nodeIds[nodeindex];
            in0[nodeindex] = i0[((node / i0div) % i0mod) * i0mul + i0add];
        }

        float in1 = i1[((i / i1div) % i1mod) * i1mul + i1add];
        o0[i * o0mul + o0add] = functor(shapeType, npts,
                                        in0, in1);
    }
}

template <class F, class I0, class I1, class O0>
struct gpuTopologyGatherMapOp_1_1_1_regular
{
    static void call(int n,
                     eavlRegularConnectivity &reg,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1 *d_i1, int i1div, int i1mod, int i1mul, int i1add,
                     O0 *d_o0, int o0mul, int o0add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        topologyGatherMapKernelRegular_1_1_1<<< blocks, threads >>>(n,
                                                           reg,
                                                           d_i0, i0div, i0mod, i0mul, i0add,
                                                           d_i1, i1div, i1mod, i1mul, i1add,
                                                           d_o0, o0mul, o0add,
                                                           idx, idxmul, idxadd,
                                                           functor);
        CUDA_CHECK_ERROR();
    }
};


template <class F>
void callTopologyGatherMapKernelStructured_1_1_1(int n,
                          eavlRegularConnectivity reg,
                          eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                          eavlArray *i1, int i1div, int i1mod, int i1mul, int i1add,
                          eavlArray *o0, int o0mul, int o0add,
                          eavlArray *idx, int idxmul, int idxadd,
                          F &functor)
{
    ///\todo: assert num for all output arrays is the same?

    i0->GetCUDAArray();
    i1->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_2_1_int<gpuTopologyGatherMapOp_1_1_1_regular>(n,
                                                  eavlArray::DEVICE,
                                                  reg,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  i1, i1div, i1mod, i1mul, i1add,
                                                  o0, o0mul, o0add,
                                                  idx, idxmul, idxadd,
                                                  functor);
}

template <class F, class I0, class I1, class O0>
__global__ void
topologyGatherMapKernelExplicit_1_1_1(int n,
                             eavlExplicitConnectivity conn,
                             I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                             I1 *i1, int i1div, int i1mod, int i1mul, int i1add,
                             O0 *o0, int o0mul, int o0add,
                             int *idx, int idxmul, int idxadd,
                             F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIds[12];
    int npts;
    for (int i = threadID; i < n; i += numThreads)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int shapeType = conn.GetElementComponents(in_index, npts, nodeIds);

        /// \todo: we're converting explicitly to float here,
        /// forcing floats in the functor operator()
        float in0[12];
        for (int nodeindex=0; nodeindex<npts; nodeindex++)
        {
            int node = nodeIds[nodeindex];
            in0[nodeindex] = i0[((node / i0div) % i0mod) * i0mul + i0add];
        }

        float in1 = i1[((i / i1div) % i1mod) * i1mul + i1add];
        o0[i * o0mul + o0add] = functor(shapeType, npts,
                                        in0, in1);
    }
}


template <class F, class I0, class I1, class O0>
struct gpuTopologyGatherMapOp_1_1_1_explicit
{
    static void call(int n,
                     eavlExplicitConnectivity &conn,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1 *d_i1, int i1div, int i1mod, int i1mul, int i1add,
                     O0 *d_o0, int o0mul, int o0add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        topologyGatherMapKernelExplicit_1_1_1<<< blocks, threads >>>(n,
                                                            conn,
                                                            d_i0, i0div, i0mod, i0mul, i0add,
                                                            d_i1, i1div, i1mod, i1mul, i1add,
                                                            d_o0, o0mul, o0add,
                                                            idx, idxmul, idxadd,
                                                            functor);
        CUDA_CHECK_ERROR();
    }
};

template <class F>
void callTopologyGatherMapKernelExplicit_1_1_1(int n,
                            eavlExplicitConnectivity &conn,
                            eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                            eavlArray *i1, int i1div, int i1mod, int i1mul, int i1add,
                            eavlArray *o0, int o0mul, int o0add,
                            eavlArray *idx, int idxmul, int idxadd,
                            F &functor)
{
    ///\todo: assert num for all output arrays is the same?


    conn.shapetype.NeedOnDevice();
    conn.connectivity.NeedOnDevice();
    conn.mapCellToIndex.NeedOnDevice();

    // Send all the arrays to the device if needed; we're only doing this
    // right now to make timing the kernel itself easier later.
    i0->GetCUDAArray();
    i1->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_2_1_int<gpuTopologyGatherMapOp_1_1_1_explicit>(n,
                                                    eavlArray::DEVICE,
                                                    conn,
                                                    i0, i0div, i0mod, i0mul, i0add,
                                                    i1, i1div, i1mod, i1mul, i1add,
                                                    o0, o0mul, o0add,
                                                    idx, idxmul, idxadd,
                                                    functor);

    conn.shapetype.NeedOnHost();
    conn.connectivity.NeedOnHost();
    conn.mapCellToIndex.NeedOnHost();
}

#endif
#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlTopologyGatherMapOp_1_1_1<F>
//
// Purpose:
///   This is nearly identical to eavlTopologyMapOp, except the output
///   arrays are sparse with the original input item indices specified by
///   an additional array.  This version has 1 input array on the source
///   topology, 1 input array on the (sparsely listed) destination topology,
///   and 1 output array on the (sparsely listed) destination topology.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    March 6, 2012
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlTopologyGatherMapOp_1_1_1 : public eavlOperation
{
  protected:
    eavlCellSet     *cells;
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex inArray1;
    eavlArrayWithLinearIndex outArray0;
    eavlArrayWithLinearIndex indicesArray;
    F                functor;
    eavlTopology topology;
  public:
    eavlTopologyGatherMapOp_1_1_1(eavlCellSet *inCells,
                         eavlTopology topo,
                         eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex in1,
                         eavlArrayWithLinearIndex out0,
                         eavlArrayWithLinearIndex indices,
                         F f)
        : cells(inCells),
          topology(topo),
          inArray0(in0),
          inArray1(in1),
          outArray0(out0),
          indicesArray(indices),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        int n = outArray0.array->GetNumberOfTuples();
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            eavlDispatch_2_1_int<cpuTopologyGatherMapOp_1_1_1_explicit>(n,
                                   eavlArray::HOST,
                                   elExp->GetConnectivity(topology),
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   indicesArray.array, indicesArray.mul, indicesArray.add,
                                   functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlDispatch_2_1_int<cpuTopologyGatherMapOp_1_1_1_regular>(n,
                                 eavlArray::HOST,
                                 conn,
                                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                 inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                 outArray0.array, outArray0.mul, outArray0.add,
                                 indicesArray.array, indicesArray.mul, indicesArray.add,
                                 functor);
        }
        else
        {
            THROW(eavlException,"eavlTopologyGatherMapOp didn't understand the mesh type.");
        }
    }
    virtual void GoGPU()
    {
        int n = outArray0.array->GetNumberOfTuples();
#if defined __CUDACC__
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            callTopologyGatherMapKernelExplicit_1_1_1(n,
                                   elExp->GetConnectivity(topology),
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   indicesArray.array, indicesArray.mul, indicesArray.add,
                                   functor);
        }
        else if (elStr)
        {
            callTopologyGatherMapKernelStructured_1_1_1(n,
                                 eavlRegularConnectivity(elStr->GetRegularStructure(),topology),
                                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                 inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                 outArray0.array, outArray0.mul, outArray0.add,
                                 indicesArray.array, indicesArray.mul, indicesArray.add,
                                 functor);
        }
        else
        {
            THROW(eavlException,"eavlTopologyGatherMapOp didn't understand the mesh type.");
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
}; 

#endif
