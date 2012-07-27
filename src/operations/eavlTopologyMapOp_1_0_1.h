// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TOPOLOGY_MAP_OP_1_0_1_H
#define EAVL_TOPOLOGY_MAP_OP_1_0_1_H

#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOperation.h"
#include "eavlOpDispatch_1_1.h"
#include "eavlTopology.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

#ifndef DOXYGEN
template <class F, class I0, class O0>
struct cpu_topologyMapRegular_1_0_1
{
    static void call(int nitems,
                     eavlRegularConnectivity &reg,
                     const I0 * __restrict__ i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 * __restrict__ o0, int o0mul, int o0add,
                     F &functor)
    {
        int nodeIds[12];
        for (int index = 0; index < nitems; index++)
        {
            int npts;
            int shapeType = reg.GetElementComponents(index, npts, nodeIds);

            float in0[12];
            for (int n=0; n<npts; n++)
            {
                int node = nodeIds[n];
                in0[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
            }

            o0[index * o0mul + o0add] = functor(shapeType, npts,
                                                in0);
        }
    }
};

template <class F, class I0, class O0>
struct cpu_topologyMapExplicit_1_0_1
{
    static void call(int nitems,
                     const eavlExplicitConnectivity &conn,
                     const I0 * __restrict__ i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 * __restrict__ o0, int o0mul, int o0add,
                     F &functor)
    {
        int nodeIds[12];
        int npts;
        for (int index = 0; index < nitems; index++)
        {
            int shapeType = conn.GetElementComponents(index, npts, nodeIds);

            /// \todo: we're converting explicitly to float here,
            /// forcing floats in the functor operator()
                float in0[12];
                for (int n=0; n<npts; n++)
                {
                    int node = nodeIds[n];
                    in0[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
                }

                o0[index * o0mul + o0add] = functor(shapeType, npts,
                                                    in0);
        }
    }
};

#if defined __CUDACC__

///\todo: can we specialize for the eavlTopologyMapOp_0_p_q versions?
/// this is basically like the cellMapOp.

///\todo: HERE and for ALL kernels, use const and __restrict__ on the
/// input arrays.  Can also use __restrict__ on output arrays BUT ONLY IF
/// we only have a single output array.  It's probably not safe in the
/// event that our output arrays are strided.  (It may not be a performance
/// optimization anyway.)

template <class F, class I0, class O0>
__global__ void
topologyMapKernelRegular_1_0_1(int nitems,
                           eavlRegularConnectivity reg,
                           const I0 * __restrict__ i0, int i0div, int i0mod, int i0mul, int i0add,
                           O0 * __restrict__ o0, int o0mul, int o0add,
                           F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIds[12];
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int npts;
        int shapeType = reg.GetElementComponents(index, npts, nodeIds);

        ///\todo: really, we're EITHER doing div/mod or mul/add for each dim.
        ///       we should be able to optimize this quite a bit more....
        ///       No, that's not necessarily true; we could, in theory, have
        ///       a 3-vector on each value of a logical dimension, right?
        ///       Not that there aren't many ways to optimize this still....
        float in0[12];
        for (int n=0; n<npts; n++)
        {
            int node = nodeIds[n];
            in0[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
        }

        o0[index * o0mul + o0add] = functor(shapeType, npts,
                                            in0);
    }
}

template <class F, class I0, class O0>
struct gpuTopologyMapOp_1_0_1_regular
{
    static void call(int nitems,
                     eavlRegularConnectivity &reg,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        topologyMapKernelRegular_1_0_1<<< blocks, threads >>>(nitems,
                                                           reg,
                                                           d_i0, i0div, i0mod, i0mul, i0add,
                                                           d_o0, o0mul, o0add,
                                                           functor);
        CUDA_CHECK_ERROR();
    }
};


template <class F>
void callTopologyMapKernelStructured_1_0_1(int nitems,
                          eavlRegularConnectivity reg,
                          eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                          eavlArray *o0, int o0mul, int o0add,
                          F &functor)
{
    ///\todo: assert num for all output arrays is the same?

    i0->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_1_1<gpuTopologyMapOp_1_0_1_regular>(nitems,
                                                  eavlArray::DEVICE,
                                                  reg,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  o0, o0mul, o0add,
                                                  functor);
}

template <class F, class I0, class O0>
__global__ void
topologyMapKernelExplicit_1_0_1(int nitems,
                             eavlExplicitConnectivity conn,
                             const I0 * __restrict__ i0, int i0div, int i0mod, int i0mul, int i0add,
                             O0 * __restrict__ o0, int o0mul, int o0add,
                             F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIds[12];
    int npts;
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int shapeType = conn.GetElementComponents(index, npts, nodeIds);

        /// \todo: we're converting explicitly to float here,
        /// forcing floats in the functor operator()
        float in0[12];
        for (int n=0; n<npts; n++)
        {
            int node = nodeIds[n];
            in0[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
        }

        o0[index * o0mul + o0add] = functor(shapeType, npts,
                                            in0);
    }
}


template <class F, class I0, class O0>
struct gpuTopologyMapOp_1_0_1_explicit
{
    static void call(int nitems,
                     eavlExplicitConnectivity &conn,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        topologyMapKernelExplicit_1_0_1<<< blocks, threads >>>(nitems,
                                                            conn,
                                                            d_i0, i0div, i0mod, i0mul, i0add,
                                                            d_o0, o0mul, o0add,
                                                            functor);
        CUDA_CHECK_ERROR();
    }
};

template <class F>
void callTopologyMapKernelExplicit_1_0_1(int nitems,
                            eavlExplicitConnectivity &conn,
                            eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                            eavlArray *o0, int o0mul, int o0add,
                            F &functor)
{
    conn.shapetype.NeedOnDevice();
    conn.connectivity.NeedOnDevice();
    conn.mapCellToIndex.NeedOnDevice();

    // Send all the arrays to the device if needed; we're only doing this
    // right now to make timing the kernel itself easier later.
    i0->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_1_1<gpuTopologyMapOp_1_0_1_explicit>(nitems,
                                                    eavlArray::DEVICE,
                                                    conn,
                                                    i0, i0div, i0mod, i0mul, i0add,
                                                    o0, o0mul, o0add,
                                                    functor);

    conn.shapetype.NeedOnHost();
    conn.connectivity.NeedOnHost();
    conn.mapCellToIndex.NeedOnHost();
}

#endif
#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlTopologyMapOp_1_0_1<F>
//
// Purpose:
///   Map from one topological element in a mesh to another, with 1
///   input array on the source topology and 1 output array on the
///   destination topology.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    September 6, 2011
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlTopologyMapOp_1_0_1 : public eavlOperation
{
  protected:
    eavlCellSet     *cells;
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    F                functor;
    eavlTopology topology;
  public:
    eavlTopologyMapOp_1_0_1(eavlCellSet *inCells,
                         eavlTopology topo,
                         eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex out0,
                         F f)
        : cells(inCells),
          topology(topo),
          inArray0(in0),
          outArray0(out0),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            eavlDispatch_1_1<cpu_topologyMapExplicit_1_0_1>(outArray0.array->GetNumberOfTuples(),
                                                            eavlArray::HOST,
                                                            elExp->GetConnectivity(topology),
                                                            inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                                            outArray0.array, outArray0.mul, outArray0.add,
                                                            functor);

        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlDispatch_1_1<cpu_topologyMapRegular_1_0_1>(outArray0.array->GetNumberOfTuples(),
                                                           eavlArray::HOST,
                                                           conn,
                                                           inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                                           outArray0.array, outArray0.mul, outArray0.add,
                                                           functor);
        }
        else
        {
            THROW(eavlException,"eavlTopologyMapOp didn't understand the mesh type.");
        }
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            ///\todo: assert that div,mod are always 1,INT_MAX?
            callTopologyMapKernelExplicit_1_0_1(outArray0.array->GetNumberOfTuples(),
                                   elExp->GetConnectivity(topology),
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   functor);
        }
        else if (elStr)
        {
            ///\todo: ideas/suggestions here:
            /// 1) for a rectilinear mesh, can we just create a 2D (or ideally fake up a 3d,
            ///    too -- see http://forums.nvidia.com/index.php?showtopic=164979)
            ///    thread grid and use thread IDs .x and .y for logical index?
            ///    that obviously saves a lot of divs and mods
            /// 2) create a specialized versions without div/mod (i.e. 
            ///    for curvilinear grids).  note: this was only a 10% speedup
            ///    when I tried it.  obviously not the bottleneck right now on an 8800GTS...
            /// 3) create a specialized version for separated coords.
            ///    since the one without div/mod wasn't a big speedup, this
            ///    is may be pointless right now....
            ///\todo: assert the out arrays are not logical -- i.e. div,mod are 1,INT_MAX?
            callTopologyMapKernelStructured_1_0_1(outArray0.array->GetNumberOfTuples(),
                                 eavlRegularConnectivity(elStr->GetRegularStructure(),topology),
                                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                 outArray0.array, outArray0.mul, outArray0.add,
                                 functor);
        }
        else
        {
            THROW(eavlException,"eavlTopologyMapOp didn't understand the mesh type.");
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
}; 

#endif
